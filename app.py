import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import torch
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pencarian Properti Cerdas",
    page_icon="🏠",
    layout="wide",
)

# ─────────────────────────────────────────────
# PATHS  — sesuaikan jika folder berbeda
# ─────────────────────────────────────────────
PATH_DATA       = "data/properties_ultimate.csv"
PATH_BM25       = "data/bm25_index.pkl"
PATH_EMBEDDINGS = "data/embeddings.npy"

# ─────────────────────────────────────────────
# REGEX POLA BEBAS BANJIR (dari notebook cell 30)
# ─────────────────────────────────────────────
POLA_BEBAS_BANJIR = [
    r'\bbebas\s*banjir\b', r'\banti[\s\-]?banjir\b',
    r'\btidak\s*(pernah\s*)?(kena|terkena|tergenang|kebanjiran|banjir)\b',
    r'\bnggak\s*(pernah\s*)?(banjir|kebanjiran)\b',
    r'\bgak\s*(pernah\s*)?(banjir|kebanjiran)\b',
    r'\btdk\s*(pernah\s*)?(banjir|kebanjiran)\b',
    r'\baman\s*(dari\s*)?(banjir|genangan|bencana\s*banjir)\b',
    r'\bjalan\s*tidak\s*banjir\b',
    r'\barea\s*(bebas|aman)\s*(banjir|genangan)\b',
    r'\bkawasan\s*(bebas|aman)\s*(banjir|genangan)\b',
    r'\blingkungan\s*(bebas|aman)\s*banjir\b',
    r'\bbukan\s*daerah\s*banjir\b', r'\bbukan\s*langganan\s*banjir\b',
    r'\btidak\s*langganan\s*banjir\b', r'\bnon[\s\-]?flood\b', r'\bfree\s*flood\b',
    r'\bkontur\s*tanah\s*tinggi\b', r'\bposisi\s*tanah\s*(tinggi|elevated)\b',
    r'\belevasi\s*(tinggi|baik)\b', r'\btanah\s*tinggi\b', r'\bdataran\s*tinggi\b',
    r'\bdrainase\s*(baik|lancar|bagus)\b', r'\bsaluran\s*air\s*(baik|lancar|bagus)\b',
    r'\btidak\s*tergenang\b', r'\btidak\s*ada\s*genangan\b', r'\btidak\s*pernah\s*genangan\b',
]

def cek_regex_banjir(teks):
    teks_lower = str(teks).lower()
    for pola in POLA_BEBAS_BANJIR:
        if re.search(pola, teks_lower):
            return 1
    return 0

def min_max_normalize(scores):
    if len(scores) == 0:
        return scores
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

# ─────────────────────────────────────────────
# LOAD RESOURCES  (cached — only runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model dan data, harap tunggu...")
def load_resources():
    df = pd.read_csv(PATH_DATA)

    # Build Hybrid_Bebas_Banjir if not present
    if "Hybrid_Bebas_Banjir" not in df.columns:
        df["Regex_Bebas_Banjir"]  = df["teks_gabungan"].apply(cek_regex_banjir)
        df["Hybrid_Bebas_Banjir"] = (
            (df.get("AI_Bebas_Banjir", 0) == 1) | (df["Regex_Bebas_Banjir"] == 1)
        ).astype(int)

    with open(PATH_BM25, "rb") as f:
        bm25 = pickle.load(f)

    model_sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    doc_tensor  = torch.tensor(np.load(PATH_EMBEDDINGS))

    return df, bm25, model_sbert, doc_tensor

# ─────────────────────────────────────────────
# SEARCH FUNCTION
# ─────────────────────────────────────────────
def hybrid_search(query, df, bm25, model_sbert, doc_tensor,
                  top_k=10, bm25_w=0.4, sbert_w=0.6,
                  filter_banjir=False, filter_kpr=False, filter_shm=False):
    """
    Two-Stage Hybrid Search.
    Stage 1 — Hard filter  : Bebas Banjir (Hybrid), KPR (AI), SHM (AI)
    Stage 2 — Soft ranking : 40% BM25 + 60% SBERT (configurable)
    """
    query_lower = query.lower()

    # Auto-detect from query text PLUS explicit sidebar checkboxes
    wajib_banjir = filter_banjir or "banjir" in query_lower
    wajib_kpr    = filter_kpr    or "kpr"    in query_lower
    wajib_shm    = filter_shm    or "shm"    in query_lower or "hak milik" in query_lower

    valid_indices = df.index.tolist()

    if wajib_banjir and "Hybrid_Bebas_Banjir" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "Hybrid_Bebas_Banjir"] == 1]
    if wajib_kpr and "AI_Bisa_KPR" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Bisa_KPR"] == 1]
    if wajib_shm and "AI_Legalitas_SHM" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Legalitas_SHM"] == 1]

    if not valid_indices:
        return [], 0

    # BM25
    tokenized_query = query_lower.split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    # SBERT
    query_emb    = model_sbert.encode(query, convert_to_tensor=True)
    doc_t        = doc_tensor.to(query_emb.device)
    cosine_scores = util.cos_sim(query_emb, doc_t)[0].cpu().numpy()

    # Fusion
    f_bm25 = [bm25_scores[i]  for i in valid_indices]
    f_bert = [cosine_scores[i] for i in valid_indices]

    norm_bm25 = min_max_normalize(f_bm25)
    norm_bert = min_max_normalize(f_bert)

    hybrid_scores = [(bm25_w * bm) + (sbert_w * bt)
                     for bm, bt in zip(norm_bm25, norm_bert)]

    best_local = np.argsort(hybrid_scores)[::-1][:top_k]
    results = []
    for local_idx in best_local:
        orig_idx = valid_indices[local_idx]
        results.append({
            "idx":   orig_idx,
            "score": hybrid_scores[local_idx],
            "row":   df.loc[orig_idx],
        })

    return results, len(valid_indices)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_harga(val):
    try:
        v = float(val)
        if v >= 1_000_000_000:
            return f"Rp {v/1_000_000_000:.2f} M"
        elif v >= 1_000_000:
            return f"Rp {v/1_000_000:.0f} Jt"
        else:
            return f"Rp {v:,.0f}"
    except Exception:
        return str(val)

def label_badge(label, color):
    return (f'<span style="background:{color};color:#fff;'
            f'padding:2px 8px;border-radius:12px;font-size:12px;'
            f'margin-right:4px;">{label}</span>')

# Modern modal dialog for property details
@st.dialog("✨ Eksplorasi Properti", width="large")
def show_property_modal(row):
    title = str(row.get("title", "Properti"))
    
    harga = format_harga(row.get("harga_rp", 0))
    lt = row.get("luas_tanah_m2", "-")
    lb = row.get("luas_bangunan_m2", "-")
    
    # Modern Airbnb-like header box for modal
    st.markdown(f"""
    <div style='background:linear-gradient(145deg, #1e1e1e, #2a2a2a); border-radius:16px; padding:24px; display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; border:1px solid #333; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
        <div>
            <h2 style='color:#FF5A5F; font-size:32px; margin:0; font-weight:800;'>{harga}</h2>
            <p style='color:#aaa; font-size:12px; margin:4px 0 0 0; text-transform:uppercase; letter-spacing:1px;'>"Harga Penawaran"</p>
        </div>
        <div style='display:flex; gap:32px; text-align:right;'>
            <div>
                <h3 style='margin:0; color:#fff; font-size:24px;'>{lt}</h3>
                <p style='margin:0; color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:1px;'>Luas Tanah (m²)</p>
            </div>
            <div>
                <h3 style='margin:0; color:#fff; font-size:24px;'>{lb}</h3>
                <p style='margin:0; color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:1px;'>Luas Bangunan (m²)</p>
            </div>
        </div>
    </div>
    <h3 style='color:#fff; margin-bottom: 16px;'>{title}</h3>
    """, unsafe_allow_html=True)
    
    # Priority detail from dataframe column
    full_desc = str(row.get("full_description", ""))
    if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
        full_desc = str(row.get("teks_gabungan", ""))
        if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
            full_desc = str(row.get("text_blob", ""))
            
    if full_desc:
        # Pre-wrap block to keep nicely spaced description 
        st.markdown(f"<div style='color:#ccc; font-size:15px; line-height:1.7; white-space:pre-wrap; background:#181818; padding:24px; border-radius:12px; border:1px solid #2a2a2a;'>{full_desc}</div>", unsafe_allow_html=True)
    else:
        st.info("Tidak ada deskripsi detail tambahan.")

def render_card(item, rank):
    row   = item["row"]
    score = item["score"]

    title    = str(row.get("title", "Properti"))
    harga    = format_harga(row.get("harga_rp", 0))
    lt       = row.get("luas_tanah_m2", "-")
    lb       = row.get("luas_bangunan_m2", "-")

    # Priority detail from dataframe column
    full_desc = str(row.get("full_description", ""))
    if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
        full_desc = str(row.get("teks_gabungan", ""))
        if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
            full_desc = str(row.get("text_blob", ""))
            
    snippet  = full_desc[:200] + "..." if len(full_desc) > 200 else full_desc

    badges = ""
    if row.get("Hybrid_Bebas_Banjir", 0) == 1:
        badges += label_badge("✓ Bebas Banjir", "#2B6858")
    if row.get("AI_Bisa_KPR", 0) == 1:
        badges += label_badge("✓ KPR", "#1A5276")
    if row.get("AI_Legalitas_SHM", 0) == 1:
        badges += label_badge("✓ SHM", "#935116")

    score_pct = int(score * 100)

    # Injected CSS for smooth hover expansion over snippet
    st.markdown(f"""
<style>
.property-card-{rank} {{
    border: 1px solid #333;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    background: #1c1c1c;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}}
.property-card-{rank}:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.6);
    border-color: #555;
    background: #222;
}}
.snippet-{rank} {{
    font-size: 14.5px;
    color: #999;
    line-height: 1.6;
    margin: 0;
}}
</style>
<div class="property-card-{rank}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="flex:1; padding-right:16px;">
      <p style="font-size:11px;color:#FF5A5F;margin:0 0 6px;letter-spacing:1px;font-weight:700;">#{rank} · RELEVANSI {score_pct}%</p>
      <p style="font-size:20px;font-weight:700;margin:0 0 8px;color:#eee;">{title}</p>
      <div style="margin-bottom:12px;">{badges}</div>
    </div>
    <div style="text-align:right;min-width:140px;">
      <p style="font-size:24px;font-weight:800;color:#FF5A5F;margin:0;">{harga}</p>
    </div>
  </div>
  <div style="display:flex;gap:16px;font-size:13px;color:#fff;margin-bottom:12px;opacity:0.8;">
    <span style="background:#2a2a2a; border-radius:6px; padding:4px 10px;">📐 LT: {lt} m²</span>
    <span style="background:#2a2a2a; border-radius:6px; padding:4px 10px;">🏠 LB: {lb} m²</span>
  </div>
  <p class="snippet-{rank}">{snippet}</p>
</div>
""", unsafe_allow_html=True)

    if st.button("✦ Lihat Detail Lengkap", key=f"btn_detail_{item['idx']}", type="secondary", use_container_width=True):
        show_property_modal(row)

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:4px;'>🏠 Pencarian Properti Cerdas</h1>
<p style='color:#666;font-size:15px;margin-top:0;'>
  Didukung IndoBERT · BM25 · Sentence-BERT Hybrid Search
</p>
""", unsafe_allow_html=True)

# Load
try:
    df, bm25_model, sbert_model, doc_tensor = load_resources()
    data_ok = True
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.info("Pastikan folder `data/` berisi: `properties_ultimate.csv`, `bm25_index.pkl`, `embeddings.npy`")
    data_ok = False

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filter & Pengaturan")

    st.subheader("Filter Wajib")
    cb_banjir = st.checkbox("✅ Bebas Banjir")
    cb_kpr    = st.checkbox("✅ Bisa KPR")
    cb_shm    = st.checkbox("✅ Legalitas SHM")

    st.divider()

    st.subheader("Bobot Hybrid")
    bm25_weight = st.slider("BM25 (kata kunci)", 0.0, 1.0, 0.4, 0.1,
                            help="Semakin tinggi → lebih sensitif terhadap kata kunci eksak")
    sbert_weight = round(1.0 - bm25_weight, 1)
    st.caption(f"SBERT (makna) otomatis: **{sbert_weight}**")

    st.divider()

    top_k = st.slider("Jumlah hasil", 3, 20, 10)

    st.divider()
    if data_ok:
        st.metric("Total Properti", f"{len(df):,}")
        if "Hybrid_Bebas_Banjir" in df.columns:
            st.metric("Bebas Banjir", f"{df['Hybrid_Bebas_Banjir'].sum():,}")
        if "AI_Bisa_KPR" in df.columns:
            st.metric("Bisa KPR", f"{df['AI_Bisa_KPR'].sum():,}")
        if "AI_Legalitas_SHM" in df.columns:
            st.metric("SHM", f"{df['AI_Legalitas_SHM'].sum():,}")

# ── Search Bar ───────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Ketik pencarian...",
        placeholder='cth: "Rumah mewah di Jakarta Selatan bebas banjir bisa KPR"',
        label_visibility="collapsed",
    )
with col_btn:
    cari = st.button("🔍 Cari", use_container_width=True, type="primary")

# Example queries
st.caption("💡 Contoh: &nbsp;"
           "_Rumah minimalis 2 lantai dekat MRT_ &nbsp;·&nbsp; "
           "_Townhouse mewah bebas banjir SHM_ &nbsp;·&nbsp; "
           "_Properti strategis bisa KPR Jakarta_")

st.divider()

# ── Results ──────────────────────────────────
if (cari or query) and data_ok and query.strip():
    with st.spinner("Mencari properti terbaik untuk Anda..."):
        results, total_lolos = hybrid_search(
            query, df, bm25_model, sbert_model, doc_tensor,
            top_k=top_k,
            bm25_w=bm25_weight,
            sbert_w=sbert_weight,
            filter_banjir=cb_banjir,
            filter_kpr=cb_kpr,
            filter_shm=cb_shm,
        )

    if not results:
        st.warning("😕 Tidak ada properti yang memenuhi kriteria Anda. Coba kurangi filter.")
    else:
        st.markdown(
            f"<p style='color:#666;font-size:14px;'>Ditemukan <b>{total_lolos}</b> properti "
            f"yang memenuhi filter — menampilkan <b>{len(results)}</b> terbaik.</p>",
            unsafe_allow_html=True
        )
        for rank, item in enumerate(results, 1):
            render_card(item, rank)

elif data_ok:
    st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#aaa;">
  <p style="font-size:48px;margin:0;">🏡</p>
  <p style="font-size:16px;">Masukkan pencarian di atas untuk mulai.</p>
</div>
""", unsafe_allow_html=True)