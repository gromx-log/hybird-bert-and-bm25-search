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
# PATHS
# ─────────────────────────────────────────────
PATH_DATA       = "data/properties_enriched.csv"
PATH_BM25       = "data/bm25_index.pkl"
PATH_EMBEDDINGS = "data/sbert_embeddings.npy"

# ─────────────────────────────────────────────
# REGEX POLA BEBAS BANJIR
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
# LOAD RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model dan data, harap tunggu...")
def load_resources():
    df = pd.read_csv(PATH_DATA)

    # Build Hybrid_Bebas_Banjir if not present (as fallback)
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
                  top_k=10, bm25_w=0.7, sbert_w=0.3,
                  filter_banjir=False, filter_kpr=False, filter_shm=False,
                  price_range=None, lt_range=None, lb_range=None, sort_by="Kecocokan AI"):
    """
    Two-Stage Hybrid Search with Advanced Range Filters and Sorting.
    Stage 1 — Hard filter  : Bebas Banjir (Hybrid), KPR (AI), SHM (AI) + Price, LT, LB range
    Stage 2 — Soft ranking : BM25 + SBERT (configurable weights)
    """
    query_lower = query.lower()

    # Auto-detect from query text PLUS explicit sidebar checkboxes
    wajib_banjir = filter_banjir or "banjir" in query_lower
    wajib_kpr    = filter_kpr    or "kpr"    in query_lower
    wajib_shm    = filter_shm    or "shm"    in query_lower or "hak milik" in query_lower

    valid_indices = df.index.tolist()

    # 1. Hard Filter AI
    if wajib_banjir and "Hybrid_Bebas_Banjir" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "Hybrid_Bebas_Banjir"] == 1]
    if wajib_kpr and "AI_Bisa_KPR" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Bisa_KPR"] == 1]
    if wajib_shm and "AI_Legalitas_SHM" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Legalitas_SHM"] == 1]

    # 2. Hard Filter Fisik (Harga, LT, LB)
    if price_range is not None:
        min_p, max_p = price_range
        valid_indices = [i for i in valid_indices if min_p <= df.loc[i, "harga_rp"] <= max_p]
    if lt_range is not None:
        min_lt, max_lt = lt_range
        valid_indices = [i for i in valid_indices if min_lt <= df.loc[i, "luas_tanah_m2"] <= max_lt]
    if lb_range is not None:
        min_lb, max_lb = lb_range
        valid_indices = [i for i in valid_indices if min_lb <= df.loc[i, "luas_bangunan_m2"] <= max_lb]

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

    # Build results list
    all_results = []
    for local_idx, orig_idx in enumerate(valid_indices):
        all_results.append({
            "idx":   orig_idx,
            "score": hybrid_scores[local_idx],
            "row":   df.loc[orig_idx],
        })

    # Sort results
    if sort_by == "Harga Terendah":
        all_results.sort(key=lambda x: float(x["row"].get("harga_rp", 0)))
    elif sort_by == "Harga Tertinggi":
        all_results.sort(key=lambda x: float(x["row"].get("harga_rp", 0)), reverse=True)
    elif sort_by == "Luas Tanah Terbesar":
        all_results.sort(key=lambda x: float(x["row"].get("luas_tanah_m2", 0)), reverse=True)
    else:  # Default: Kecocokan AI
        all_results.sort(key=lambda x: x["score"], reverse=True)

    return all_results[:top_k], len(valid_indices)

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

def label_badge(label, bg_color, text_color, border_color):
    return (f'<span style="background:{bg_color}; color:{text_color}; border:1px solid {border_color};'
            f'padding:4px 12px; border-radius:30px; font-size:12px; font-weight:600;'
            f'margin-right:6px; display:inline-block; box-shadow:0 2px 4px rgba(0,0,0,0.1);">{label}</span>')

# Modern modal dialog for property details
@st.dialog("✨ Eksplorasi Properti", width="large")
def show_property_modal(row):
    title = str(row.get("title", "Properti"))
    harga = format_harga(row.get("harga_rp", 0))
    lt = row.get("luas_tanah_m2", "-")
    lb = row.get("luas_bangunan_m2", "-")
    url = str(row.get("url", ""))

    # Modern Airbnb-like header box for modal
    st.markdown(f"""
    <div style='background:linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%); border-radius:20px; padding:28px; border:1px solid rgba(255, 90, 95, 0.2); margin-bottom:24px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <p style='color:#aaa; font-size:12px; margin:0; text-transform:uppercase; letter-spacing:1.5px;'>Harga Penawaran</p>
                <h2 style='color:#FF5A5F; font-size:36px; margin:4px 0 0 0; font-weight:800; text-shadow: 0 2px 10px rgba(255, 90, 95, 0.2);'>{harga}</h2>
            </div>
            <div style='display:flex; gap:24px;'>
                <div style='background:rgba(255,255,255,0.05); padding:12px 20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); text-align:center;'>
                    <span style='font-size:20px; font-weight:800; color:#fff; display:block;'>{lt} m²</span>
                    <span style='color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Luas Tanah</span>
                </div>
                <div style='background:rgba(255,255,255,0.05); padding:12px 20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); text-align:center;'>
                    <span style='font-size:20px; font-weight:800; color:#fff; display:block;'>{lb} m²</span>
                    <span style='color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Luas Bangunan</span>
                </div>
            </div>
        </div>
    </div>
    <h3 style='color:#fff; font-size:22px; font-weight:700; margin-bottom:16px;'>{title}</h3>
    """, unsafe_allow_html=True)
    
    # Priority detail from dataframe column
    full_desc = str(row.get("full_description", ""))
    if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
        full_desc = str(row.get("teks_gabungan", ""))
        if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
            full_desc = str(row.get("text_blob", ""))
            
    if full_desc:
        st.markdown(f"<div style='color:#ccc; font-size:15px; line-height:1.7; white-space:pre-wrap; background:#181818; padding:24px; border-radius:12px; border:1px solid #2a2a2a; margin-bottom:20px;'>{full_desc}</div>", unsafe_allow_html=True)
    else:
        st.info("Tidak ada deskripsi detail tambahan.")

    # Display Link if present
    if url and url.strip().lower() not in ["nan", "not_found", ""]:
        st.markdown(f'<a href="{url}" target="_blank" style="text-align:center; display:block; background:#FF5A5F; color:#fff; padding:12px 24px; border-radius:12px; font-weight:bold; text-decoration:none; box-shadow:0 4px 15px rgba(255,90,95,0.3); transition:all 0.3s;">🌐 Buka Halaman Sumber Properti</a>', unsafe_allow_html=True)

# Render property card
def render_card(item, rank):
    row   = item["row"]
    score = item["score"]

    title    = str(row.get("title", "Properti"))
    harga    = format_harga(row.get("harga_rp", 0))
    lt       = row.get("luas_tanah_m2", "-")
    lb       = row.get("luas_bangunan_m2", "-")

    full_desc = str(row.get("full_description", ""))
    if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
        full_desc = str(row.get("teks_gabungan", ""))
        if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
            full_desc = str(row.get("text_blob", ""))
            
    snippet  = full_desc[:200] + "..." if len(full_desc) > 200 else full_desc

    # Modern badging system
    badges = ""
    if row.get("Hybrid_Bebas_Banjir", 0) == 1:
        badges += label_badge("🛡️ Bebas Banjir", "rgba(43, 104, 88, 0.2)", "#6ee7b7", "rgba(110, 231, 183, 0.3)")
    if row.get("AI_Bisa_KPR", 0) == 1:
        badges += label_badge("💳 Bisa KPR", "rgba(26, 82, 118, 0.2)", "#93c5fd", "rgba(147, 197, 253, 0.3)")
    if row.get("AI_Legalitas_SHM", 0) == 1:
        badges += label_badge("📄 Legalitas SHM", "rgba(147, 81, 22, 0.2)", "#fde047", "rgba(253, 224, 71, 0.3)")

    score_pct = int(score * 100)

    # Injected CSS for smooth hover expansion over snippet
    st.markdown(f"""
<style>
.property-card-{rank} {{
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 0px;
    background: linear-gradient(135deg, rgba(28, 28, 28, 0.6) 0%, rgba(20, 20, 20, 0.6) 100%);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}}
.property-card-{rank}:hover {{
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(255, 90, 95, 0.12), 0 1px 1px rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 90, 95, 0.4);
    background: linear-gradient(135deg, rgba(38, 38, 38, 0.75) 0%, rgba(24, 24, 24, 0.75) 100%);
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
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
         <span style="font-size:11px;color:#fff;font-weight:800;background:#FF5A5F;padding:3px 10px;border-radius:12px;box-shadow: 0 2px 8px rgba(255, 90, 95, 0.4);">#{rank}</span>
         <div style="flex:1; max-width: 100px; height:6px; background:#333; border-radius:3px; overflow:hidden;">
            <div style="width:{score_pct}%; height:100%; background:linear-gradient(90deg, #FF5A5F, #ff8a8e); border-radius:3px;"></div>
         </div>
         <span style="font-size:11px; color:#FF5A5F; font-weight:700; letter-spacing:1px;">{score_pct}% MATCH</span>
      </div>
      <p style="font-size:22px;font-weight:700;margin:0 0 10px;color:#fff;line-height:1.3;">{title}</p>
      <div style="margin-bottom:14px; display:flex; flex-wrap:wrap; gap:4px;">{badges}</div>
    </div>
    <div style="text-align:right;min-width:140px;">
      <p style="font-size:26px;font-weight:800;color:#FF5A5F;margin:0;text-shadow:0 2px 8px rgba(255,90,95,0.2);">{harga}</p>
    </div>
  </div>
  <div style="display:flex;gap:16px;font-size:13px;color:#fff;margin-bottom:14px;opacity:0.9;">
    <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:6px 12px; font-weight:600;">📐 LT: {lt} m²</span>
    <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:6px 12px; font-weight:600;">🏠 LB: {lb} m²</span>
  </div>
  <p class="snippet-{rank}">{snippet}</p>
</div>
""", unsafe_allow_html=True)

    col_btn_detail, col_btn_compare = st.columns([3, 1])
    with col_btn_detail:
        if st.button("✦ Lihat Detail Lengkap", key=f"btn_detail_{item['idx']}", type="secondary", use_container_width=True):
            show_property_modal(row)
    with col_btn_compare:
        # Comparison logic
        if "compare_list" not in st.session_state:
            st.session_state.compare_list = []
        
        is_compared = item["idx"] in st.session_state.compare_list
        btn_label = "➖ Hapus Banding" if is_compared else "⚖️ Bandingkan"
        
        if st.button(btn_label, key=f"btn_comp_{item['idx']}", type="primary" if is_compared else "secondary", use_container_width=True):
            if is_compared:
                st.session_state.compare_list.remove(item["idx"])
            else:
                if len(st.session_state.compare_list) >= 4:
                    st.warning("Maksimal perbandingan adalah 4 properti!")
                else:
                    st.session_state.compare_list.append(item["idx"])
            st.rerun()
            
    # Spacer
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI LAYOUT - HERO
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg, rgba(255, 90, 95, 0.18) 0%, rgba(0,0,0,0) 100%); padding: 64px 24px; border-radius: 24px; text-align: center; margin-bottom: 32px; border: 1px solid rgba(255, 255, 255, 0.08); box-shadow:0 12px 32px rgba(0,0,0,0.2);'>
    <h1 style='font-size: 58px; font-weight: 800; color: #fff; margin-bottom: 16px; letter-spacing: -1.5px; line-height:1.2; text-shadow:0 2px 20px rgba(0,0,0,0.6);'>Find Your Dream Home<br><span style='color: #FF5A5F; background:linear-gradient(90deg, #FF5A5F, #ff8a8e); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>With AI Power.</span></h1>
    <p style='color: #aaa; font-size: 16px; max-width: 600px; margin: 0 auto 16px auto; line-height:1.6;'>Coba jelaskan kriteria rumah impian Anda secara natural. Model Hibrida Lexical-Semantic AI kami akan merekomendasikan pilihan properti terbaik untuk Anda.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
try:
    df, bm25_model, sbert_model, doc_tensor = load_resources()
    data_ok = True
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.info("Pastikan folder `data/` berisi: `properties_enriched.csv`, `bm25_index.pkl`, `sbert_embeddings.npy`")
    data_ok = False

# ── Sidebar tabbed navigation ────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding-bottom: 10px; margin-top: -15px;'>
        <h2 style='margin: 0; color: #FF5A5F; font-size: 28px; font-weight: 900;'>🏠 SmartSearch</h2>
        <p style='color: #888; font-size: 13px; margin: 4px 0 0 0;'>Engine Konfigurasi & Filter</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # Reset button
    if st.button("🔄 Reset Semua Filter", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "query_val": # Keep query search text if desired
                del st.session_state[key]
        st.rerun()

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    if data_ok:
        min_p = float(df["harga_rp"].min())
        max_p = float(df["harga_rp"].max())
        min_lt_val = float(df["luas_tanah_m2"].min())
        max_lt_val = float(df["luas_tanah_m2"].max())
        min_lb_val = float(df["luas_bangunan_m2"].min())
        max_lb_val = float(df["luas_bangunan_m2"].max())
        
        tab_fisik, tab_ai, tab_bobot = st.tabs(["🔍 Filter Fisik", "🤖 Filter AI", "⚙️ Bobot & Info"])
        
        with tab_fisik:
            st.markdown("<h4 style='margin-bottom:10px; color:#fff;'>Spesifikasi Fisik</h4>", unsafe_allow_html=True)
            
            # Harga slider
            st.write("Range Harga:")
            p_range = st.slider(
                "Pilih Range Harga",
                min_value=int(min_p),
                max_value=int(max_p),
                value=(int(min_p), int(max_p)),
                step=50_000_000,
                label_visibility="collapsed",
                key="slider_price"
            )
            st.caption(f"Terpilih: **{format_harga(p_range[0])}** - **{format_harga(p_range[1])}**")
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            # LT slider
            st.write("Luas Tanah (LT):")
            lt_range = st.slider(
                "Pilih Luas Tanah",
                min_value=int(min_lt_val),
                max_value=int(max_lt_val),
                value=(int(min_lt_val), int(max_lt_val)),
                step=10,
                label_visibility="collapsed",
                key="slider_lt"
            )
            st.caption(f"Terpilih: **{lt_range[0]} m²** - **{lt_range[1]} m²**")
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            # LB slider
            st.write("Luas Bangunan (LB):")
            lb_range = st.slider(
                "Pilih Luas Bangunan",
                min_value=int(min_lb_val),
                max_value=int(max_lb_val),
                value=(int(min_lb_val), int(max_lb_val)),
                step=10,
                label_visibility="collapsed",
                key="slider_lb"
            )
            st.caption(f"Terpilih: **{lb_range[0]} m²** - **{lb_range[1]} m²**")
            st.divider()
            
            # Sorting
            st.markdown("<h4 style='margin-bottom:10px; color:#fff;'>Urutan Hasil</h4>", unsafe_allow_html=True)
            sort_by = st.selectbox(
                "Metode Pengurutan",
                ["Kecocokan AI", "Harga Terendah", "Harga Tertinggi", "Luas Tanah Terbesar"],
                key="select_sort"
            )
            
            # Limit results
            top_k = st.slider("Jumlah Hasil", 3, 20, 10, key="slider_topk")

        with tab_ai:
            st.markdown("<h4 style='margin-bottom:10px; color:#fff;'>Syarat Mutlak (Hard Filter)</h4>", unsafe_allow_html=True)
            st.write("Saring properti berdasarkan klasifikasi biner otomatis model IndoBERT:")
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            cb_banjir = st.checkbox("✅ Wajib Bebas Banjir", key="cb_banjir")
            cb_kpr    = st.checkbox("✅ Wajib Bisa KPR", key="cb_kpr")
            cb_shm    = st.checkbox("✅ Wajib Sertifikat SHM", key="cb_shm")
            
        with tab_bobot:
            st.markdown("<h4 style='margin-bottom:10px; color:#fff;'>Advanced AI Config</h4>", unsafe_allow_html=True)
            bm25_weight = st.slider(
                "BM25 Weight (Lexical)", 0.0, 1.0, 0.7, 0.1,
                help="Mengatur sensitivitas terhadap kecocokan kata kunci eksak.",
                key="slider_bm25"
            )
            sbert_weight = round(1.0 - bm25_weight, 1)
            st.caption(f"SBERT Weight (Semantic) otomatis: **{sbert_weight}**")
            
            st.divider()
            
            st.markdown("<h4 style='margin-bottom:10px; color:#fff;'>Statistik Korpus</h4>", unsafe_allow_html=True)
            st.metric("Total Properti Terindeks", f"{len(df):,}")
            if "Hybrid_Bebas_Banjir" in df.columns:
                st.metric("Coverage Bebas Banjir", f"{df['Hybrid_Bebas_Banjir'].sum():,}")
            if "AI_Bisa_KPR" in df.columns:
                st.metric("Coverage Bisa KPR", f"{df['AI_Bisa_KPR'].sum():,}")
            if "AI_Legalitas_SHM" in df.columns:
                st.metric("Coverage Legalitas SHM", f"{df['AI_Legalitas_SHM'].sum():,}")
    else:
        st.error("Data tidak berhasil dimuat di sidebar.")

# ─────────────────────────────────────────────
# SEARCH BAR & QUERY CONTROLLER
# ─────────────────────────────────────────────
if "query_val" not in st.session_state:
    st.session_state.query_val = ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Ketik pencarian...",
        value=st.session_state.query_val,
        placeholder='cth: "Rumah mewah di Jakarta Selatan bebas banjir bisa KPR"',
        label_visibility="collapsed",
    )
    # Update session state query value when user writes
    st.session_state.query_val = query
with col_btn:
    cari = st.button("🔍 Cari Properti", use_container_width=True, type="primary")

# Search recommendations
st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
st.write("💡 Pencarian Populer:")
rec_cols = st.columns([1, 1, 1])
with rec_cols[0]:
    if st.button("✨ Mansion Mewah Jakarta", use_container_width=True, key="rec1"):
        st.session_state.query_val = "Mansion Mewah Jakarta"
        st.rerun()
with rec_cols[1]:
    if st.button("🚆 Townhouse Dekat MRT", use_container_width=True, key="rec2"):
        st.session_state.query_val = "Townhouse Dekat MRT"
        st.rerun()
with rec_cols[2]:
    if st.button("🏊 Villa Bali Kolam Renang", use_container_width=True, key="rec3"):
        st.session_state.query_val = "Villa Bali Kolam Renang"
        st.rerun()

st.divider()

# ─────────────────────────────────────────────
# SEARCH RESULTS RENDERING
# ─────────────────────────────────────────────
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
            price_range=p_range,
            lt_range=lt_range,
            lb_range=lb_range,
            sort_by=sort_by
        )

    if not results:
        st.warning("😕 Tidak ada properti yang memenuhi kriteria Anda. Coba kurangi filter atau perbaiki kueri.")
    else:
        st.markdown(
            f"<p style='color:#ccc; font-size:15px; margin-bottom: 20px;'>Ditemukan <b>{total_lolos}</b> properti "
            f"yang memenuhi filter — menampilkan <b>{len(results)}</b> terbaik berdasarkan <b>{sort_by}</b>.</p>",
            unsafe_allow_html=True
        )
        for rank, item in enumerate(results, 1):
            render_card(item, rank)

elif data_ok:
    st.markdown("""
<div style='text-align:center; padding: 40px 20px;'>
    <div style='font-size: 72px; margin-bottom: 24px;'>🏡</div>
    <h2 style='color:#fff; font-weight:800; font-size:28px;'>Siap Mengeksplorasi Properti Impian?</h2>
    <p style='color:#888; font-size:15.5px; max-width:460px; margin: 8px auto 32px auto; line-height:1.6;'>
        Gunakan kolom pencarian di atas dengan bahasa natural atau klik pencarian populer untuk melihat performa pencari AI.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FLOATING PROPERTY COMPARISON PANEL
# ─────────────────────────────────────────────
if "compare_list" in st.session_state and st.session_state.compare_list:
    st.divider()
    st.markdown("""
    <div style='background:rgba(255, 90, 95, 0.1); padding:12px 20px; border-radius:12px; border:1px solid rgba(255, 90, 95, 0.3); margin-bottom: 20px;'>
        <h3 style='margin:0; color:#fff; font-size:20px; font-weight:700; display:flex; align-items:center; gap:8px;'>
            ⚖️ Panel Perbandingan Properti <span style='font-size:12px; background:#FF5A5F; padding:2px 8px; border-radius:10px;'>{len(st.session_state.compare_list)} Terpilih</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)

    compare_df = df.loc[st.session_state.compare_list]
    
    # Render table
    cols_comp = st.columns(len(compare_df) + 1)
    
    with cols_comp[0]:
        st.markdown("<p style='font-weight:800; color:#FF5A5F; font-size:14px; margin-bottom:10px;'>Spesifikasi</p>", unsafe_allow_html=True)
        st.write("**Harga**")
        st.write("**Luas Tanah**")
        st.write("**Luas Bangunan**")
        st.write("**Bebas Banjir**")
        st.write("**Bisa KPR**")
        st.write("**Surat SHM**")
        
    for i, (orig_idx, row) in enumerate(compare_df.iterrows(), 1):
        with cols_comp[i]:
            title_short = str(row.get("title", "Properti"))
            if len(title_short) > 28:
                title_short = title_short[:25] + "..."
            st.markdown(f"<p style='font-weight:700; color:#fff; font-size:14px; margin-bottom:10px;'>{title_short}</p>", unsafe_allow_html=True)
            st.write(format_harga(row.get("harga_rp", 0)))
            st.write(f"{row.get('luas_tanah_m2', '-')} m²")
            st.write(f"{row.get('luas_bangunan_m2', '-')} m²")
            
            def yes_no_badge(val):
                return "✅ Ya" if val == 1 else "❌ Tidak"
                
            st.write(yes_no_badge(row.get("Hybrid_Bebas_Banjir", 0)))
            st.write(yes_no_badge(row.get("AI_Bisa_KPR", 0)))
            st.write(yes_no_badge(row.get("AI_Legalitas_SHM", 0)))
            
            if st.button("Hapus", key=f"btn_remove_comp_{orig_idx}", type="secondary", use_container_width=True):
                st.session_state.compare_list.remove(orig_idx)
                st.rerun()