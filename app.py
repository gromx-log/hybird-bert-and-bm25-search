import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Pencarian Properti Cerdas",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------
# CUSTOM CSS FOR GOOGLE ANTIGRAVITY LIGHT & RAINBOW THEME
# ---------------------------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #ffffff !important;
    color: #202124 !important;
}

[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid #dadce0 !important;
}

/* Apply theme font & color only to standard content elements in the sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] span:not([class*="stIcon"]):not([data-testid="stIconMaterial"]) {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #202124 !important;
}

[data-testid="stSidebar"] button:not([data-testid="stSidebarCollapseButton"] button) {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #202124 !important;
}

/* Restore native fonts for Streamlit collapse button and material symbols */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] *,
span[class*="stIcon"],
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", inherit !important;
}

/* Tab text styling */
button[data-baseweb="tab"] {
    color: #5f6368 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4285F4 !important;
    border-bottom: 2px solid #4285F4 !important;
}

/* Input boxes styling with focus ring overrides */
div[data-testid="stTextInput"] > div[data-baseweb="input"],
div[data-testid="stTextInput"] > div[data-baseweb="input"] > div {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stTextInput"] input {
    border-radius: 24px !important;
    border: 1px solid #dadce0 !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    color: #202124 !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 6px rgba(32,33,36,0.08) !important;
    transition: all 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #4285F4 !important;
    box-shadow: 0 1px 6px rgba(32,33,36,0.16), 0 0 0 1px #4285F4 !important;
    outline: none !important;
}

/* Primary buttons */
button[kind="primary"] {
    background-color: #4285F4 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 2px rgba(66,133,244,0.3) !important;
}
button[kind="primary"]:hover {
    background-color: #357ae8 !important;
    box-shadow: 0 4px 8px rgba(66,133,244,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Secondary/normal buttons */
div.stButton > button {
    border-radius: 18px !important;
    background-color: #f1f3f4 !important;
    color: #3c4043 !important;
    border: 1px solid transparent !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    background-color: #e8eaed !important;
    color: #202124 !important;
    border-color: #dadce0 !important;
}

/* Specific styling for sidebar components */
div[data-testid="stMarkdownContainer"] p {
    color: #202124 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #202124 !important;
    font-weight: 800 !important;
}

/* Dividers */
hr {
    border-color: #dadce0 !important;
}

/* Sidebar spaciousness improvement */
div[data-testid="stSlider"] {
    margin-bottom: 24px !important;
    padding: 0 8px !important;
}
[data-testid="stSidebar"] h4 {
    font-weight: 700 !important;
    margin-top: 16px !important;
    margin-bottom: 12px !important;
    font-size: 16px !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
    margin-bottom: 12px !important;
}

/* Clean styling via marker classes */
div:has(> div > [class^="marker-"]) {
    display: none !important;
}

/* Premium Search Input Styling with Rainbow Glow Underline */
div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"] {
    position: relative !important;
}

div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"]::after {
    content: '' !important;
    position: absolute !important;
    bottom: 0 !important;
    left: 20px !important;
    right: 20px !important;
    height: 3px !important;
    background: transparent !important;
    border-radius: 2px !important;
    transition: background 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    z-index: 10 !important;
}

div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"]:focus-within::after {
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853) !important;
}

div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"] input {
    border-radius: 28px !important;
    border: 1px solid #dadce0 !important;
    padding: 14px 24px !important;
    height: 52px !important;
    font-size: 16px !important;
    color: #202124 !important;
    background-color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"] input:hover {
    border-color: #c0c0c0 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

div:has(> div > .marker-search-input) + div div[data-testid="stTextInput"] input:focus {
    border-color: #4285F4 !important;
    box-shadow: 0 4px 16px rgba(66, 133, 244, 0.25) !important;
    outline: none !important;
}

/* Premium Search Button Styling */
div:has(> div > .marker-search-btn) + div button {
    border-radius: 28px !important;
    height: 52px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    background-color: #4285F4 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3) !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div:has(> div > .marker-search-btn) + div button:hover {
    background-color: #357ae8 !important;
    box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Suggestion Chips styling */
div:has(> div > [class^="marker-chip-"]) + div button {
    border-radius: 20px !important;
    background-color: #ffffff !important;
    color: #3c4043 !important;
    border: 1px solid #dadce0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

div:has(> div > .marker-chip-blue) + div button {
    border-left: 4px solid #4285F4 !important;
}
div:has(> div > .marker-chip-blue) + div button:hover {
    border-color: #4285F4 !important;
    background-color: rgba(66, 133, 244, 0.04) !important;
    color: #4285F4 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(66, 133, 244, 0.1) !important;
}

div:has(> div > .marker-chip-red) + div button {
    border-left: 4px solid #EA4335 !important;
}
div:has(> div > .marker-chip-red) + div button:hover {
    border-color: #EA4335 !important;
    background-color: rgba(234, 67, 53, 0.04) !important;
    color: #EA4335 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(234, 67, 53, 0.1) !important;
}

div:has(> div > .marker-chip-green) + div button {
    border-left: 4px solid #34A853 !important;
}
div:has(> div > .marker-chip-green) + div button:hover {
    border-color: #34A853 !important;
    background-color: rgba(52, 168, 83, 0.04) !important;
    color: #34A853 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(52, 168, 83, 0.1) !important;
}

/* Floating comparison button container */
div:has(> div > .marker-compare-float-btn) + div {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    z-index: 99999 !important;
    border-radius: 28px !important;
    box-shadow: 0 8px 24px rgba(66, 133, 244, 0.35) !important;
    background: transparent !important;
}

div:has(> div > .marker-compare-float-btn) + div button {
    border-radius: 28px !important;
    height: 56px !important;
    padding: 0 28px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    background: linear-gradient(90deg, #4285F4, #34A853) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div:has(> div > .marker-compare-float-btn) + div button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 12px 32px rgba(66, 133, 244, 0.45) !important;
}

/* Compared property card overrides */
.property-card-compared {
    background: rgba(66, 133, 244, 0.02) !important;
    border: 1px solid #4285F4 !important;
    box-shadow: 0 4px 16px rgba(66, 133, 244, 0.08) !important;
}
.property-card-compared::before {
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853) !important;
    height: 5px !important;
}

/* Comparison Table Modal Styling */
.comparison-table-wrapper {
    overflow-x: auto !important;
    margin-bottom: 20px !important;
}
.comparison-table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-family: 'Outfit', sans-serif !important;
    margin-bottom: 24px !important;
}
.comparison-table th, .comparison-table td {
    padding: 12px 16px !important;
    text-align: left !important;
    border-bottom: 1px solid #dadce0 !important;
    font-size: 14.5px !important;
    color: #202124 !important;
    vertical-align: middle !important;
}
.comparison-table th {
    font-weight: 700 !important;
    background-color: #f8f9fa !important;
    color: #202124 !important;
    border-top: 1px solid #dadce0 !important;
    font-size: 15px !important;
}
.comparison-table td:first-child {
    font-weight: 600 !important;
    color: #5f6368 !important;
    background-color: #f8f9fa !important;
    width: 160px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# PATHS
# ---------------------------------------------
PATH_DATA       = "data/properties_enriched.csv"
PATH_BM25       = "data/bm25_index.pkl"
PATH_EMBEDDINGS = "data/sbert_embeddings.npy"

# ---------------------------------------------
# REGEX POLA BEBAS BANJIR
# ---------------------------------------------
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

# ---------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------
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

# ---------------------------------------------
# SEARCH FUNCTION
# ---------------------------------------------
def hybrid_search(query, df, bm25, model_sbert, doc_tensor,
                  top_k=10, bm25_w=0.7, sbert_w=0.3,
                  filter_banjir=False, filter_kpr=False, filter_shm=False,
                  price_range=None, lt_range=None, lb_range=None, sort_by="Kecocokan AI"):
    """
    Two-Stage Hybrid Search with Global Score Normalization & Relevance Threshold Filtering.
    """
    query_lower = query.lower()

    # 1. Calculate raw scores first (required for global normalization and relevance filtering)
    tokenized_query = query_lower.split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    query_emb    = model_sbert.encode(query, convert_to_tensor=True)
    doc_t        = doc_tensor.to(query_emb.device)
    cosine_scores = util.cos_sim(query_emb, doc_t)[0].cpu().numpy()

    # 2. Global Score Normalization
    norm_bm25 = min_max_normalize(bm25_scores)
    norm_bert = min_max_normalize(cosine_scores)

    # Auto-detect from query text PLUS explicit sidebar checkboxes
    wajib_banjir = filter_banjir or "banjir" in query_lower
    wajib_kpr    = filter_kpr    or "kpr"    in query_lower
    wajib_shm    = filter_shm    or "shm"    in query_lower or "hak milik" in query_lower

    valid_indices = df.index.tolist()

    # 3. Hard Filter AI
    if wajib_banjir and "Hybrid_Bebas_Banjir" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "Hybrid_Bebas_Banjir"] == 1]
    if wajib_kpr and "AI_Bisa_KPR" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Bisa_KPR"] == 1]
    if wajib_shm and "AI_Legalitas_SHM" in df.columns:
        valid_indices = [i for i in valid_indices if df.loc[i, "AI_Legalitas_SHM"] == 1]

    # 4. Hard Filter Fisik (Harga, LT, LB)
    if price_range is not None:
        min_p, max_p = price_range
        valid_indices = [i for i in valid_indices if min_p <= df.loc[i, "harga_rp"] <= max_p]
    if lt_range is not None:
        min_lt, max_lt = lt_range
        valid_indices = [i for i in valid_indices if min_lt <= df.loc[i, "luas_tanah_m2"] <= max_lt]
    if lb_range is not None:
        min_lb, max_lb = lb_range
        valid_indices = [i for i in valid_indices if min_lb <= df.loc[i, "luas_bangunan_m2"] <= max_lb]

    # 5. Relevance Threshold Filter
    # Exclude properties where BM25 score is 0 and SBERT cosine similarity is < 0.22
    valid_indices = [i for i in valid_indices if not (bm25_scores[i] <= 0 and cosine_scores[i] < 0.22)]

    if not valid_indices:
        return [], 0

    # 6. Fusion using globally normalized scores
    hybrid_scores = [(bm25_w * norm_bm25[i]) + (sbert_w * norm_bert[i])
                     for i in valid_indices]

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

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
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
@st.dialog("Eksplorasi Properti", width="large")
def show_property_modal(row):
    title = str(row.get("title", "Properti"))
    harga = format_harga(row.get("harga_rp", 0))
    lt = row.get("luas_tanah_m2", "-")
    lb = row.get("luas_bangunan_m2", "-")
    url = str(row.get("url", ""))

    # Modern Google-styled header box for modal
    st.markdown(f"""
<div style='background:#f8f9fa; border-radius:16px; padding:24px; border:1px solid #dadce0; border-top: 4px solid #4285F4; margin-bottom:20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
<div style='display:flex; justify-content:space-between; align-items:center;'>
<div>
<p style='color:#5f6368; font-size:12px; margin:0; text-transform:uppercase; letter-spacing:1px; font-weight:600; font-family:"Outfit",sans-serif;'>Harga Penawaran</p>
<h2 style='color:#4285F4; font-size:32px; margin:4px 0 0 0; font-weight:800; font-family:"Outfit",sans-serif;'>{harga}</h2>
</div>
<div style='display:flex; gap:16px;'>
<div style='background:#ffffff; padding:10px 16px; border-radius:8px; border:1px solid #dadce0; text-align:center;'>
<span style='font-size:18px; font-weight:800; color:#202124; display:block; font-family:"Outfit",sans-serif;'>{lt} m&sup2;</span>
<span style='color:#5f6368; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; font-family:"Outfit",sans-serif;'>Luas Tanah</span>
</div>
<div style='background:#ffffff; padding:10px 16px; border-radius:8px; border:1px solid #dadce0; text-align:center;'>
<span style='font-size:18px; font-weight:800; color:#202124; display:block; font-family:"Outfit",sans-serif;'>{lb} m&sup2;</span>
<span style='color:#5f6368; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; font-family:"Outfit",sans-serif;'>Luas Bangunan</span>
</div>
</div>
</div>
</div>
<h3 style='color:#202124; font-family:"Outfit",sans-serif; font-size:22px; font-weight:700; margin-bottom:16px;'>{title}</h3>
""", unsafe_allow_html=True)
    
    # Priority detail from dataframe column
    full_desc = str(row.get("full_description", ""))
    if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
        full_desc = str(row.get("teks_gabungan", ""))
        if full_desc.strip().upper() in ["NOT_FOUND", "NAN", ""]:
            full_desc = str(row.get("text_blob", ""))
            
    if full_desc:
        st.markdown(f"<div style='color:#202124; font-family:\"Outfit\",sans-serif; font-size:15px; line-height:1.7; white-space:pre-wrap; background:#ffffff; padding:24px; border-radius:12px; border:1px solid #dadce0; margin-bottom:20px;'>{full_desc}</div>", unsafe_allow_html=True)
    else:
        st.info("Tidak ada deskripsi detail tambahan.")

    # Display Link if present
    if url and url.strip().lower() not in ["nan", "not_found", ""]:
        st.markdown(f'<a href="{url}" target="_blank" style="text-align:center; display:block; background:#4285F4; color:#ffffff; padding:12px 24px; border-radius:8px; font-weight:bold; text-decoration:none; box-shadow:0 2px 6px rgba(66,133,244,0.25); transition:all 0.3s; font-family:\"Outfit\",sans-serif;">Buka Halaman Sumber Properti</a>', unsafe_allow_html=True)

# Modern modal dialog for property comparison
@st.dialog("Perbandingan Properti", width="large")
def show_comparison_dialog(df):
    if "compare_list" not in st.session_state or not st.session_state.compare_list:
        st.write("Tidak ada properti yang dipilih untuk perbandingan.")
        return
        
    compare_df = df.loc[st.session_state.compare_list]
    
    # 1. Build the HTML Table dynamically
    html = '<div class="comparison-table-wrapper"><table class="comparison-table">'
    
    # Header row
    html += '<thead><tr><th>Spesifikasi</th>'
    for orig_idx, row in compare_df.iterrows():
        title_short = str(row.get("title", "Properti"))
        if len(title_short) > 28:
            title_short = title_short[:25] + "..."
        html += f'<th>{title_short}</th>'
    html += '</tr></thead><tbody>'
    
    # Row: Harga
    html += '<tr><td>Harga</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td><strong style="color: #4285F4; font-size: 15px;">{format_harga(row.get("harga_rp", 0))}</strong></td>'
    html += '</tr>'
    
    # Row: Luas Tanah
    html += '<tr><td>Luas Tanah</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td>{row.get("luas_tanah_m2", "-")} m&sup2;</td>'
    html += '</tr>'
    
    # Row: Luas Bangunan
    html += '<tr><td>Luas Bangunan</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td>{row.get("luas_bangunan_m2", "-")} m&sup2;</td>'
    html += '</tr>'
    
    def get_badge_html(val):
        if val == 1:
            return '<span style="color:#137333; font-weight:600; background:rgba(52,168,83,0.1); padding:4px 10px; border-radius:12px; font-size:12px;">Ya</span>'
        else:
            return '<span style="color:#c5221f; font-weight:600; background:rgba(234,67,53,0.1); padding:4px 10px; border-radius:12px; font-size:12px;">Tidak</span>'
            
    # Row: Bebas Banjir
    html += '<tr><td>Bebas Banjir</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td>{get_badge_html(row.get("Hybrid_Bebas_Banjir", 0))}</td>'
    html += '</tr>'
    
    # Row: Bisa KPR
    html += '<tr><td>Bisa KPR</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td>{get_badge_html(row.get("AI_Bisa_KPR", 0))}</td>'
    html += '</tr>'
    
    # Row: Surat SHM
    html += '<tr><td>Surat SHM</td>'
    for orig_idx, row in compare_df.iterrows():
        html += f'<td>{get_badge_html(row.get("AI_Legalitas_SHM", 0))}</td>'
    html += '</tr>'
    
    html += '</tbody></table></div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # 2. Render Aligned Control Buttons below the table
    cols_btn = st.columns([1.2] + [2.0] * len(compare_df))
    with cols_btn[0]:
        st.write("") # Spacer
    for i, (orig_idx, row) in enumerate(compare_df.iterrows(), 1):
        with cols_btn[i]:
            # Stacked action buttons
            if st.button("Lihat Detail", key=f"btn_det_comp_{orig_idx}", type="secondary", use_container_width=True):
                show_property_modal(row)
            st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
            if st.button("Hapus", key=f"btn_rem_comp_{orig_idx}", type="primary", use_container_width=True):
                st.session_state.compare_list.remove(orig_idx)
                st.rerun()

# Render property card
def render_card(item, rank):
    row   = item["row"]
    score = item["score"]

    # Check comparison status
    if "compare_list" not in st.session_state:
        st.session_state.compare_list = []
    is_compared = item["idx"] in st.session_state.compare_list
    compared_class = "property-card-compared" if is_compared else ""
    compare_badge = '<span style="font-size:11px;color:#fff;font-weight:800;background:#34A853;padding:3px 10px;border-radius:12px;box-shadow: 0 2px 6px rgba(52, 168, 83, 0.25); font-family:\'Outfit\',sans-serif;">TERPILIH UNTUK BANDING</span>' if is_compared else ""

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

    # Modern badging system (Emoji-free)
    badges = ""
    if row.get("Hybrid_Bebas_Banjir", 0) == 1:
        badges += label_badge("Bebas Banjir", "rgba(52, 168, 83, 0.08)", "#137333", "rgba(52, 168, 83, 0.25)")
    if row.get("AI_Bisa_KPR", 0) == 1:
        badges += label_badge("Bisa KPR", "rgba(66, 133, 244, 0.08)", "#1a73e8", "rgba(66, 133, 244, 0.25)")
    if row.get("AI_Legalitas_SHM", 0) == 1:
        badges += label_badge("Legalitas SHM", "rgba(251, 188, 5, 0.08)", "#b06000", "rgba(251, 188, 5, 0.25)")

    score_pct = int(score * 100)

    # Injected CSS for smooth hover expansion over snippet with google rainbow glow
    st.markdown(f"""
<style>
.property-card-{rank} {{
    border: 1px solid #dadce0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 0px;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}
.property-card-{rank}::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: transparent;
    transition: background 0.3s;
}}
.property-card-{rank}:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(66, 133, 244, 0.1);
    border-color: #4285F4;
}}
.property-card-{rank}:hover::before {{
    background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853);
}}
.snippet-{rank} {{
    font-size: 14.5px;
    color: #5f6368;
    line-height: 1.6;
    margin: 0;
    font-family: 'Outfit', sans-serif;
}}
</style>
<div class="property-card-{rank} {compared_class}">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div style="flex:1; padding-right:16px;">
<div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
<span style="font-size:11px;color:#fff;font-weight:800;background:#4285F4;padding:3px 10px;border-radius:12px;box-shadow: 0 2px 6px rgba(66, 133, 244, 0.25); font-family:'Outfit',sans-serif;">RANK {rank}</span>
{compare_badge}
<div style="flex:1; max-width: 100px; height:6px; background:#e8eaed; border-radius:3px; overflow:hidden;">
<div style="width:{score_pct}%; height:100%; background:linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853); border-radius:3px;"></div>
</div>
<span style="font-size:11px; color:#4285F4; font-weight:700; letter-spacing:1px; font-family:'Outfit',sans-serif;">{score_pct}% MATCH</span>
</div>
<p style="font-size:20px;font-weight:700;margin:0 0 10px;color:#202124;line-height:1.4;font-family:'Outfit',sans-serif;">{title}</p>
<div style="margin-bottom:14px; display:flex; flex-wrap:wrap; gap:4px;">{badges}</div>
</div>
<div style="text-align:right;min-width:140px;">
<p style="font-size:24px;font-weight:800;color:#4285F4;margin:0;font-family:'Outfit',sans-serif;">{harga}</p>
</div>
</div>
<div style="display:flex;gap:12px;font-size:13px;color:#5f6368;margin-bottom:14px;">
<span style="background:#f8f9fa; border:1px solid #dadce0; border-radius:8px; padding:6px 12px; font-weight:600; font-family:'Outfit',sans-serif;">LT: {lt} m&sup2;</span>
<span style="background:#f8f9fa; border:1px solid #dadce0; border-radius:8px; padding:6px 12px; font-weight:600; font-family:'Outfit',sans-serif;">LB: {lb} m&sup2;</span>
</div>
<p class="snippet-{rank}">{snippet}</p>
</div>
""", unsafe_allow_html=True)

    col_btn_detail, col_btn_compare = st.columns([3, 1])
    with col_btn_detail:
        if st.button("Lihat Detail Lengkap", key=f"btn_detail_{item['idx']}", type="secondary", use_container_width=True):
            show_property_modal(row)
    with col_btn_compare:
        # Comparison logic
        if "compare_list" not in st.session_state:
            st.session_state.compare_list = []
        
        is_compared = item["idx"] in st.session_state.compare_list
        btn_label = "Hapus Banding" if is_compared else "Bandingkan"
        
        if st.button(btn_label, key=f"btn_comp_{item['idx']}", type="primary" if is_compared else "secondary", use_container_width=True):
            if is_compared:
                st.session_state.compare_list.remove(item["idx"])
                st.rerun()
            else:
                if len(st.session_state.compare_list) >= 4:
                    st.toast("Maksimal perbandingan adalah 4 properti!")
                else:
                    st.session_state.compare_list.append(item["idx"])
                    st.rerun()
            
    # Spacer
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------
try:
    df, bm25_model, sbert_model, doc_tensor = load_resources()
    data_ok = True
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.info("Pastikan folder `data/` berisi: `properties_enriched.csv`, `bm25_index.pkl`, `sbert_embeddings.npy`")
    data_ok = False

# -- Sidebar tabbed navigation ----------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding-bottom: 10px; margin-top: -15px;'>
        <h2 style='margin: 0; color: #4285F4; font-family: "Outfit", sans-serif; font-size: 28px; font-weight: 900;'>SmartSearch</h2>
        <p style='color: #5f6368; font-size: 13px; margin: 4px 0 0 0;'>Engine Konfigurasi & Filter</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # Reset button
    if st.button("Reset Semua Filter", use_container_width=True):
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
        
        tab_fisik, tab_ai, tab_bobot = st.tabs(["Filter Fisik", "Filter AI", "Bobot & Info"])
        
        with tab_fisik:
            st.markdown("<h4 style='margin-bottom:10px; color:#202124; font-family:\"Outfit\",sans-serif;'>Spesifikasi Fisik</h4>", unsafe_allow_html=True)
            
            # Harga slider (Scaled to Billions for clean display)
            st.markdown("<p style='font-weight:600; margin-bottom: 2px; color:#202124;'>Range Harga (Miliar Rp)</p>", unsafe_allow_html=True)
            min_p_b = float(min_p) / 1_000_000_000
            max_p_b = float(max_p) / 1_000_000_000
            p_range_b = st.slider(
                "Pilih Range Harga",
                min_value=min_p_b,
                max_value=max_p_b,
                value=(min_p_b, max_p_b),
                step=0.05,
                label_visibility="collapsed",
                key="slider_price"
            )
            p_range = (p_range_b[0] * 1_000_000_000, p_range_b[1] * 1_000_000_000)
            st.caption(f"Terpilih: **{format_harga(p_range[0])}** - **{format_harga(p_range[1])}**")
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            # LT slider
            st.markdown("<p style='font-weight:600; margin-bottom: 2px; color:#202124;'>Luas Tanah (LT)</p>", unsafe_allow_html=True)
            lt_range = st.slider(
                "Pilih Luas Tanah",
                min_value=int(min_lt_val),
                max_value=int(max_lt_val),
                value=(int(min_lt_val), int(max_lt_val)),
                step=10,
                label_visibility="collapsed",
                key="slider_lt"
            )
            st.caption(f"Terpilih: **{lt_range[0]} m2** - **{lt_range[1]} m2**")
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

            # LB slider
            st.markdown("<p style='font-weight:600; margin-bottom: 2px; color:#202124;'>Luas Bangunan (LB)</p>", unsafe_allow_html=True)
            lb_range = st.slider(
                "Pilih Luas Bangunan",
                min_value=int(min_lb_val),
                max_value=int(max_lb_val),
                value=(int(min_lb_val), int(max_lb_val)),
                step=10,
                label_visibility="collapsed",
                key="slider_lb"
            )
            st.caption(f"Terpilih: **{lb_range[0]} m2** - **{lb_range[1]} m2**")
            st.divider()
            
            # Sorting
            st.markdown("<h4 style='margin-bottom:10px; color:#202124; font-family:\"Outfit\",sans-serif;'>Urutan Hasil</h4>", unsafe_allow_html=True)
            sort_by = st.selectbox(
                "Metode Pengurutan",
                ["Kecocokan AI", "Harga Terendah", "Harga Tertinggi", "Luas Tanah Terbesar"],
                key="select_sort"
            )
            
            # Limit results
            top_k = st.slider("Jumlah Hasil", 3, 20, 10, key="slider_topk")

        with tab_ai:
            st.markdown("<h4 style='margin-bottom:10px; color:#202124; font-family:\"Outfit\",sans-serif;'>Syarat Mutlak (Hard Filter)</h4>", unsafe_allow_html=True)
            st.write("Saring properti berdasarkan klasifikasi biner otomatis model IndoBERT:")
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            cb_banjir = st.checkbox("Wajib Bebas Banjir", key="cb_banjir")
            cb_kpr    = st.checkbox("Wajib Bisa KPR", key="cb_kpr")
            cb_shm    = st.checkbox("Wajib Sertifikat SHM", key="cb_shm")
            
        with tab_bobot:
            st.markdown("<h4 style='margin-bottom:10px; color:#202124; font-family:\"Outfit\",sans-serif;'>Advanced AI Config</h4>", unsafe_allow_html=True)
            bm25_weight = st.slider(
                "BM25 Weight (Lexical)", 0.0, 1.0, 0.7, 0.1,
                help="Mengatur sensitivitas terhadap kecocokan kata kunci eksak.",
                key="slider_bm25"
            )
            sbert_weight = round(1.0 - bm25_weight, 1)
            st.caption(f"SBERT Weight (Semantic) otomatis: **{sbert_weight}**")
            
            st.divider()
            
            st.markdown("<h4 style='margin-bottom:10px; color:#202124; font-family:\"Outfit\",sans-serif;'>Statistik Korpus</h4>", unsafe_allow_html=True)
            st.metric("Total Properti Terindeks", f"{len(df):,}")
            if "Hybrid_Bebas_Banjir" in df.columns:
                st.metric("Coverage Bebas Banjir", f"{df['Hybrid_Bebas_Banjir'].sum():,}")
            if "AI_Bisa_KPR" in df.columns:
                st.metric("Coverage Bisa KPR", f"{df['AI_Bisa_KPR'].sum():,}")
            if "AI_Legalitas_SHM" in df.columns:
                st.metric("Coverage Legalitas SHM", f"{df['AI_Legalitas_SHM'].sum():,}")
    else:
        st.error("Data tidak berhasil dimuat di sidebar.")

# ---------------------------------------------
# SEARCH BAR & QUERY CONTROLLER
# ---------------------------------------------
if "query_val" not in st.session_state:
    st.session_state.query_val = ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    st.markdown('<div class="marker-search-input"></div>', unsafe_allow_html=True)
    query = st.text_input(
        "Ketik pencarian...",
        value=st.session_state.query_val,
        placeholder='cth: "Rumah mewah di Jakarta Selatan bebas banjir bisa KPR"',
        label_visibility="collapsed",
    )
    # Update session state query value when user writes
    st.session_state.query_val = query
with col_btn:
    st.markdown('<div class="marker-search-btn"></div>', unsafe_allow_html=True)
    cari = st.button("Cari Properti", use_container_width=True, type="primary")

# Search recommendations
st.markdown("<div style='margin-top: 12px; margin-bottom: 8px; font-weight:600; color:#5f6368;'>Pencarian Populer:</div>", unsafe_allow_html=True)
rec_cols = st.columns([1, 1, 1])
with rec_cols[0]:
    st.markdown('<div class="marker-chip-blue"></div>', unsafe_allow_html=True)
    if st.button("Mansion Mewah Jakarta", use_container_width=True, key="rec1"):
        st.session_state.query_val = "Mansion Mewah Jakarta"
        st.rerun()
with rec_cols[1]:
    st.markdown('<div class="marker-chip-red"></div>', unsafe_allow_html=True)
    if st.button("Townhouse Dekat MRT", use_container_width=True, key="rec2"):
        st.session_state.query_val = "Townhouse Dekat MRT"
        st.rerun()
with rec_cols[2]:
    st.markdown('<div class="marker-chip-green"></div>', unsafe_allow_html=True)
    if st.button("Rumah Murah Bekasi KPR", use_container_width=True, key="rec3"):
        st.session_state.query_val = "Rumah Murah Bekasi KPR"
        st.rerun()

# Floating Comparison Trigger Button
if "compare_list" in st.session_state and st.session_state.compare_list:
    count = len(st.session_state.compare_list)
    st.markdown('<div class="marker-compare-float-btn"></div>', unsafe_allow_html=True)
    if st.button(f"Buka Perbandingan ({count}/4)", key="btn_float_compare", type="primary"):
        show_comparison_dialog(df)

st.divider()

# ---------------------------------------------
# SEARCH RESULTS RENDERING
# ---------------------------------------------
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
        st.warning("Tidak ada properti yang memenuhi kriteria Anda. Coba kurangi filter atau perbaiki kueri.")
    else:
        st.markdown(
            f"<p style='color:#5f6368; font-size:15.5px; margin-bottom: 24px; font-family:\"Outfit\",sans-serif;'>Ditemukan <b>{total_lolos}</b> properti "
            f"yang memenuhi filter - menampilkan <b>{len(results)}</b> terbaik berdasarkan <b>{sort_by}</b>.</p>",
            unsafe_allow_html=True
        )
        for rank, item in enumerate(results, 1):
            render_card(item, rank)

elif data_ok:
    st.markdown("""
<div style='text-align:center; padding: 60px 20px; background:#f8f9fa; border-radius:16px; border: 1px solid #dadce0;'>
    <div style='margin-bottom: 24px; display:flex; justify-content:center;'>
        <svg viewBox="0 0 24 24" width="72" height="72" stroke="#4285F4" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
    </div>
    <h2 style='color:#202124; font-family:"Outfit", sans-serif; font-weight:800; font-size:28px;'>Siap Mengeksplorasi Properti Impian?</h2>
    <p style='color:#5f6368; font-size:15.5px; max-width:480px; margin: 12px auto 0 auto; line-height:1.6;'>
        Gunakan kolom pencarian di atas dengan bahasa natural atau klik salah satu pencarian populer untuk melihat performa pencarian hibrida cerdas.
    </p>
</div>
""", unsafe_allow_html=True)

# (Comparison features are now handled via trigger banner and modal dialog)