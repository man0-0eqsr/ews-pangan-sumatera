import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import json
import os

st.set_page_config(
    page_title="EWS Volatilitas Harga Pangan – Pulau Sumatera",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #F8F7F4; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }

.ews-header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; color: white; }
.ews-header h1 { font-size: 1.4rem; font-weight: 600; margin: 0 0 4px 0; }
.ews-header p { font-size: 0.8rem; opacity: 0.65; margin: 0; }
.badge-ews { background: #e74c3c; color: white; font-size: 10px; font-weight: 600;
  padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em; margin-right: 8px; }
.badge-bi { background: rgba(255,255,255,0.15); color: white; font-size: 10px;
  padding: 3px 10px; border-radius: 20px; }

.metric-card { background: white; border-radius: 10px; padding: 1rem 1.2rem;
  border-left: 4px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.metric-card.danger { border-left-color: #e74c3c; }
.metric-card.warning { border-left-color: #f39c12; }
.metric-card.success { border-left-color: #27ae60; }
.metric-card.info { border-left-color: #2980b9; }
.metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 2rem; font-weight: 600; line-height: 1.1; margin: 4px 0 2px; }
.metric-sub { font-size: 11px; color: #aaa; }
.metric-card.danger .metric-value { color: #e74c3c; }
.metric-card.warning .metric-value { color: #f39c12; }
.metric-card.success .metric-value { color: #27ae60; }
.metric-card.info .metric-value { color: #2980b9; }

.risk-high { background: #fdecea; color: #c0392b; font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 12px; }
.risk-med  { background: #fef9e7; color: #d68910; font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 12px; }
.risk-low  { background: #eafaf1; color: #1e8449; font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 12px; }

.alert-box { border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-left: 3px solid; }
.alert-high { background: #fdecea; border-color: #e74c3c; }
.alert-med  { background: #fef9e7; border-color: #f39c12; }
.alert-low  { background: #eafaf1; border-color: #27ae60; }
.alert-title { font-size: 12px; font-weight: 600; }
.alert-body  { font-size: 11px; color: #555; margin-top: 2px; }

.section-title { font-size: 12px; font-weight: 600; color: #555;
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem; }
.interp-box { background: #f0f4ff; border-radius: 8px; padding: 0.75rem 1rem;
  font-size: 12px; color: #2c3e50; line-height: 1.6; border-left: 3px solid #2980b9; }

div[data-testid="stSidebar"] { background: #1a1a2e; }
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label { color: rgba(255,255,255,0.7) !important; font-size: 12px; }
div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
PROVS = ['Aceh','Sumut','Sumbar','Riau','Kepri','Jambi','Bengkulu','Sumsel','Babel','Lampung']
PROV_FULLNAME = {
    'Aceh':'Aceh', 'Sumut':'Sumatera Utara', 'Sumbar':'Sumatera Barat',
    'Riau':'Riau', 'Kepri':'Kepulauan Riau', 'Jambi':'Jambi',
    'Bengkulu':'Bengkulu', 'Sumsel':'Sumatera Selatan',
    'Babel':'Bangka-Belitung', 'Lampung':'Lampung',
}
KOMODS = ['Cabai Rawit','Cabai Merah','Bawang Merah','Bawang Putih',
          'Daging Sapi','Daging Ayam','Minyak Goreng','Beras']

LEVERAGE = np.array([
    [0.2157, 0.2024, 0.0178, 0.1131, 0.4716, 0.0611, 0.0115, 0.1053],
    [0.1946, 0.1932, 0.0377, 0.0835,-0.0437,-0.0326, 0.0235,-0.0809],
    [0.2006, 0.1626, 0.0680, 0.1458, 0.0813, 0.0145, 0.0797, 0.0414],
    [0.1993, 0.2267,-0.0546,-0.7611,-0.3268, 0.0331, 0.0469, 0.4306],
    [0.2817, 0.0205, 0.0999, 0.0280, 0.1272, 0.1206, 0.0311, 0.0796],
    [0.1688, 0.1692, 0.0680, 0.3161, 0.4656, 0.0438, 0.3641,-0.1756],
    [0.0657, 0.1296, 0.0240, 0.2952, 0.0680, 0.0416,-0.0789,-0.0352],
    [0.1417, 0.1517, 0.1250,-0.0299,-0.3505, 0.0556,-0.3815,-0.1166],
    [0.2072, 0.1395, 0.0193,-0.0789, 0.0677, 0.3001, 0.0190, 0.0356],
    [0.1960, 0.1410, 0.0668, 0.1517, 0.3347, 0.0803, 0.1704,-0.0191],
])

ABS_LEV = np.where(LEVERAGE == None, np.nan, np.abs(LEVERAGE.astype(float)))

RANK = np.array([
    [2,3,5,1,7,8,10,4], [7,3,7,7,10,9,8,5], [4,5,4,5,7,10,4,7],
    [5,1,6,1,5,8,6,1],  [1,10,2,10,6,2,7,6],[8,4,3,2,2,6,2,2],
    [10,9,8,3,8,7,5,9], [6,6,1,9,3,5,1,3],  [3,8,9,8,9,1,9,8],
    [6,7,5,4,4,3,3,10],
])

ALERTS = [
    dict(prov='Sumsel', komod='Telur Ayam', lv='high', time='Jul 2023',
         msg='Leverage 0.382 — efek asimetris tertinggi Sumsel. Harga sangat responsif terhadap shock negatif.'),
    dict(prov='Riau', komod='Bawang Putih', lv='high', time='Jun 2023',
         msg='Leverage −0.761 — shock penurunan harga memicu volatilitas 3× lebih besar dari shock kenaikan.'),
    dict(prov='Jambi', komod='Daging Sapi', lv='high', time='Jun 2023',
         msg='Leverage 0.466 — risiko transmisi inflasi pangan ke inflasi inti meningkat.'),
    dict(prov='Lampung', komod='Daging Sapi', lv='high', time='Mei 2023',
         msg='Leverage 0.335 — perlu pemantauan rantai pasok daging sapi koridor Lampung–Jabodetabek.'),
    dict(prov='Sumsel', komod='Minyak Goreng', lv='high', time='Apr 2023',
         msg='Leverage −0.382 — volatile pasca normalisasi harga HET, sinyal masih aktif.'),
    dict(prov='Kepri', komod='Cabai Rawit', lv='med', time='Mar 2023',
         msg='Leverage 0.282 — harga rentan terhadap gangguan pasokan dari sentra produksi.'),
    dict(prov='Babel', komod='Daging Ayam', lv='high', time='Feb 2023',
         msg='Leverage 0.300 — pola musiman dikonfirmasi, waspadai saat hari raya.'),
    dict(prov='Aceh', komod='Daging Sapi', lv='high', time='Jan 2023',
         msg='Leverage 0.472 — tertinggi se-Sumatera, terkait isolasi pasokan di musim hujan.'),
    dict(prov='Sumbar', komod='Daging Ayam', lv='med', time='Des 2022',
         msg='Estimasi EGARCH anomali (non-konvergensi). Perlu estimasi ulang dengan data yang lebih bersih.'),
    dict(prov='Sumsel', komod='Bawang Merah', lv='high', time='Nov 2022',
         msg='Leverage 0.125 — tertinggi se-Sumatera untuk bawang merah, waspadai gangguan distribusi.'),
]

def risk_level(val):
    if val is None or np.isnan(val): return 'anomaly'
    if val > 0.25: return 'high'
    if val > 0.12: return 'med'
    return 'low'

def risk_color(lv):
    return {'high':'#e74c3c','med':'#f39c12','low':'#27ae60','anomaly':'#95a5a6'}[lv]

def risk_label(lv):
    return {'high':'🔴 Tinggi','med':'🟡 Sedang','low':'🟢 Rendah','anomaly':'⚪ Anomali'}[lv]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Panel Kontrol")
    st.markdown("---")
    sel_prov = st.selectbox("Provinsi Fokus", ["Semua Provinsi"] + PROVS, index=8)
    sel_komod = st.selectbox("Komoditas", ["Semua Komoditas"] + KOMODS, index=0)
    st.markdown("---")
    st.markdown("**Threshold Risiko**")
    thresh_high = st.slider("Batas Risiko Tinggi (|γ|)", 0.10, 0.50, 0.25, 0.01)
    thresh_med  = st.slider("Batas Risiko Sedang (|γ|)", 0.05, 0.30, 0.12, 0.01)
    st.markdown("---")
    st.markdown("**Tampilkan**")
    show_labels = st.checkbox("Label provinsi di peta", True)
    show_anomali = st.checkbox("Tampilkan anomali", False)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px; opacity:0.5; line-height:1.6;'>
    Sumber data: PIHPS Nasional<br>
    Metode: EGARCH(1,1)<br>
    Periode: Juli 2017 – Juli 2023<br>
    Model estimasi: Python arch library<br><br>
    Bank Indonesia Sumatera Selatan<br>
    Call for Paper 2024
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ews-header">
  <div style="margin-bottom:8px;">
    <span class="badge-ews">EWS PANGAN</span>
    <span class="badge-bi">Bank Indonesia Sumatera Selatan</span>
  </div>
  <h1>Sistem Peringatan Dini Volatilitas Harga Pangan</h1>
  <p>Berbasis Model EGARCH · Analisis Volatilitas Asimetris · Pulau Sumatera · 2017–2023</p>
</div>
""", unsafe_allow_html=True)

# ── Tentukan indeks aktif ─────────────────────────────────────────────────────
pi = PROVS.index(sel_prov) if sel_prov != "Semua Provinsi" else 7  # default Sumsel
ki = KOMODS.index(sel_komod) if sel_komod != "Semua Komoditas" else None

if ki is None:
    vals = np.nanmean(ABS_LEV, axis=1)
    prov_vals = dict(zip(PROVS, vals))
    row_vals = ABS_LEV[pi]
else:
    vals = ABS_LEV[:, ki]
    prov_vals = dict(zip(PROVS, vals))
    row_vals = ABS_LEV[pi]

avg_prov = float(np.nanmean(row_vals))
high_count = int(np.sum([risk_level(v) == 'high' for v in row_vals]))
min_idx = int(np.nanargmin(row_vals))
max_idx = int(np.nanargmax(row_vals))

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card danger">
    <div class="metric-label">⚠ Komoditas Risiko Tinggi</div>
    <div class="metric-value">{high_count}</div>
    <div class="metric-sub">dari {len(KOMODS)} komoditas · {sel_prov}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card warning">
    <div class="metric-label">📈 Rata-rata |Leverage|</div>
    <div class="metric-value">{avg_prov:.3f}</div>
    <div class="metric-sub">nilai absolut γ EGARCH · {sel_prov}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card success">
    <div class="metric-label">✅ Paling Stabil</div>
    <div class="metric-value" style="font-size:1.1rem;padding-top:6px;">{KOMODS[min_idx]}</div>
    <div class="metric-sub">|γ| = {row_vals[min_idx]:.3f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card danger">
    <div class="metric-label">🔥 Paling Volatile</div>
    <div class="metric-value" style="font-size:1.1rem;padding-top:6px;">{KOMODS[max_idx]}</div>
    <div class="metric-sub">|γ| = {row_vals[max_idx]:.3f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Peta + Ranking ────────────────────────────────────────────────────────────
col_map, col_rank = st.columns([3, 2])

with col_map:
    st.markdown('<div class="section-title">🗺 Peta Choropleth Leverage EGARCH – Pulau Sumatera</div>', unsafe_allow_html=True)

    GEOJSON_PATH = os.path.join(os.path.dirname(__file__), 'sumatera.geojson')
    gdf = gpd.read_file(GEOJSON_PATH)

    # Buat DataFrame leverage untuk merge
    df_map = pd.DataFrame({
        'name': [PROV_FULLNAME[p] for p in PROVS],
        'short': PROVS,
        'lev': [prov_vals.get(p, np.nan) for p in PROVS],
    })
    df_map['risk'] = df_map['lev'].apply(risk_level)

    # Sembunyikan anomali jika tidak ditampilkan
    if not show_anomali:
        df_map.loc[(df_map['short']=='Sumbar') & (ki==5 if ki is not None else False), 'lev'] = np.nan

    gdf_plot = gdf.merge(df_map, on='name', how='left')
    gdf_plot['lev_display'] = gdf_plot['lev'].fillna(-1)
    gdf_plot['tooltip'] = gdf_plot.apply(
        lambda r: f"<b>{r['short']}</b><br>|γ| = {r['lev']:.4f}<br>Risiko: {risk_label(r['risk'])}"
        if not np.isnan(r['lev']) else f"<b>{r['short']}</b><br>Anomali estimasi", axis=1
    )

    fig_map = go.Figure()

    # Choropleth layer
    fig_map.add_trace(go.Choropleth(
        geojson=json.loads(gdf_plot.to_json()),
        locations=gdf_plot.index,
        z=gdf_plot['lev_display'],
        colorscale=[
            [0.0, '#95a5a6'],
            [0.001, '#27ae60'],
            [0.4,  '#f39c12'],
            [1.0,  '#c0392b'],
        ],
        zmin=0, zmax=float(np.nanmax(list(prov_vals.values()))),
        marker_line_color='white',
        marker_line_width=1.2,
        colorbar=dict(
            title=dict(text='|Leverage γ|', font=dict(size=11)),
            thickness=12, len=0.7,
            tickfont=dict(size=10),
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=gdf_plot['tooltip'],
    ))

    # Highlight Sumsel dengan border tebal
    sumsel_gdf = gdf_plot[gdf_plot['short'] == 'Sumsel']
    if not sumsel_gdf.empty and sel_prov != "Semua Provinsi":
        fig_map.add_trace(go.Choropleth(
            geojson=json.loads(sumsel_gdf.to_json()),
            locations=sumsel_gdf.index,
            z=[0],
            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
            marker_line_color='#1a1a2e',
            marker_line_width=3,
            showscale=False,
            hoverinfo='skip',
        ))

    # Label provinsi
    if show_labels:
        centroids = gdf_plot.copy()
        centroids['cx'] = centroids.geometry.centroid.x
        centroids['cy'] = centroids.geometry.centroid.y

        # Kepri perlu offset (kepulauan)
        kepri_mask = centroids['short'] == 'Kepri'
        centroids.loc[kepri_mask, 'cx'] = 104.5
        centroids.loc[kepri_mask, 'cy'] = 1.0

        fig_map.add_trace(go.Scattergeo(
            lon=centroids['cx'],
            lat=centroids['cy'],
            mode='text',
            text=centroids['short'],
            textfont=dict(size=10, color='#1a1a2e', family='IBM Plex Sans'),
            hoverinfo='skip',
            showlegend=False,
        ))

    fig_map.update_geos(
        fitbounds='locations',
        visible=False,
        bgcolor='#f8f7f4',
    )
    fig_map.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#f8f7f4',
        geo=dict(bgcolor='#f8f7f4'),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    # Legenda
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.markdown('<span style="font-size:11px;">🟢 Stabil (|γ| < 0.12)</span>', unsafe_allow_html=True)
    col_l2.markdown('<span style="font-size:11px;">🟡 Sedang (0.12–0.25)</span>', unsafe_allow_html=True)
    col_l3.markdown('<span style="font-size:11px;">🔴 Volatile (> 0.25)</span>', unsafe_allow_html=True)
    col_l4.markdown('<span style="font-size:11px;">⚪ Anomali estimasi</span>', unsafe_allow_html=True)

with col_rank:
    st.markdown('<div class="section-title">📊 Ranking Volatilitas Komoditas</div>', unsafe_allow_html=True)

    rank_data = pd.DataFrame({
        'Komoditas': KOMODS,
        'abs_lev': ABS_LEV[pi],
        'raw_lev': LEVERAGE[pi].astype(float),
    }).dropna().sort_values('abs_lev', ascending=False).reset_index(drop=True)

    rank_data['risk'] = rank_data['abs_lev'].apply(risk_level)
    rank_data['color'] = rank_data['risk'].apply(risk_color)
    rank_data['rank'] = range(1, len(rank_data)+1)

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        y=rank_data['Komoditas'][::-1],
        x=rank_data['abs_lev'][::-1],
        orientation='h',
        marker_color=rank_data['color'][::-1],
        text=[f"{v:.3f}" for v in rank_data['abs_lev'][::-1]],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>|γ| = %{x:.4f}<extra></extra>',
    ))
    fig_rank.add_vline(x=thresh_high, line_dash='dash', line_color='#e74c3c', line_width=1,
                       annotation_text='Batas Tinggi', annotation_font_size=9)
    fig_rank.add_vline(x=thresh_med,  line_dash='dash', line_color='#f39c12', line_width=1,
                       annotation_text='Batas Sedang', annotation_font_size=9)
    fig_rank.update_layout(
        height=360,
        margin=dict(l=0, r=50, t=10, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(title='|Leverage γ|', gridcolor='#f0f0f0', tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(fig_rank, use_container_width=True, config={'displayModeBar': False})

    # Interpretasi cepat
    top_komod = rank_data.iloc[0]
    raw = float(top_komod['raw_lev'])
    arah = "berita buruk (harga naik)" if raw > 0 else "berita baik (harga turun)"
    st.markdown(f"""<div class="interp-box">
    <b>Interpretasi:</b> Di <b>{sel_prov}</b>, <b>{top_komod['Komoditas']}</b> memiliki efek leverage
    tertinggi (|γ| = {top_komod['abs_lev']:.3f}). Artinya, shock dari <em>{arah}</em>
    meningkatkan volatilitas secara signifikan — indikasi pasar yang asimetris dan rentan terhadap guncangan harga.
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Heatmap + Line Chart ───────────────────────────────────────────────────────
col_heat, col_line = st.columns(2)

with col_heat:
    st.markdown('<div class="section-title">🔲 Heatmap Leverage: Semua Provinsi × Komoditas</div>', unsafe_allow_html=True)

    hm_data = pd.DataFrame(ABS_LEV, index=PROVS, columns=KOMODS)
    
    # Warna custom: hijau - kuning - merah
    custom_scale = [
        [0.0,  '#27ae60'], [0.3, '#2ecc71'],
        [0.5,  '#f39c12'], [0.7, '#e67e22'],
        [1.0,  '#c0392b'],
    ]

    fig_hm = go.Figure(data=go.Heatmap(
        z=hm_data.values,
        x=KOMODS,
        y=PROVS,
        colorscale=custom_scale,
        text=np.round(hm_data.values, 3).astype(str),
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorbar=dict(title='|γ|', thickness=10, tickfont=dict(size=9)),
        hovertemplate='<b>%{y} – %{x}</b><br>|γ| = %{z:.4f}<extra></extra>',
    ))
    fig_hm.update_layout(
        height=320,
        margin=dict(l=60, r=20, t=10, b=80),
        paper_bgcolor='white',
        xaxis=dict(tickangle=30, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    # Highlight Sumsel
    fig_hm.add_shape(
        type='rect',
        x0=-0.5, x1=len(KOMODS)-0.5,
        y0=PROVS.index('Sumsel')-0.5, y1=PROVS.index('Sumsel')+0.5,
        line=dict(color='#1a1a2e', width=2),
        fillcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
    st.caption("⬛ Baris Sumsel di-highlight. ⚪ NaN = anomali estimasi EGARCH (Sumbar–Daging Ayam).")

with col_line:
    st.markdown('<div class="section-title">📈 Line Chart Harga: Sumsel vs Pulau Sumatera</div>', unsafe_allow_html=True)

    FILE = os.path.join(os.path.dirname(__file__), 'final_dataset sriwijaya (1).xlsx')
    SHEET_MAP = {
        'Cabai Rawit':'Cabe Rawit','Cabai Merah':'Cabe Merah',
        'Bawang Merah':'Bawang Merah','Bawang Putih':'Bawang Putih',
        'Daging Sapi':'Daging Sapi','Daging Ayam':'Daging Ayam',
        'Minyak Goreng':'Minyak Goreng','Beras':'Beras',
    }
    komod_line = sel_komod if sel_komod != "Semua Komoditas" else "Cabai Rawit"

    try:
        df_raw = pd.read_excel(FILE, sheet_name=SHEET_MAP[komod_line])
        df_raw.columns = df_raw.columns.str.strip()
        df_raw['Date'] = pd.to_datetime(df_raw['Date'].str.strip(), dayfirst=True, errors='coerce')
        df_raw = df_raw.dropna(subset=['Date']).sort_values('Date')
        prov_cols = [c for c in df_raw.columns if c != 'Date']
        df_raw[prov_cols] = df_raw[prov_cols].apply(pd.to_numeric, errors='coerce')
        df_m = df_raw.set_index('Date').resample('ME').mean().reset_index()

        sumsel_col = [c for c in df_m.columns if 'sumsel' in c.lower()][0]
        df_m['mean_sum'] = df_m[prov_cols].mean(axis=1)
        df_m['std_sum']  = df_m[prov_cols].std(axis=1)

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_m['Date'], y=df_m['mean_sum']+df_m['std_sum'],
            fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_m['Date'], y=df_m['mean_sum']-df_m['std_sum'],
            fill='tonexty', mode='lines', line=dict(width=0),
            fillcolor='rgba(41,128,185,0.12)', name='±1 SD Sumatera', hoverinfo='skip'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_m['Date'], y=df_m['mean_sum'],
            mode='lines', name='Rata-rata Sumatera',
            line=dict(color='#2980b9', width=1.8),
            hovertemplate='%{y:,.0f}<extra>Rata-rata Sumatera</extra>'
        ))
        fig_line.add_trace(go.Scatter(
            x=df_m['Date'], y=df_m[sumsel_col],
            mode='lines', name='Sumatera Selatan',
            line=dict(color='#c0392b', width=2.4),
            hovertemplate='%{y:,.0f}<extra>Sumatera Selatan</extra>'
        ))
        fig_line.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='white', plot_bgcolor='white',
            legend=dict(font=dict(size=10), orientation='h', yanchor='bottom', y=1.02),
            xaxis=dict(gridcolor='#f5f5f5', tickfont=dict(size=10)),
            yaxis=dict(gridcolor='#f5f5f5', tickfont=dict(size=10),
                       tickformat=',.0f', title='Harga (Rp)'),
            title=dict(text=f'Harga {komod_line}', font=dict(size=12), x=0),
        )
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.warning(f"Upload file Excel ke folder yang sama dengan app.py. ({e})")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Log Sinyal Peringatan ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔔 Log Sinyal Peringatan Aktif</div>', unsafe_allow_html=True)

filtered_alerts = [
    a for a in ALERTS
    if (sel_prov == "Semua Provinsi" or a['prov'] == sel_prov) and
       (sel_komod == "Semua Komoditas" or a['komod'] == sel_komod)
]
if not filtered_alerts:
    filtered_alerts = ALERTS[:5]

cols_alert = st.columns(3)
for i, a in enumerate(filtered_alerts[:6]):
    with cols_alert[i % 3]:
        css_class = f"alert-{a['lv']}"
        icon = {'high':'🔴','med':'🟡','low':'🟢'}[a['lv']]
        st.markdown(f"""<div class="alert-box {css_class}">
        <div class="alert-title">{icon} {a['prov']} · {a['komod']} · {a['time']}</div>
        <div class="alert-body">{a['msg']}</div>
        </div>""", unsafe_allow_html=True)

# ── Rekomendasi Kebijakan ─────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("📋 Rekomendasi Kebijakan – Berdasarkan Hasil EWS", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""**🎯 Prioritas Jangka Pendek**
- Intensifikasi pemantauan harga harian untuk komoditas leverage tinggi (Telur Ayam, Daging Sapi, Bawang Putih)
- Aktifkan operasi pasar segera jika harga naik >5% dalam 3 hari berturut-turut
- Koordinasi lintas provinsi untuk komoditas dengan leverage asimetris negatif (shock turun justru memperparah volatilitas)
        """)
    with c2:
        st.markdown("""**🏗 Prioritas Jangka Menengah**
- Kembangkan buffer stock regional berbasis komoditas dengan leverage tertinggi per provinsi
- Perkuat konektivitas rantai pasok koridor Sumsel–Lampung–Jambi
- Integrasikan EWS ini ke dashboard TPID (Tim Pengendalian Inflasi Daerah) masing-masing provinsi
        """)
    with c3:
        st.markdown("""**📡 Penguatan Sistem**
- Perbarui estimasi EGARCH setiap kuartal menggunakan data PIHPS terbaru
- Tambahkan variabel cuaca dan kalender Ramadan/Lebaran sebagai variabel eksogen
- Sinkronisasi threshold EWS dengan target inflasi Bank Indonesia (2,5% ± 1%)
- Re-estimasi data anomali Sumbar (Daging Ayam) dengan metode robust GARCH
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-size:11px; color:#aaa; text-align:center; line-height:1.8;'>
Sistem Peringatan Dini Volatilitas Harga Pangan Pulau Sumatera &nbsp;|&nbsp;
Berbasis Model EGARCH(1,1) &nbsp;|&nbsp;
Bank Indonesia Sumatera Selatan – Call for Paper 2024<br>
Data: PIHPS Nasional · Periode: Juli 2017 – Juli 2023 · Shapefile: Natural Earth 1:10m
</div>
""", unsafe_allow_html=True)
