import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="F1 PoleSight",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (DESIGN SYSTEM F1)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');

    /* BACKGROUND */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #3a3a3a 100%);
        color: #ffffff;
        font-family: 'Titillium Web', sans-serif;
    }

    /* HEADER HIDDEN */
    header {visibility: hidden;}
    .block-container {
        padding-top: 80px;
        padding-left: 2rem;
        padding-right: 2rem;
        max_width: 100%;
    }

    /* NAVBAR */
    .f1-header {
        position: fixed; top: 0; left: 0; width: 100%; height: 80px;
        background-color: #000000; border-bottom: 4px solid #e10600;
        display: flex; align-items: center; padding: 0 40px; z-index: 99999;
    }
    .f1-logo { height: 35px; margin-right: 30px; }
    .nav-title {
        color: #ffffff; font-size: 28px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
    }

    /* TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, label { color: #ffffff !important; }
    .stMarkdown { color: #ffffff !important; }

    /* KPI CARDS */
    .kpi-card {
        background-color: rgba(0, 0, 0, 0.6);
        border: 1px solid #333;
        border-left: 5px solid #e10600;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .kpi-title { font-size: 14px; color: #cccccc; text-transform: uppercase; font-weight: 600; }
    .kpi-value { font-size: 36px; color: #ffffff; font-weight: 700; margin-top: 5px; }
    .kpi-sub { font-size: 14px; color: #e10600; font-weight: 600; margin-top: 5px; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px; font-weight: bold; color: white;
    }
    
    /* INSIGHT BOX */
    .insight-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #e10600;
        padding: 15px;
        margin-top: 20px;
        border-radius: 4px;
    }
    
    /* INPUT FIELD COLOR & BOLD FIX */
    /* 1. Teks yang sedang terpilih (di dalam kotak) */
    .stSelectbox [data-baseweb="select"] > div:first-child [data-testid="stMarkdownContainer"] p {
        color: #333333 !important; 
        font-weight: 800 !important; /* Membuat TEBAL di dalam kotak */
    }
    
    /* 2. Teks pilihan saat dropdown dibuka */
    div[data-baseweb="popover"] li div {
        font-weight: 700 !important; /* Membuat TEBAL di daftar pilihan */
        color: #333333 !important; /* Memastikan warna teks hitam/abu gelap */
    }
    
    /* REVISI TABEL: Header Merah F1 */
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        background-color: #DC0000 !important; /* Merah F1 */
        color: white !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# NAVBAR HTML INJECTION
st.markdown("""
    <div class="f1-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg" class="f1-logo">
        <span class="nav-title">POLESIGHT ANALYTICS</span>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA CONNECTION (FORCE LOWERCASE)
# ==========================================
@st.cache_data(ttl=600)
def load_data(query):
    db_config = {"host": "localhost", "port": "5433", "database": "f1_datawarehouse",
                 "user": "warehouse_user", "password": "warehouse_password"}
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        df.columns = [col.lower() for col in df.columns]
        return df
    except: return pd.DataFrame()

try:
    df_raw = load_data("SELECT * FROM lap_telemetry_advanced")
except:
    df_raw = load_data("SELECT * FROM lap_telemetry_full")

if df_raw.empty:
    st.stop()

# ==========================================
# 4. ML TRAINING
# ==========================================
@st.cache_resource
def train_full_grid_model(df):
    # Gunakan kolom lowercase (konsisten dengan load_data)
    race_data = df.groupby(['year', 'circuit', 'driver', 'team']).agg({
        'laptime_sec': 'min', 'tracktemp': 'mean', 'max_speed_tel': 'max'
    }).reset_index()
    
    le_circuit = LabelEncoder()
    race_data['circuit_id'] = le_circuit.fit_transform(race_data['circuit'])
    le_driver = LabelEncoder()
    race_data['driver_id'] = le_driver.fit_transform(race_data['driver'])
    le_team = LabelEncoder()
    race_data['team_id'] = le_team.fit_transform(race_data['team'])
    
    X = race_data[['year', 'circuit_id', 'tracktemp', 'driver_id', 'team_id']]
    y = race_data['laptime_sec']
    
    model_time = RandomForestRegressor(n_estimators=100, random_state=42)
    model_time.fit(X, y)
    
    model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_clf.fit(X, race_data['driver_id']) # Target klasifikasi driver
    
    latest_year = df['year'].max()
    current_grid = race_data[race_data['year'] == latest_year][['driver', 'team', 'driver_id', 'team_id']].drop_duplicates('driver')
    
    # Rename HANYA untuk tampilan tabel prediksi (agar cantik), bukan untuk analisis lain
    current_grid_display = current_grid.rename(columns={'driver': 'Driver', 'team': 'Team'})
    
    # RETURN 5 VARIABEL (PENTING!)
    return model_time, model_clf, le_circuit, le_driver, current_grid

# PEMANGGILAN FUNGSI (5 VARIABEL)
model_time, model_clf, le_circuit, le_driver, current_grid = train_full_grid_model(df_raw)

# ==========================================
# 5. FILTERS
# ==========================================
st.write("") 

col_filter1, col_filter2, col_filter3 = st.columns([1, 2, 4])
with col_filter1:
    years = sorted(df_raw['year'].unique(), reverse=True)
    sel_year = st.selectbox("MUSIM", years)
    df_year = df_raw[df_raw['year'] == sel_year]
with col_filter2:
    circuits = sorted(df_year['circuit'].unique())
    sel_circuit = st.selectbox("CIRCUIT", circuits)
    df = df_year[df_year['circuit'] == sel_circuit]

# ==========================================
# 6. HEADER & KPI
# ==========================================
st.markdown(f"### GRAND PRIX {str(sel_circuit).upper()}")

best_lap = df['laptime_sec'].min()
pole_row = df.loc[df['laptime_sec'].idxmin()]
pole_driver = pole_row['driver']
top_speed = df['max_speed_tel'].max()
avg_stress = df['tire_stress_index'].mean() if 'tire_stress_index' in df.columns else 0
total_laps = len(df)

k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f"""<div class="kpi-card"><div class="kpi-title">POLE POSITION</div><div class="kpi-value">{best_lap:.3f} s</div><div class="kpi-sub">{pole_driver}</div></div>""", unsafe_allow_html=True)
with k2: st.markdown(f"""<div class="kpi-card"><div class="kpi-title">TOP SPEED</div><div class="kpi-value">{top_speed:.1f}</div><div class="kpi-sub">KM/H</div></div>""", unsafe_allow_html=True)
with k3: st.markdown(f"""<div class="kpi-card"><div class="kpi-title">TIRE STRESS</div><div class="kpi-value">{avg_stress:.1f}</div><div class="kpi-sub">AVERAGE LOAD</div></div>""", unsafe_allow_html=True)
with k4: st.markdown(f"""<div class="kpi-card"><div class="kpi-title">TOTAL DATA</div><div class="kpi-value">{total_laps:,}</div><div class="kpi-sub">LAPS RECORDED</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 7. TAB CONTENT
# ==========================================
t1, t2, t3, t4 = st.tabs(["RACE PREDICTOR", "PERFORMANCE", "DRIVER STYLE", "TIRE DATA"])

def dark_plot(fig):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color="white"), 
        legend=dict(font=dict(color="white", weight="bold")),
        title=dict(font=dict(color="white", size=20, family="Titillium Web"))
    )
    return fig

# === TAB 1: AI PREDICTOR ===
with t1:
    col_sim, col_res = st.columns([1, 2])
    with col_sim:
        st.markdown("#### SIMULATION SETTINGS")
        p_year = st.number_input("TARGET TAHUN", 2025, 2030, 2025)
        p_temp = st.slider("TRACK TEMP (°C)", 20.0, 60.0, 35.0)
        p_circuit = st.selectbox("TARGET SIRKUIT PREDIKSI", le_circuit.classes_, index=0)
        
        st.markdown("""<style>div.stButton > button {background-color: #e10600; color: white; border: none; width: 100%;}</style>""", unsafe_allow_html=True)
        run_sim = st.button("**RUN PREDICTION**", key="run_sim_btn") 
        
    with col_res:
        if run_sim:
            try:
                # Transformasi Input (Lowercase)
                c_code = le_circuit.transform([p_circuit])[0]
                sim_data = current_grid.copy()
                sim_data['year'] = p_year
                sim_data['circuit_id'] = c_code
                sim_data['tracktemp'] = p_temp
                
                # Input Model (Lowercase Columns)
                X_pred = sim_data[['year', 'circuit_id', 'tracktemp', 'driver_id', 'team_id']]
                
                # Prediksi
                sim_data['Predicted_Time'] = model_time.predict(X_pred)
                
                # Sorting
                final_grid = sim_data.sort_values('Predicted_Time').reset_index(drop=True)
                
                # 1. Tambah Kolom POS (Nomor Urut)
                final_grid.insert(0, 'POS', range(1, len(final_grid) + 1))

                # 2. Format Waktu & Gap
                final_grid['Gap'] = final_grid['Predicted_Time'] - final_grid['Predicted_Time'].iloc[0]
                final_grid['Time'] = final_grid['Predicted_Time'].apply(lambda x: f"{int(x//60)}:{x%60:06.3f}")
                final_grid['Gap'] = final_grid['Gap'].apply(lambda x: f"+{x:.3f} s" if x > 0 else "-")
                
                # 3. Rename Kolom untuk Tampilan (lowercase -> Title Case)
                final_grid = final_grid.rename(columns={'driver': 'Driver', 'team': 'Team'})
                
                # 4. Teks Pemenang (Bold)
                winner = final_grid.iloc[0]
                st.success(f" PREDICTED POLE: **{winner['Driver']}** ({winner['Team']})")
                
                # 5. Fungsi Warna (Baris 1 Emas)
                def highlight_winner(row):
                    if row.name == 0: # Jika Index 0 (Juara)
                        return ['background-color: #FFD700; color: black; font-weight: bold'] * len(row)
                    else:
                        return [''] * len(row)

                # 6. Render Tabel (Style + Config)
                st.dataframe(
                    final_grid[['POS', 'Driver', 'Team', 'Time', 'Gap']].style.apply(highlight_winner, axis=1),
                    hide_index=True, 
                    use_container_width=True,
                    column_config={"POS": st.column_config.NumberColumn("POS", format="%d", width="small")}
                )
                
            except Exception as e:
                st.error(f"Gagal memprediksi: {e}")
        else:
            st.info("Set parameters and click Run to simulate future race results.")

# === TAB 2: PERFORMANCE ===
with t2:
    # 1. SPEED ANALYSIS
    st.markdown("#### SPEED ANALYSIS")
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        fig_perf = px.scatter(df, x="max_speed_tel", y="laptime_sec", color="team", size="full_throttle_pct",
            hover_data=["driver"], title="Top Speed vs Lap Time Distribution", opacity=0.8)
        st.plotly_chart(dark_plot(fig_perf), use_container_width=True)
    
    with col_p2:
        correlation = df['max_speed_tel'].corr(df['laptime_sec'])
        if correlation < -0.5: 
            insight_speed = "Power Track (Sirkuit Cepat): Data menunjukkan mobil dengan Top Speed tinggi punya keuntungan besar di sini. Penerapan Akselerasi Penuh (Full Throttle) di trek lurus adalah kunci utama untuk menang."
        elif correlation > -0.5 and correlation < 0.2: 
            insight_speed = " **Technical Track (Sirkuit Teknikal): Kecepatan lurus bukan segalanya. Kunci kemenangan di sini adalah kelincahan di tikungan (Aerodinamika) dan skill pengereman, bukan sekadar mesin kencang."
        else: 
            insight_speed = " Balanced Track: Sirkuit ini unik. Data menunjukkan butuh keseimbangan sempurna antara mesin kencang dan mobil yang lincah di tikungan."
        st.markdown(f"""<div class="insight-box"><b> SPEED INSIGHT:</b><br>Korelasi: {correlation:.2f}<br><br>{insight_speed}</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 2. SECTOR ANALYSIS (FIX CRASH MONTREAL)
    st.markdown("#### SECTOR ANALYSIS")
    top_5_drivers = df.sort_values("laptime_sec")['driver'].unique()[:5]
    df_top5 = df[df['driver'].isin(top_5_drivers)]
    
    sec_cols = ['sector1_sec', 'sector2_sec', 'sector3_sec']
    # SAFETY CHECK: Hanya proses kalau ada data yang tidak NaN
    valid_sec_cols = [c for c in sec_cols if c in df_top5.columns and df_top5[c].notna().any()]

    if len(valid_sec_cols) == 3:
        sector_data = df_top5.groupby('driver')[valid_sec_cols].mean().reset_index()
        sector_melt = sector_data.melt(id_vars='driver', value_vars=valid_sec_cols, var_name='Sector', value_name='Seconds')
        
        c_s1, c_s2 = st.columns([3, 1])
        with c_s1:
            fig_sec = px.bar(sector_melt, x="driver", y="Seconds", color="Sector", 
                             title="Perbandingan Waktu Sektor", text_auto='.2f',
                             color_discrete_map={'sector1_sec': '#FF4B4B', 'sector2_sec': '#1C83E1', 'sector3_sec': '#FFA421'})
            st.plotly_chart(dark_plot(fig_sec), use_container_width=True)
        with c_s2:
            try:
                # HITUNG RANGE: Selisih antara pembalap paling lambat vs paling cepat di sektor itu
                ranges = {}
                for col in valid_sec_cols:
                    ranges[col] = sector_data[col].max() - sector_data[col].min()
                
                # Cari sektor dengan selisih terbesar (Sektor Pembeda)
                decisive_sector = max(ranges, key=ranges.get)
                gap_val = ranges[decisive_sector]
                
                # Format Teks
                txt = decisive_sector.replace('_sec', '').replace('sector', 'Sektor ')
                
                st.markdown(f"""
                <div class="insight-box">
                    <b> SECTOR INSIGHT:</b><br>
                    Sektor ini menjadi pembeda utama dengan gap <b>{gap_val:.3f} detik</b> antara pembalap.<br><br>
                    Kunci kemenangan di sirkuit ini terletak pada penguasaan <b>{txt.upper()}</b>.
                </div>""", unsafe_allow_html=True)
                
            except Exception as e:
                # Tampilkan error spesifik biar ketahuan (bukan cuma "tidak tersedia")
                st.info(f"Insight sektor belum tersedia. ({e})")
    else:
        st.warning("Data Sektor tidak lengkap (Wet Race/Sensor Issue).")

    st.markdown("---")

    # 3. CONSISTENCY CHECK
    st.markdown("#### CONSISTENCY CHECK")
    top_10 = df.sort_values("laptime_sec")['driver'].unique()[:10]
    df_top10 = df[df['driver'].isin(top_10)]
    
    c_c1, c_c2 = st.columns([3, 1])
    with c_c1:
        fig_cons = px.box(df_top10, x="driver", y="laptime_sec", color="team", title="Sebaran Waktu Lap")
        st.plotly_chart(dark_plot(fig_cons), use_container_width=True)
    with c_c2:
        try:
            std_dev = df_top10.groupby('driver')['laptime_sec'].std().sort_values()
            if not std_dev.empty:
                most_consistent = std_dev.index[0]
                least_consistent = std_dev.index[-1]
                
                st.markdown(f"""
                <div class="insight-box">
                    <b> PACE ANALYSIS:</b>
                    <ul style="margin-top:10px;padding-left:20px;">
                        <li style="margin-bottom:10px;">
                             <b>{most_consistent.upper()}</b>:<br>
                            Menunjukkan Lap Time Consistency tertinggi. Mampu mempertahankan ritme lap yang presisi dan deviasi waktu yang sangat minimal.
                        </li>
                        <li>
                             <b>{least_consistent.upper()}</b>:<br>
                            Terdeteksi Volatilitas Pace Tinggi (High Volatility). Sulit mempertahankan ritme lap, rentan terhadap Driver Errors, atau kesulitan dalam Traffic Management.
                        </li>
                    </ul>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Data driver tidak cukup.")
        except:
            st.info("Data konsistensi tidak cukup.")

# === TAB 3: DRIVER STYLE (INTEGRASI FOTO) ===
with t3:
    if 'driver_aggression' in df.columns:
        st.markdown("#### PETA GAYA BALAP & PROFIL DRIVER")
        
        # 1. Grafik Scatter Utama (Tetap Ada)
        fig_style = px.scatter(df, x="driver_aggression", y="tire_stress_index", color="team", text="driver", 
                               title="Aggressiveness vs Tire Wear Matrix")
        avg_x = df['driver_aggression'].mean()
        avg_y = df['tire_stress_index'].mean()
        fig_style.add_vline(x=avg_x, line_dash="dash", line_color="gray")
        fig_style.add_hline(y=avg_y, line_dash="dash", line_color="gray")
        fig_style.update_traces(textposition='top center')
        st.plotly_chart(dark_plot(fig_style), use_container_width=True)
        
        st.markdown("---")
        st.markdown("####  DRIVER DEEP DIVE")
        
        # 2. Selectbox untuk Memilih Driver
        driver_list = sorted(df['driver'].unique())
        selected_driver_profile = st.selectbox("Pilih Driver untuk Analisis Detail:", driver_list, key="driver_profile_select")
        
        # Filter Data untuk Driver Terpilih
        driver_stats = df[df['driver'] == selected_driver_profile].iloc[0]
        
        # 3. Layout 2 Kolom (Foto Kiri, Stats Kanan)
        col_img, col_info = st.columns([1, 2])
        
        with col_img:
            # Load Gambar
            try:
                # Path relatif ke folder images
                img_path = f"dashboard/images/drivers/{selected_driver_profile}.png"
                st.image(img_path, caption=selected_driver_profile, use_container_width=True)
            except:
                # Fallback kalau foto belum ada
                st.warning(f"Foto {selected_driver_profile} belum tersedia.")
                # st.image("https://placehold.co/200x200?text=No+Image", width=200) # Opsional: Placeholder
        
        with col_info:
            # Tampilkan Statistik Spesifik
            aggro_val = driver_stats['driver_aggression']
            stress_val = driver_stats['tire_stress_index']
            
            # Logika Penilaian Sederhana
            style_desc = "Aggressive" if aggro_val > avg_x else "Smooth"
            tire_desc = "High Load" if stress_val > avg_y else "Low Load"
            
            st.markdown(f"""
            <div class="insight-box" style="border-left: 5px solid #e10600;">
                <h3 style="margin-top:0; color: white;">{selected_driver_profile} <span style="font-size: 18px; color: #b0b0b0;">({driver_stats['team']})</span></h3>
                <p><b>GAYA BALAP:</b> {style_desc} & {tire_desc}</p>
                <hr style='border-top: 1px solid #333;'>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <span style="color:#b0b0b0; font-size: 12px;">AGGRESSION SCORE</span><br>
                        <span style="font-size: 24px; font-weight: bold;">{aggro_val:.1f}</span>
                    </div>
                    <div>
                        <span style="color:#b0b0b0; font-size: 12px;">TIRE STRESS IDX</span><br>
                        <span style="font-size: 24px; font-weight: bold;">{stress_val:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4. Insight Global
        try:
            aggro = df.loc[df['driver_aggression'].idxmax()]['driver']
            smooth = df.loc[df['tire_stress_index'].idxmin()]['driver']
            corr_style = df['driver_aggression'].corr(df['tire_stress_index'])
            
            insight_saran = "Sangat sensitif: Terdapat korelasi kuat antara agresi dan kehausan ban." if corr_style > 0.5 else "Rendah sensitivitas: Driver dapat menekan batas lebih jauh."

            st.markdown(f"""
            <div class="insight-box" style="margin-top: 20px;">
                <b>GLOBAL STYLE INSIGHT:</b><br>
                <br>
                 <b>Most Aggressive:</b> <b>{aggro.upper()}</b><br>
                 <b>Most Efficient:</b> <b>{smooth.upper()}</b><br>
                <hr style='border-top: 1px solid #444;'>
                <p style='margin:0;'><b>Rekomendasi Strategi:</b> {insight_saran}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except:
             st.info("Insight global tidak tersedia.")
            
    else:
        st.warning("Data Aggression belum tersedia.")

# === TAB 4: TIRE DATA ===
with t4:
    st.markdown("#### ANALISIS DEGRADASI BAN")
    tire_trend = df.groupby(['tyrelife', 'compound'])['laptime_sec'].mean().reset_index()
    fig_tire = px.line(tire_trend, x="tyrelife", y="laptime_sec", color="compound", title="Tire Degradation Curve",
                       color_discrete_map={"SOFT": "#FF3333", "MEDIUM": "#FFEB3B", "HARD": "#FFFFFF", "INTERMEDIATE": "#39B54A", "WET": "#005AFF"})
    st.plotly_chart(dark_plot(fig_tire), use_container_width=True)
    
    deg_correlation = df['tyrelife'].corr(df['laptime_sec'])
    
    # 1. Tentukan Status Degradasi
    status = ""
    strategy_rec = ""
    
    if deg_correlation > 0.5:
        status = "CRITICAL / HIGH"
        strategy_rec = "CRITICAL: Degradasi tinggi, direkomendasikan strategi minimum dua (2) kali pit stop, atau pertimbangkan ban Hard/Medium."
    elif deg_correlation > 0.2:
        status = "MODERATE / LINEAR"
        strategy_rec = "NORMAL: Degradasi stabil dan dapat diprediksi. Strategi satu (1) kali pit stop per set ban utama sangat layak dipertimbangkan."
    else:
        status = "NEGLIGIBLE / NEGATIVE"
        strategy_rec = "LOW RISK: Degradasi sangat rendah. Tim dapat mempertimbangkan untuk melakukan overcut atau memperpanjang stint (mempertahankan ban lama)."

    # 2. Tampilkan Hasil
    st.markdown(f"""
    <div class="insight-box">
        <b> STRATEGY REPORT:</b><br>
        Tingkat Degradasi Ban: <b>{status}</b><br><br>
        {strategy_rec}
    </div>""", unsafe_allow_html=True)