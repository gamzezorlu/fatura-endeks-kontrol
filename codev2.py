import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Doğalgaz Anomali Tespit", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# 📅 Ay isimleri eşleştirmesi
# ------------------------------------------------------------
MONTH_MAP = {
    'Oca': 1, 'Şub': 2, 'Mar': 3, 'Nis': 4, 'May': 5, 'Haz': 6,
    'Tem': 7, 'Ağu': 8, 'Eyl': 9, 'Eki': 10, 'Kas': 11, 'Ara': 12
}
REVERSE_MONTH_MAP = {v: k for k, v in MONTH_MAP.items()}

# ------------------------------------------------------------
# 🔢 Yardımcı Fonksiyonlar
# ------------------------------------------------------------
def parse_date(date_str):
    """Tarih string'ini parse et (örn: Ocak 23 -> 2023, 1)"""
    try:
        if pd.isna(date_str):
            return None, None
        date_str = str(date_str).strip()

        if ' ' in date_str:
            parts = date_str.split(' ')
        elif '.' in date_str:
            parts = date_str.split('.')
        else:
            return None, None

        if len(parts) != 2:
            return None, None

        month_name = parts[0].strip()[:3].capitalize()
        year_short = parts[1].strip()

        month_replacements = {'Sub': 'Şub', 'Agu': 'Ağu'}
        month_name = month_replacements.get(month_name, month_name)

        if month_name not in MONTH_MAP:
            return None, None

        month = MONTH_MAP[month_name]
        year = 2000 + int(year_short)
        return year, month
    except Exception:
        return None, None


def get_consumption(df, tesisat_no, year, month):
    """Belirli tesisat, yıl ve ay için tüketim değerini getir"""
    filtered = df[(df['tesisat_no'] == tesisat_no) &
                  (df['yil'] == year) &
                  (df['ay'] == month)]
    if filtered.empty:
        return None
    val = filtered['tuketim'].values[0]
    if pd.isna(val) or val == 0:
        return None
    return float(val)


def assign_segment(avg_consumption):
    """Tüketim ortalamasına göre segment ve eşik belirle"""
    if pd.isna(avg_consumption) or avg_consumption == 0:
        return 'A', 50
    elif avg_consumption < 100:
        return 'A', 50
    elif avg_consumption < 300:
        return 'B', 40
    elif avg_consumption < 1000:
        return 'C', 30
    else:
        return 'D', 25


def analyze_facility(df, tesisat_no, analysis_year, analysis_month, threshold):
    """Ana analiz fonksiyonu"""
    current_val = get_consumption(df, tesisat_no, analysis_year, analysis_month)
    prev1_month = 12 if analysis_month == 1 else analysis_month - 1
    prev1_year = analysis_year - 1 if analysis_month == 1 else analysis_year
    prev_year1_val = get_consumption(df, tesisat_no, analysis_year - 1, analysis_month)

    # Ortalama ve segment belirle
    recent_data = df[(df['tesisat_no'] == tesisat_no) &
                     (df['tuketim'] > 0) &
                     (df['tuketim'].notna())]
    avg_consumption = recent_data['tuketim'].tail(6).mean() if not recent_data.empty else 0
    segment, seg_threshold = assign_segment(avg_consumption)

    anomaly_flag, anomaly_reason = False, ""
    change_percent = 0

    if current_val and prev1_val := get_consumption(df, tesisat_no, prev1_year, prev1_month):
        change_percent = ((current_val - prev1_val) / prev1_val) * 100
        if abs(change_percent) >= seg_threshold:
            anomaly_flag = True
            anomaly_reason = f"Aydan Aya Değişim %{change_percent:.1f}"

    elif current_val and prev_year1_val:
        change_percent = ((current_val - prev_year1_val) / prev_year1_val) * 100
        if abs(change_percent) >= seg_threshold:
            anomaly_flag = True
            anomaly_reason = f"Yıllık Değişim %{change_percent:.1f}"

    return {
        'tesisat_no': tesisat_no,
        'segment': segment,
        'ortalama_tuketim': round(avg_consumption, 2),
        'mevcut_tuketim': round(current_val or 0, 2),
        'degisim_%': round(change_percent, 1),
        'anomali': "VAR" if anomaly_flag else "YOK",
        'anlam': anomaly_reason
    }

# ------------------------------------------------------------
# 🌐 Streamlit Arayüzü
# ------------------------------------------------------------
st.title("📊 Doğalgaz Tüketim Anomali Tespit Sistemi")
st.caption("**Excel çıktılı sürüm** – Her satır bir ay verisini temsil eder.")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Excel dosyasını yükleyin", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # Otomatik sütun tespiti
    tesisat_col = next((c for c in df_raw.columns if 'tesisat' in c), None)
    tarih_col = next((c for c in df_raw.columns if 'tarih' in c or 'ay' in c or 'donem' in c), None)
    tuketim_col = next((c for c in df_raw.columns if 'tuketim' in c or 'm3' in c or 'miktar' in c), None)

    if not all([tesisat_col, tarih_col, tuketim_col]):
        st.error("❌ Sütun isimleri otomatik algılanamadı. Lütfen kontrol edin.")
        st.stop()

    df = df_raw[[tesisat_col, tarih_col, tuketim_col]].copy()
    df.columns = ['tesisat_no', 'tarih', 'tuketim']
    df['yil'], df['ay'] = zip(*df['tarih'].apply(parse_date))
    df = df[(df['yil'].notna()) & (df['ay'].notna())]
    df['yil'] = df['yil'].astype(int)
    df['ay'] = df['ay'].astype(int)
    df['tuketim'] = pd.to_numeric(df['tuketim'], errors='coerce')

    if df.empty:
        st.error("❌ Veri işlenemedi! Tarih formatını kontrol edin (örnek: Ocak 23, Şub 23).")
        st.stop()

    st.success(f"✅ {df['tesisat_no'].nunique()} tesisat, {len(df)} satır veri başarıyla yüklendi.")

    # Analiz parametreleri
    col1, col2 = st.columns(2)
    years = sorted(df['yil'].unique(), reverse=True)
    with col1:
        year = st.selectbox("Analiz yılı", years)
    with col2:
        month = st.selectbox("Analiz ayı", list(REVERSE_MONTH_MAP.keys()),
                             format_func=lambda x: REVERSE_MONTH_MAP[x], index=9)

    if st.button("🔍 Analizi Başlat", type="primary", use_container_width=True):
        with st.spinner("Analiz yapılıyor..."):
            results = [analyze_facility(df, t, year, month, 20) for t in df['tesisat_no'].unique()]
            df_results = pd.DataFrame(results)

            st.markdown("### 🚨 Anomali Sonuçları")
            st.dataframe(df_results, use_container_width=True)

            # 🔽 Excel Çıktısı
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_results.to_excel(writer, sheet_name='Anomali Analizi', index=False)
            st.download_button(
                label="📥 Excel Sonuçlarını İndir",
                data=buffer.getvalue(),
                file_name=f"anomali_sonuclari_{datetime.now():%Y%m%d_%H%M}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("👆 Lütfen Excel dosyanızı yükleyin.")
