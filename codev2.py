import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Fatura & Endeks Kontrol", layout="wide")

st.title("📊 Fatura & Endeks Kontrol Uygulaması")

# Türkçe ay isimleri
MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12
}

# Tarih parse fonksiyonu
def parse_date(date_str):
    if pd.isna(date_str):
        return None, None
    date_str = str(date_str).lower().replace(".", "").strip()
    for m_name, m_num in MONTHS.items():
        if m_name in date_str:
            try:
                year = int("20" + date_str[-2:])
                return year, m_num
            except:
                return None, None
    try:
        dt = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
        if pd.notna(dt):
            return dt.year, dt.month
    except:
        pass
    return None, None

uploaded_file = st.file_uploader("📎 Excel dosyasını yükle (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    # Olası sütun adları
    tesisat_cols = ['tesisat', 'tesisat_no', 'tesisatno']
    tarih_cols = ['tarih', 'ay', 'donem', 'tarihi']
    tuketim_cols = ['tuketim', 'm3', 'miktar']

    # Gerçek sütun adlarını bul
    tesisat_col = next((c for c in tesisat_cols if c in df.columns), None)
    tarih_col = next((c for c in tarih_cols if c in df.columns), None)
    tuketim_col = next((c for c in tuketim_cols if c in df.columns), None)

    if not all([tesisat_col, tarih_col, tuketim_col]):
        st.error("⚠️ Gerekli sütunlar bulunamadı! Lütfen sütun adlarını kontrol edin.")
    else:
        # Tarihleri çözümle
        df[['year', 'month']] = df[tarih_col].apply(lambda x: pd.Series(parse_date(x)))
        df = df.dropna(subset=['year', 'month'])
        df[tuketim_col] = pd.to_numeric(df[tuketim_col], errors='coerce')
        df = df.dropna(subset=[tuketim_col])

        df = df.astype({'year': int, 'month': int})
        df = df.sort_values(by=[tesisat_col, 'year', 'month']).reset_index(drop=True)

        def get_consumption(df, tesisat_no, year, month):
            row = df[(df[tesisat_col] == tesisat_no) &
                     (df['year'] == year) &
                     (df['month'] == month)]
            if not row.empty:
                return row.iloc[0][tuketim_col]
            return None

        results = []

        for i, row in df.iterrows():
            tesisat_no = row[tesisat_col]
            year = row['year']
            month = row['month']
            current_val = row[tuketim_col]
            anomaly = ""

            # Önceki ayların bilgileri
            prev1_year, prev1_month = (year, month - 1) if month > 1 else (year - 1, 12)
            prev2_year, prev2_month = (year, month - 2) if month > 2 else (year - 1, month + 10)
            prev3_year, prev3_month = (year, month - 3) if month > 3 else (year - 1, month + 9)

            prev1_val = get_consumption(df, tesisat_no, prev1_year, prev1_month)
            prev2_val = get_consumption(df, tesisat_no, prev2_year, prev2_month)
            prev3_val = get_consumption(df, tesisat_no, prev3_year, prev3_month)

            if current_val and prev1_val:
                increase_ratio = current_val / prev1_val if prev1_val > 0 else None
                if increase_ratio and increase_ratio > 1.5:
                    anomaly = "⚠️ Ani artış"
            elif current_val and prev2_val:
                increase_ratio = current_val / prev2_val if prev2_val > 0 else None
                if increase_ratio and increase_ratio > 1.5:
                    anomaly = "⚠️ 2 ay öncesine göre artış"
            elif current_val and prev3_val:
                increase_ratio = current_val / prev3_val if prev3_val > 0 else None
                if increase_ratio and increase_ratio > 1.5:
                    anomaly = "⚠️ 3 ay öncesine göre artış"

            results.append({
                "tesisat": tesisat_no,
                "yıl": year,
                "ay": month,
                "tüketim": current_val,
                "durum": anomaly
            })

        results_df = pd.DataFrame(results)

        st.success(f"✅ {df[tesisat_col].nunique()} tesisat, {len(df)} satır veri işlendi.")
        st.dataframe(results_df, use_container_width=True)

        # Excel çıktısı
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Sonuçlar')
        st.download_button(
            label="📥 Excel çıktısını indir",
            data=buffer.getvalue(),
            file_name="fatura_kontrol_sonuc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
