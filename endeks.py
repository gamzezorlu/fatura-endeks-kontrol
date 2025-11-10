import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

st.set_page_config(page_title="Doğalgaz Anomali Tespit", page_icon="📊", layout="wide")

# Ay isimleri mapping
MONTH_MAP = {
    'Oca': 1, 'Şub': 2, 'Mar': 3, 'Nis': 4, 'May': 5, 'Haz': 6,
    'Tem': 7, 'Ağu': 8, 'Eyl': 9, 'Eki': 10, 'Kas': 11, 'Ara': 12
}

REVERSE_MONTH_MAP = {
    1: 'Oca', 2: 'Şub', 3: 'Mar', 4: 'Nis', 5: 'May', 6: 'Haz',
    7: 'Tem', 8: 'Ağu', 9: 'Eyl', 10: 'Eki', 11: 'Kas', 12: 'Ara'
}

def parse_month_column(col):
    """Sütun adından yıl ve ay bilgisini çıkar (örn: Oca.23 -> 2023, 1)"""
    try:
        parts = col.split('.')
        if len(parts) != 2:
            return None
        
        month_name = parts[0].capitalize()
        year_short = parts[1]
        
        if month_name not in MONTH_MAP:
            return None
        
        month = MONTH_MAP[month_name]
        year = 2000 + int(year_short)
        
        return {'year': year, 'month': month, 'original': col}
    except:
        return None

def get_month_column(year, month):
    """Yıl ve aydan sütun adı oluştur (örn: 2023, 1 -> Oca.23)"""
    month_name = REVERSE_MONTH_MAP.get(month)
    if not month_name:
        return None
    year_short = str(year)[2:]
    return f"{month_name}.{year_short}"

def get_value(row, col_name):
    """Satırdan değer al, boş/0 kontrolü yap"""
    if col_name not in row.index:
        return None
    val = row[col_name]
    if pd.isna(val) or val == '' or val == 0:
        return None
    return float(val)

def calculate_trend(v1, v2, v3):
    """3 değer arasındaki ortalama trendi hesapla"""
    if v1 is None or v2 is None or v3 is None:
        return None
    diff1 = v2 - v1
    diff2 = v3 - v2
    return (diff1 + diff2) / 2

def analyze_facility(row, analysis_month, threshold, tesisat_col):
    """Tek bir tesisat için anomali analizi yap"""
    parsed = parse_month_column(analysis_month)
    if not parsed:
        return None
    
    year = parsed['year']
    month = parsed['month']
    current_col = analysis_month
    
    # Tesisat numarasını al
    tesisat_no = row[tesisat_col] if tesisat_col in row.index else 'Bilinmiyor'
    
    # Mevcut ay değeri
    current_val = get_value(row, current_col)
    
    # Önceki 2 ay
    prev1_month = 12 if month == 1 else month - 1
    prev1_year = year - 1 if month == 1 else year
    prev2_month = 11 if month <= 2 else (12 if month == 2 else month - 2)
    prev2_year = year - 1 if month <= 2 else year
    
    prev1_col = get_month_column(prev1_year, prev1_month)
    prev2_col = get_month_column(prev2_year, prev2_month)
    
    prev1_val = get_value(row, prev1_col) if prev1_col else None
    prev2_val = get_value(row, prev2_col) if prev2_col else None
    
    # Önceki 2 yılın aynı ayı
    prev_year1_col = get_month_column(year - 1, month)
    prev_year2_col = get_month_column(year - 2, month)
    
    prev_year1_val = get_value(row, prev_year1_col) if prev_year1_col else None
    prev_year2_val = get_value(row, prev_year2_col) if prev_year2_col else None
    
    # Sonraki 2 ay (trend için)
    next1_month = 1 if month == 12 else month + 1
    next1_year = year + 1 if month == 12 else year
    next2_month = 2 if month >= 11 else (1 if month == 11 else month + 2)
    next2_year = year + 1 if month >= 11 else year
    
    next1_col = get_month_column(next1_year, next1_month)
    next2_col = get_month_column(next2_year, next2_month)
    
    next1_val = get_value(row, next1_col) if next1_col else None
    next2_val = get_value(row, next2_col) if next2_col else None
    
    # 2024 ve 2023 için aynı aylar (trend)
    y2024_m1_col = get_month_column(year - 1, next1_month)
    y2024_m2_col = get_month_column(year - 1, next2_month)
    y2023_m1_col = get_month_column(year - 2, next1_month)
    y2023_m2_col = get_month_column(year - 2, next2_month)
    
    y2024_m1_val = get_value(row, y2024_m1_col) if y2024_m1_col else None
    y2024_m2_val = get_value(row, y2024_m2_col) if y2024_m2_col else None
    y2023_m1_val = get_value(row, y2023_m1_col) if y2023_m1_col else None
    y2023_m2_val = get_value(row, y2023_m2_col) if y2023_m2_col else None
    
    # ANALİZ 1: Önceki 2 ay ile karşılaştırma
    anomaly1 = {'detected': False, 'type': None, 'reason': '', 'change': None}
    
    if current_val is not None and prev1_val is not None:
        change_percent = ((current_val - prev1_val) / prev1_val) * 100
        if abs(change_percent) >= threshold:
            anomaly1['detected'] = True
            anomaly1['type'] = 'decrease' if change_percent < 0 else 'increase'
            anomaly1['change'] = round(change_percent, 1)
            anomaly1['reason'] = f"{prev1_col}: {prev1_val:.1f} → {current_col}: {current_val:.1f} ({'+' if change_percent > 0 else ''}{change_percent:.1f}%)"
    elif current_val is None:
        anomaly1['reason'] = f"{current_col}: Veri yok"
    elif prev1_val is None:
        anomaly1['reason'] = f"{prev1_col}: Veri yok"
    
    # ANALİZ 2: Önceki 2 yılın aynı ayı ile karşılaştırma
    anomaly2 = {'detected': False, 'type': None, 'reason': '', 'change': None}
    
    if current_val is not None and prev_year1_val is not None:
        change_percent = ((current_val - prev_year1_val) / prev_year1_val) * 100
        if abs(change_percent) >= threshold:
            anomaly2['detected'] = True
            anomaly2['type'] = 'decrease' if change_percent < 0 else 'increase'
            anomaly2['change'] = round(change_percent, 1)
            anomaly2['reason'] = f"{prev_year1_col}: {prev_year1_val:.1f} → {current_col}: {current_val:.1f} ({'+' if change_percent > 0 else ''}{change_percent:.1f}%)"
    elif current_val is None:
        anomaly2['reason'] = f"{current_col}: Veri yok"
    elif prev_year1_val is None:
        anomaly2['reason'] = f"{prev_year1_col}: Veri yok"
    
    # ANALİZ 3: Trend karşılaştırması
    anomaly3 = {'detected': False, 'type': None, 'reason': ''}
    
    trend_2025 = calculate_trend(prev2_val, prev1_val, current_val)
    trend_2024 = calculate_trend(prev_year1_val, y2024_m1_val, y2024_m2_val)
    trend_2023 = calculate_trend(prev_year2_val, y2023_m1_val, y2023_m2_val)
    
    if trend_2025 is not None and (trend_2024 is not None or trend_2023 is not None):
        trend_anomaly = False
        trend_reasons = []
        
        if trend_2024 is not None:
            trend_diff = abs(trend_2025 - trend_2024)
            if trend_diff >= threshold:
                trend_anomaly = True
                trend_reasons.append(f"2024 trend: {trend_2024:.1f} vs {year} trend: {trend_2025:.1f}")
        else:
            trend_reasons.append("2024: Eksik veri")
        
        if trend_2023 is not None:
            trend_diff = abs(trend_2025 - trend_2023)
            if trend_diff >= threshold:
                trend_anomaly = True
                trend_reasons.append(f"2023 trend: {trend_2023:.1f} vs {year} trend: {trend_2025:.1f}")
        else:
            trend_reasons.append("2023: Eksik veri")
        
        if trend_anomaly:
            anomaly3['detected'] = True
            anomaly3['type'] = 'decrease' if trend_2025 < 0 else 'increase'
            anomaly3['reason'] = ', '.join(trend_reasons)
        elif trend_reasons:
            anomaly3['reason'] = ', '.join(trend_reasons)
    else:
        anomaly3['reason'] = "Trend hesaplanamadı (eksik veri)"
    
    # Genel anomali durumu
    has_anomaly = anomaly1['detected'] or anomaly2['detected'] or anomaly3['detected']
    anomaly_type = None
    if anomaly1['detected']:
        anomaly_type = anomaly1['type']
    elif anomaly2['detected']:
        anomaly_type = anomaly2['type']
    elif anomaly3['detected']:
        anomaly_type = anomaly3['type']
    
    return {
        'tesisat_no': tesisat_no,
        'current_val': current_val,
        'anomaly1': anomaly1,
        'anomaly2': anomaly2,
        'anomaly3': anomaly3,
        'has_anomaly': has_anomaly,
        'anomaly_type': anomaly_type
    }

# Başlık
st.title("📊 Doğalgaz Tüketim Anomali Tespit Sistemi")
st.markdown("---")

# Dosya yükleme
uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Dosyayı oku
    df = pd.read_excel(uploaded_file)
    
    # Tesisat sütununu bul
    tesisat_cols = [col for col in df.columns if 'tesisat' in col.lower()]
    tesisat_col = tesisat_cols[0] if tesisat_cols else df.columns[0]
    
    # Tarih sütunlarını bul
    date_columns = [col for col in df.columns if col != tesisat_col and parse_month_column(col) is not None]
    date_columns = sorted(date_columns, key=lambda x: (parse_month_column(x)['year'], parse_month_column(x)['month']))
    
    st.success(f"✓ {len(df):,} tesisat verisi yüklendi")
    
    # Parametreler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_month = st.selectbox(
            "Analiz Ayı",
            options=date_columns,
            index=len(date_columns) - 1 if date_columns else 0
        )
    
    with col2:
        threshold = st.number_input(
            "Anomali Eşiği (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=5.0
        )
    
    with col3:
        st.write("")
        st.write("")
        analyze_button = st.button("🔍 Analizi Başlat", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner('Analiz ediliyor...'):
            # Her tesisat için analiz yap
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                result = analyze_facility(row, analysis_month, threshold, tesisat_col)
                if result:
                    results.append(result)
                progress_bar.progress((idx + 1) / len(df))
            
            progress_bar.empty()
            
            # Sonuçları session state'e kaydet
            st.session_state['results'] = results
            st.session_state['analysis_month'] = analysis_month
            st.session_state['threshold'] = threshold

# Sonuçları göster
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # İstatistikler
    st.markdown("### 📈 İstatistikler")
    
    total_facilities = len(results)
    anomaly_count = sum(1 for r in results if r['has_anomaly'])
    decrease_count = sum(1 for r in results if r['has_anomaly'] and r['anomaly_type'] == 'decrease')
    increase_count = sum(1 for r in results if r['has_anomaly'] and r['anomaly_type'] == 'increase')
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Toplam Tesisat", f"{total_facilities:,}")
    col2.metric("Anomali Tespit Edilen", f"{anomaly_count:,}", 
                delta=f"%{(anomaly_count/total_facilities*100):.1f}" if total_facilities > 0 else "0%")
    col3.metric("Düşüş Anomalisi", f"{decrease_count:,}", delta="Öncelikli", delta_color="inverse")
    col4.metric("Artış Anomalisi", f"{increase_count:,}")
    
    st.markdown("---")
    
    # Filtreler
    st.markdown("### 🔍 Filtreler")
    filter_col1, filter_col2 = st.columns([2, 1])
    
    with filter_col1:
        filter_type = st.radio(
            "Anomali Tipi",
            options=['Tümü', 'Sadece Düşüşler', 'Sadece Artışlar'],
            horizontal=True
        )
    
    # Filtreleme
    filtered_results = [r for r in results if r['has_anomaly']]
    
    if filter_type == 'Sadece Düşüşler':
        filtered_results = [r for r in filtered_results if r['anomaly_type'] == 'decrease']
    elif filter_type == 'Sadece Artışlar':
        filtered_results = [r for r in filtered_results if r['anomaly_type'] == 'increase']
    
    st.info(f"Gösterilen: {len(filtered_results):,} anomali")
    
    # Excel İndirme
    if filtered_results:
        export_data = []
        for r in filtered_results:
            export_data.append({
                'Tesisat No': r['tesisat_no'],
                'Mevcut Tüketim (m³)': f"{r['current_val']:.1f}" if r['current_val'] is not None else 'Yok',
                'Anomali Tipi': 'Düşüş' if r['anomaly_type'] == 'decrease' else 'Artış',
                'Analiz 1 (Önceki 2 Ay)': '✓' if r['anomaly1']['detected'] else '-',
                'Analiz 1 Detay': r['anomaly1']['reason'] or '-',
                'Analiz 2 (Önceki 2 Yıl)': '✓' if r['anomaly2']['detected'] else '-',
                'Analiz 2 Detay': r['anomaly2']['reason'] or '-',
                'Analiz 3 (Trend)': '✓' if r['anomaly3']['detected'] else '-',
                'Analiz 3 Detay': r['anomaly3']['reason'] or '-'
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Excel'e dönüştür
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Anomaliler')
        
        st.download_button(
            label="📥 Excel İndir",
            data=output.getvalue(),
            file_name=f"anomali_raporu_{st.session_state['analysis_month']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.markdown("---")
    
    # Anomali Listesi
    st.markdown("### 🚨 Tespit Edilen Anomaliler")
    
    for idx, result in enumerate(filtered_results, 1):
        with st.expander(
            f"**{idx}. Tesisat: {result['tesisat_no']}** - "
            f"{'🔻 Düşüş' if result['anomaly_type'] == 'decrease' else '🔺 Artış'} - "
            f"Tüketim: {result['current_val']:.1f} m³" if result['current_val'] is not None else "Veri yok"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📅 Analiz 1: Önceki 2 Ay**")
                if result['anomaly1']['detected']:
                    st.error(f"✓ Anomali Tespit Edildi")
                    st.write(result['anomaly1']['reason'])
                else:
                    st.success("Anomali yok")
                    if result['anomaly1']['reason']:
                        st.caption(result['anomaly1']['reason'])
            
            with col2:
                st.markdown("**📆 Analiz 2: Önceki 2 Yıl**")
                if result['anomaly2']['detected']:
                    st.error(f"✓ Anomali Tespit Edildi")
                    st.write(result['anomaly2']['reason'])
                else:
                    st.success("Anomali yok")
                    if result['anomaly2']['reason']:
                        st.caption(result['anomaly2']['reason'])
            
            with col3:
                st.markdown("**📊 Analiz 3: Trend Karşılaştırma**")
                if result['anomaly3']['detected']:
                    st.error(f"✓ Anomali Tespit Edildi")
                    st.write(result['anomaly3']['reason'])
                else:
                    st.success("Anomali yok")
                    if result['anomaly3']['reason']:
                        st.caption(result['anomaly3']['reason'])

else:
    st.info("👆 Lütfen Excel dosyanızı yükleyin ve analizi başlatın.")
    
    st.markdown("---")
    st.markdown("### 📋 Veri Formatı")
    st.markdown("""
    Excel dosyanız şu formatta olmalıdır:
    
    | Tesisat no | Oca.23 | Şub.23 | Mar.23 | Nis.23 | ... |
    |------------|--------|--------|--------|--------|-----|
    | 123        | 20     | 50     | 60     | 45     | ... |
    | 456        | 30     | 40     | 35     | 38     | ... |
    
    - İlk sütun: Tesisat numarası
    - Diğer sütunlar: Ay.Yıl formatında (örn: Oca.23, Şub.23)
    - Değerler: m³ cinsinden tüketim
    """)
