import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go

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

def parse_date(date_str):
    """Tarih string'ini parse et"""
    try:
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
        
        month_replacements = {
            'Sub': 'Şub',
            'Şub': 'Şub', 
            'Agu': 'Ağu',
            'Ağu': 'Ağu'
        }
        month_name = month_replacements.get(month_name, month_name)
        
        if month_name not in MONTH_MAP:
            return None, None
        
        month = MONTH_MAP[month_name]
        year = 2000 + int(year_short)
        
        return year, month
    except:
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

def calculate_trend(v1, v2, v3):
    """3 değer arasındaki ortalama trendi hesapla"""
    if v1 is None or v2 is None or v3 is None:
        return None
    diff1 = v2 - v1
    diff2 = v3 - v2
    return (diff1 + diff2) / 2

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
    """Tek bir tesisat için anomali analizi yap"""
    
    current_val = get_consumption(df, tesisat_no, analysis_year, analysis_month)
    
    prev1_month = 12 if analysis_month == 1 else analysis_month - 1
    prev1_year = analysis_year - 1 if analysis_month == 1 else analysis_year
    prev2_month = 11 if analysis_month <= 2 else (12 if analysis_month == 2 else analysis_month - 2)
    prev2_year = analysis_year - 1 if analysis_month <= 2 else analysis_year
    
    prev1_val = get_consumption(df, tesisat_no, prev1_year, prev1_month)
    prev2_val = get_consumption(df, tesisat_no, prev2_year, prev2_month)
    
    prev_year1_val = get_consumption(df, tesisat_no, analysis_year - 1, analysis_month)
    prev_year2_val = get_consumption(df, tesisat_no, analysis_year - 2, analysis_month)
    
    next1_month = 1 if analysis_month == 12 else analysis_month + 1
    next1_year = analysis_year + 1 if analysis_month == 12 else analysis_year
    next2_month = 2 if analysis_month >= 11 else (1 if analysis_month == 11 else analysis_month + 2)
    next2_year = analysis_year + 1 if analysis_month >= 11 else analysis_year
    
    next1_val = get_consumption(df, tesisat_no, next1_year, next1_month)
    next2_val = get_consumption(df, tesisat_no, next2_year, next2_month)
    
    y2024_m1_val = get_consumption(df, tesisat_no, analysis_year - 1, next1_month)
    y2024_m2_val = get_consumption(df, tesisat_no, analysis_year - 1, next2_month)
    y2023_m1_val = get_consumption(df, tesisat_no, analysis_year - 2, next1_month)
    y2023_m2_val = get_consumption(df, tesisat_no, analysis_year - 2, next2_month)
    
    recent_data = df[(df['tesisat_no'] == tesisat_no) & 
                     (df['tuketim'] > 0) & 
                     (df['tuketim'].notna())]
    if not recent_data.empty:
        avg_consumption = recent_data['tuketim'].tail(6).mean()
    else:
        avg_consumption = 0
    
    segment, segment_threshold = assign_segment(avg_consumption)
    
    # ANALİZ 1
    anomaly1 = {'detected': False, 'type': None, 'reason': '', 'change': None}
    
    if current_val is not None and prev1_val is not None:
        change_percent = ((current_val - prev1_val) / prev1_val) * 100
        if abs(change_percent) >= segment_threshold:
            anomaly1['detected'] = True
            anomaly1['type'] = 'decrease' if change_percent < 0 else 'increase'
            anomaly1['change'] = round(change_percent, 1)
            anomaly1['reason'] = f"{REVERSE_MONTH_MAP[prev1_month]}.{str(prev1_year)[2:]}: {prev1_val:.1f} → {REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year)[2:]}: {current_val:.1f} ({'+' if change_percent > 0 else ''}{change_percent:.1f}%)"
    elif current_val is None:
        anomaly1['reason'] = f"{REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year)[2:]}: Veri yok"
    elif prev1_val is None:
        anomaly1['reason'] = f"{REVERSE_MONTH_MAP[prev1_month]}.{str(prev1_year)[2:]}: Veri yok"
    
    # ANALİZ 2
    anomaly2 = {'detected': False, 'type': None, 'reason': '', 'change': None}
    
    if current_val is not None and prev_year1_val is not None:
        change_percent = ((current_val - prev_year1_val) / prev_year1_val) * 100
        if abs(change_percent) >= segment_threshold:
            anomaly2['detected'] = True
            anomaly2['type'] = 'decrease' if change_percent < 0 else 'increase'
            anomaly2['change'] = round(change_percent, 1)
            anomaly2['reason'] = f"{REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year-1)[2:]}: {prev_year1_val:.1f} → {REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year)[2:]}: {current_val:.1f} ({'+' if change_percent > 0 else ''}{change_percent:.1f}%)"
    elif current_val is None:
        anomaly2['reason'] = f"{REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year)[2:]}: Veri yok"
    elif prev_year1_val is None:
        anomaly2['reason'] = f"{REVERSE_MONTH_MAP[analysis_month]}.{str(analysis_year-1)[2:]}: Veri yok"
    
    # ANALİZ 3
    anomaly3 = {'detected': False, 'type': None, 'reason': ''}
    
    trend_current = calculate_trend(prev2_val, prev1_val, current_val)
    trend_2024 = calculate_trend(prev_year1_val, y2024_m1_val, y2024_m2_val)
    trend_2023 = calculate_trend(prev_year2_val, y2023_m1_val, y2023_m2_val)
    
    if trend_current is not None and (trend_2024 is not None or trend_2023 is not None):
        trend_anomaly = False
        trend_reasons = []
        
        if trend_2024 is not None:
            trend_diff = abs(trend_current - trend_2024)
            if trend_diff >= segment_threshold:
                trend_anomaly = True
                trend_reasons.append(f"2024 trend: {trend_2024:.1f} vs {analysis_year} trend: {trend_current:.1f}")
        else:
            trend_reasons.append("2024: Eksik veri")
        
        if trend_2023 is not None:
            trend_diff = abs(trend_current - trend_2023)
            if trend_diff >= segment_threshold:
                trend_anomaly = True
                trend_reasons.append(f"2023 trend: {trend_2023:.1f} vs {analysis_year} trend: {trend_current:.1f}")
        else:
            trend_reasons.append("2023: Eksik veri")
        
        if trend_anomaly:
            anomaly3['detected'] = True
            anomaly3['type'] = 'decrease' if trend_current < 0 else 'increase'
            anomaly3['reason'] = ', '.join(trend_reasons)
        elif trend_reasons:
            anomaly3['reason'] = ', '.join(trend_reasons)
    else:
        anomaly3['reason'] = "Trend hesaplanamadı (eksik veri)"
    
    has_anomaly = anomaly1['detected'] or anomaly2['detected'] or anomaly3['detected']
    anomaly_type = None
    if anomaly1['detected']:
        anomaly_type = anomaly1['type']
    elif anomaly2['detected']:
        anomaly_type = anomaly2['type']
    elif anomaly3['detected']:
        anomaly_type = anomaly3['type']
    
    priority_score = 0
    if has_anomaly and current_val is not None:
        max_change = max(
            abs(anomaly1['change']) if anomaly1['change'] else 0,
            abs(anomaly2['change']) if anomaly2['change'] else 0
        )
        priority_score = (avg_consumption * max_change) / 100
    
    return {
        'tesisat_no': tesisat_no,
        'current_val': current_val,
        'avg_consumption': avg_consumption,
        'segment': segment,
        'anomaly1': anomaly1,
        'anomaly2': anomaly2,
        'anomaly3': anomaly3,
        'has_anomaly': has_anomaly,
        'anomaly_type': anomaly_type,
        'priority_score': priority_score
    }

# Başlık
st.title("📊 Doğalgaz Tüketim Anomali Tespit Sistemi")
st.markdown("**Uzun Format (Long Format) - Her satır bir ay verisi**")
st.markdown("---")

# Dosya yükleme
uploaded_file = st.file_uploader("Excel dosyasını yükleyin", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0)
        
        st.write("📌 **Okundu! İlk 5 satır:**")
        st.dataframe(df_raw.head())
        st.write(f"**Sütun adları:** {list(df_raw.columns)}")
        
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        
        st.write(f"**Normalize edilmiş sütunlar:** {list(df_raw.columns)}")
        
        tesisat_col = None
        tarih_col = None
        tuketim_col = None
        
        for col in df_raw.columns:
            col_lower = col.lower().strip()
            if 'tesisat' in col_lower or 'facility' in col_lower:
                tesisat_col = col
            if 'tarih' in col_lower or 'ay' in col_lower or 'donem' in col_lower or 'periode' in col_lower:
                tarih_col = col
            if 'tuketim' in col_lower or 'm3' in col_lower or 'miktar' in col_lower or 'consumption' in col_lower:
                tuketim_col = col
        
        st.write(f"**Tespit edilen sütunlar:** tesisat={tesisat_col}, tarih={tarih_col}, tuketim={tuketim_col}")
        
        if not all([tesisat_col, tarih_col, tuketim_col]):
            st.error("❌ Sütunlar tespit edilemedi!")
            st.warning("**Sütun adlarının şunları içermesi gerekir:** 'tesisat', 'tarih' (veya 'ay'), 'tuketim' (veya 'm3')")
            st.stop()
        
        df = df_raw[[tesisat_col, tarih_col, tuketim_col]].copy()
        df.columns = ['tesisat_no', 'tarih', 'tuketim']
        
        df['yil'], df['ay'] = zip(*df['tarih'].apply(parse_date))
        
        valid_rows = df[(df['yil'].notna()) & (df['ay'].notna())]
        
        st.write(f"**Geçerli satır sayısı:** {len(valid_rows)} / {len(df)}")
        
        if len(valid_rows) == 0:
            st.error("❌ Tarih formatı parselenemedi! Örnek format: 'Ocak 23', 'Oca 23', 'Şubat 23'")
            st.write("**Örnek tarihler:**", df['tarih'].unique()[:10])
            st.stop()
        
        df = valid_rows
        df['yil'] = df['yil'].astype(int)
        df['ay'] = df['ay'].astype(int)
        df['tuketim'] = pd.to_numeric(df['tuketim'], errors='coerce')
        
        unique_tesisats = df['tesisat_no'].unique()
        
        st.success(f"✓ {len(unique_tesisats):,} tesisat, {len(df):,} satır veri yüklendi")
        
        with st.expander("📋 Veri Önizleme"):
            st.dataframe(df.head(20))
        
        st.markdown("### ⚙️ Analiz Parametreleri")
        col1, col2, col3 = st.columns(3)
        
        available_years = sorted(df['yil'].unique(), reverse=True)
        
        with col1:
            analysis_year = st.selectbox(
                "Analiz Yılı",
                options=available_years,
                index=0
            )
        
        with col2:
            analysis_month = st.selectbox(
                "Analiz Ayı",
                options=list(REVERSE_MONTH_MAP.keys()),
                format_func=lambda x: REVERSE_MONTH_MAP[x],
                index=9
            )
        
        with col3:
            base_threshold = st.number_input(
                "Baz Anomali Eşiği (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=5.0
            )
        
        st.info(f"📅 Seçilen dönem: **{REVERSE_MONTH_MAP[analysis_month]} {analysis_year}**")
        
        if st.button("🔍 Analizi Başlat", type="primary", use_container_width=True):
            with st.spinner('Analiz ediliyor...'):
                results = []
                progress_bar = st.progress(0)
                
                for idx, tesisat_no in enumerate(unique_tesisats):
                    result = analyze_facility(df, tesisat_no, analysis_year, analysis_month, base_threshold)
                    if result:
                        results.append(result)
                    progress_bar.progress((idx + 1) / len(unique_tesisats))
                
                progress_bar.empty()
                
                st.session_state['results'] = results
                st.session_state['df'] = df
                st.session_state['analysis_year'] = analysis_year
                st.session_state['analysis_month'] = analysis_month

    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")
        st.write("Lütfen Excel dosyasını kontrol edin ve tekrar yükleyin.")

# Sonuçları göster
if 'results' in st.session_state:
    results = st.session_state['results']
    df = st.session_state['df']
    analysis_year = st.session_state['analysis_year']
    analysis_month = st.session_state['analysis_month']
    
    st.markdown("---")
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
    
    st.markdown("#### 📊 Segment Dağılımı")
    segment_stats = {}
    for r in results:
        seg = r['segment']
        if seg not in segment_stats:
            segment_stats[seg] = {'total': 0, 'anomaly': 0}
        segment_stats[seg]['total'] += 1
        if r['has_anomaly']:
            segment_stats[seg]['anomaly'] += 1
    
    seg_cols = st.columns(4)
    for idx, (seg, stats) in enumerate(sorted(segment_stats.items())):
        with seg_cols[idx]:
            st.metric(
                f"Segment {seg}",
                f"{stats['total']:,} tesisat",
                delta=f"{stats['anomaly']} anomali"
            )
    
    st.markdown("---")
    
    st.markdown("### 🔍 Filtreler")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        filter_type = st.radio(
            "Anomali Tipi",
            options=['Tümü', 'Sadece Düşüşler', 'Sadece Artışlar'],
            horizontal=True
        )
    
    with filter_col2:
        filter_segment = st.multiselect(
            "Segment Filtresi",
            options=['A', 'B', 'C', 'D'],
            default=['A', 'B', 'C', 'D']
        )
    
    with filter_col3:
        min_priority = st.number_input(
            "Min. Öncelik Skoru",
            min_value=0.0,
            value=0.0,
            step=10.0
        )
    
    filtered_results = [r for r in results if r['has_anomaly']]
    
    if filter_type == 'Sadece Düşüşler':
        filtered_results = [r for r in filtered_results if r['anomaly_type'] == 'decrease']
    elif filter_type == 'Sadece Artışlar':
        filtered_results = [r for r in filtered_results if r['anomaly_type'] == 'increase']
    
    filtered_results = [r for r in filtered_results if r['segment'] in filter_segment]
    filtered_results = [r for r in filtered_results if r['priority_score'] >= min_priority]
    
    filtered_results = sorted(filtered_results, key=lambda x: x['priority_score'], reverse=True)
    
    st.info(f"📊 Gösterilen: **{len(filtered_results):,}** anomali")
    
    if filtered_results:
        export_data = []
        for r in filtered_results:
            export_data.append({
                'Tesisat No': r['tesisat_no'],
                'Segment': r['segment'],
                'Ort. Tüketim (m³)': f"{r['avg_consumption']:.1f}" if r['avg_consumption'] else 'N/A',
                'Mevcut Tüketim (m³)': f"{r['current_val']:.1f}" if r['current_val'] is not None else 'Yok',
                'Anomali Tipi': 'Düşüş' if r['anomaly_type'] == 'decrease' else 'Artış',
                'Öncelik Skoru': f"{r['priority_score']:.0f}",
                'Analiz 1': '✓' if r['anomaly1']['detected'] else '-',
                'Analiz 1 Detay': r['anomaly1']['reason'] or '-',
                'Analiz 2': '✓' if r['anomaly2']['detected'] else '-',
                'Analiz 2 Detay': r['anomaly2']['reason'] or '-',
                'Analiz 3': '✓' if r['anomaly3']['detected'] else '-',
                'Analiz 3 Detay': r['anomaly3']['reason'] or '-'
            })
        
        export_df = pd.DataFrame(export_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Anomaliler')
        
        st.download_button(
            label="📥 Excel İndir",
            data=output.getvalue(),
            file_name=f"anomali_raporu_{REVERSE_MONTH_MAP[analysis_month]}_{analysis_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.markdown("---")
    
    st.markdown("### 🚨 Tespit Edilen Anomaliler (Öncelik Sırasına Göre)")
    
    items_per_page = 20
    total_pages = (len(filtered_results) - 1) // items_per_page + 1 if filtered_results else 0
    
    if total_pages > 0:
        page = st.selectbox("Sayfa", range(1, total_pages + 1), key='page_select')
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_results))
        
        page_results = filtered_results[start_idx:end_idx]
        
        for idx, result in enumerate(page_results, start=start_idx + 1):
            priority_color = "🔴" if result['priority_score'] >= 1000 else "🟡" if result['priority_score'] >= 100 else "🟢"
            
            expander_title = (
                f"{priority_color} **#{idx} - Tesisat: {result['tesisat_no']}** | "
                f"Segment: {result['segment']} | "
                f"Öncelik: {result['priority_score']:.0f} | "
                f"{'🔻 Düşüş' if result['anomaly_type'] == 'decrease' else '🔺 Artış'} | "
                f"Tüketim: {result['current_val']:.1f} m³" if result['current_val'] is not None else "Veri yok"
            )
            
            with st.expander(expander_title):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**📊 Genel Bilgiler**")
                    st.write(f"Ortalama Tüketim: **{result['avg_consumption']:.1f} m³**")
                    if result['current_val']:
                        st.write(f"Mevcut Tüketim: **{result['current_val']:.1f} m³**")
                    else:
                        st.write("Mevcut Tüketim: **Veri yok**")
                    st.write(f"Segment: **{result['segment']}**")
                    st.write(f"Öncelik Skoru: **{result['priority_score']:.0f}**")
                
                with col2:
                    tesisat_data = df[df['tesisat_no'] == result['tesisat_no']].copy()
                    tesisat_data = tesisat_data.sort_values(['yil', 'ay'])
                    tesisat_data['tarih_str'] = tesisat_data.apply(
                        lambda x: f"{REVERSE_MONTH_MAP[int(x['ay'])]}.{str(int(x['yil']))[2:]}", axis=1
                    )
                    
                    fig = px.line(tesisat_data, x='tarih_str', y='tuketim',
                                title=f'Tüketim Trendi - {result["tesisat_no"]}',
                                markers=True)
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📅 Analiz 1: Önceki 2 Ay**")
                    if result['anomaly1']['detected']:
                        st.error("✓ Anomali Tespit Edildi")
                        st.write(result['anomaly1']['reason'])
                    else:
                        st.success("Anomali yok")
                        if result['anomaly1']['reason']:
                            st.caption(result['anomaly1']['reason'])
                
                with col2:
                    st.markdown("**📆 Analiz 2: Önceki 2 Yıl**")
                    if result['anomaly2']['detected']:
                        st.error("✓ Anomali Tespit Edildi")
                        st.write(result['anomaly2']['reason'])
                    else:
                        st.success("Anomali yok")
                        if result['anomaly2']['reason']:
                            st.caption(result['anomaly2']['reason'])
                
                with col3:
                    st.markdown("**📊 Analiz 3: Trend**")
                    if result['anomaly3']['detected']:
                        st.error("✓ Anomali Tespit Edildi")
                        st.write(result['anomaly3']['reason'])
                    else:
                        st.success("Anomali yok")
                        if result['anomaly3']['reason']:
                            st.caption(result['anomaly3']['reason'])
    else:
        st.info("Seçili filtrelere göre anomali bulunamadı.")

else:
    st.info("👆 Lütfen Excel dosyanızı yükleyin")
    
    st.markdown("---")
    st.markdown("### 📋 Beklenen Veri Formatı (Uzun Format)")
    
    example_data = pd.DataFrame({
        'Tesisat no': [123, 123, 123, 456, 456, 456],
        'tarih': ['Ocak 23', 'Şubat 23', 'Mart 23', 'Ocak 23', 'Şubat 23', 'Mart 23'],
        'tüketim m3': [20, 50, 60, 30, 40, 35]
    })
    
    st.dataframe(example_data)
    
    st.markdown("""
    **Önemli Noktalar:**
    - Her satır bir tesisat-ay kombinasyonu
    - Tesisat numarası her ay için tekrar edilmeli
    - Tarih formatı: **
