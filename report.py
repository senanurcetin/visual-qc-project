"""Excel production report (pandas + XlsxWriter), built in memory."""
import io

import pandas as pd


def build_report(df: pd.DataFrame) -> io.BytesIO:
    """Return an .xlsx workbook (Dashboard + log sheets) for the given unit history."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        workbook = writer.book

        # --- 1. SEKMESİ: DASHBOARD (Özet ve Grafikler) ---
        dash_sheet = workbook.add_worksheet('Dashboard')

        # Formatlar (Kurumsal Tasarım)
        title_fmt = workbook.add_format({'bold': True, 'font_size': 20, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#1E293B', 'font_color': 'white'})
        kpi_header_fmt = workbook.add_format({'bold': True, 'font_size': 11, 'align': 'center', 'bg_color': '#E2E8F0', 'border': 1})
        kpi_val_fmt = workbook.add_format({'font_size': 16, 'align': 'center', 'border': 1, 'bold': True})

        dash_sheet.merge_range('B2:F3', 'ÜRETİM PERFORMANS RAPORU', title_fmt)

        # KPI Hesaplamaları
        total_units = len(df)
        ok_units = len(df[df['status'] == 'OK'])
        yield_rate = (ok_units / total_units * 100) if total_units > 0 else 0
        avg_oee = df['oee_score'].mean() * 100

        # KPI Yazdırma
        dash_sheet.write('B5', 'Toplam Üretim', kpi_header_fmt)
        dash_sheet.write('B6', total_units, kpi_val_fmt)

        dash_sheet.write('C5', 'Sağlam (OK)', kpi_header_fmt)
        dash_sheet.write('C6', ok_units, kpi_val_fmt)

        dash_sheet.write('D5', 'Başarı Oranı (%)', kpi_header_fmt)
        dash_sheet.write('D6', f"{yield_rate:.1f}%", kpi_val_fmt)

        dash_sheet.write('E5', 'Ort. OEE (%)', kpi_header_fmt)
        dash_sheet.write('E6', f"{avg_oee:.1f}%", kpi_val_fmt)

        # Pasta Grafiği (OK vs NOK)
        status_counts = df['status'].value_counts()
        dash_sheet.write_column('AA1', status_counts.index) # Gizli Veri Alanı
        dash_sheet.write_column('AB1', status_counts.values)

        pie_chart = workbook.add_chart({'type': 'pie'})
        pie_chart.add_series({
            'name': 'Kalite Dağılımı',
            'categories': ['Dashboard', 0, 26, len(status_counts)-1, 26], # AA1:AAn
            'values':     ['Dashboard', 0, 27, len(status_counts)-1, 27], # AB1:ABn
            'points': [{'fill': {'color': '#10B981'}}, {'fill': {'color': '#EF4444'}}],
        })
        pie_chart.set_title({'name': 'OK vs FAIL Oranı'})
        dash_sheet.insert_chart('B9', pie_chart)

        # --- 2. SEKMESİ: DETAYLI LOGLAR ---
        df.to_excel(writer, sheet_name='Detaylı Loglar', index=False)
        log_sheet = writer.sheets['Detaylı Loglar']

        # Tablo Tasarımı
        header_fmt = workbook.add_format({'bold': True, 'fg_color': '#1E293B', 'font_color': 'white', 'border': 1})
        ok_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'}) # Açık Yeşil
        fail_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'}) # Açık Kırmızı

        # Başlıkları boya
        for col_num, value in enumerate(df.columns.values):
            log_sheet.write(0, col_num, value, header_fmt)
            log_sheet.set_column(col_num, col_num, 20) # Sütun genişliği

        # Koşullu Biçimlendirme (Yeşil/Kırmızı)
        log_sheet.conditional_format(f'C2:C{len(df)+1}', {'type': 'cell', 'criteria': '==', 'value': '"OK"', 'format': ok_fmt})
        log_sheet.conditional_format(f'C2:C{len(df)+1}', {'type': 'cell', 'criteria': '!=', 'value': '"OK"', 'format': fail_fmt})

    output.seek(0)
    return output
