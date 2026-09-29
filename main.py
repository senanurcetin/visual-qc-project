import io
import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd  # Excel raporlama ve veri manipülasyonu için
from flask import Flask, Response, jsonify, render_template_string, request, send_file, session

import line_sim
from case_study import case_study_bp
from rag_demo import rag_bp

app = Flask(__name__, static_folder="web/line", static_url_path="/line/static")
# Session çerezi yalnızca demo hattının kontrol durumunu imzalar; gizli veri taşımaz.
app.secret_key = os.environ.get("SECRET_KEY", "visual-qc-public-demo")
app.permanent_session_lifetime = timedelta(days=7)
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                  # Polls must never re-send the cookie: a slow GET would overwrite a newer command.
                  SESSION_REFRESH_EACH_REQUEST=False)
app.register_blueprint(case_study_bp)
app.register_blueprint(rag_bp)

# --- 1. FRONTEND: HMI ARAYÜZÜ (HTML/JS) ---
# Gerçek bir endüstriyel panel (HMI) simülasyonu için karanlık tema ve yüksek kontrastlı tasarım.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionQC - Pro HMI v34.2</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Roboto+Mono:wght@500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/line/static/line3d.css">
    <script type="importmap">
        {"imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
        }}
    </script>
    <script type="module" src="/line/static/line3d.js"></script>
    <style>
        :root {
            --bg: #111827; --card: #1f293b; --border: #374151; --text: #f3f4f6;
            --green: #10b981; --yellow: #f59e0b; --red: #ef4444; --blue: #3b82f6; --purple: #8b5cf6; --teal: #14b8a6;
        }
        html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; background: var(--bg); color: var(--text); }
        body { font-family: 'Inter', sans-serif; display: flex; flex-direction: column; }

        .header { flex-shrink: 0; display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; padding: 12px; background: #030712; border-bottom: 1px solid var(--border); }
        .kpi-card { background: var(--card); border: 1px solid var(--border); padding: 8px 12px; border-radius: 4px; }
        .kpi-title { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; margin-bottom: 4px; font-weight: 600; }
        .kpi-val { font-family: 'Roboto Mono', monospace; font-size: 1.4rem; font-weight: 500; }

        .control-bar { flex-shrink: 0; padding: 10px; display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; background: #111827; border-bottom: 1px solid var(--border); }
        .btn { padding: 10px; border: none; border-radius: 4px; color: white; font-weight: 600; cursor: pointer; text-transform: uppercase; font-size: 0.85rem; transition: 0.2s; text-decoration: none; display: flex; align-items: center; justify-content: center; }
        .btn-start { background: #065f46; border-bottom: 3px solid #064e3b; } .btn-start:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-pause { background: #92400e; border-bottom: 3px solid #78350f; } .btn-pause:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-reset { background: #1e40af; border-bottom: 3px solid #1e3a8a; } .btn-reset:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-estop { background: #991b1b; border-bottom: 3px solid #7f1d1d; } .btn-estop:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-sim-fail { background: #6b21a8; border-bottom: 3px solid #581c87; } .btn-sim-fail:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-case-study { background: #0369a1; border-bottom: 3px solid #075985; } .btn-case-study:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-rag { background: #4338ca; border-bottom: 3px solid #3730a3; } .btn-rag:active { transform: translateY(2px); border-bottom: 0px; }
        .btn-export { background: var(--teal); border-bottom: 3px solid #0f766e; } .btn-export:active { transform: translateY(2px); border-bottom: 0px; }

        .main-grid { flex-grow: 1; display: grid; grid-template-columns: 1.8fr 1.2fr; gap: 12px; padding: 12px; min-height: 0; }
        .panel { background: var(--card); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; display: flex; flex-direction: column; }
        .video-box { background: #000; position: relative; }
        .video-box img { width: 100%; height: 100%; object-fit: contain; }

        .right-col { display: flex; flex-direction: column; gap: 12px; min-height: 0; background: #0f172a; padding: 8px; border-radius: 4px; border: 1px solid var(--border); }
        .section-header { padding: 6px 10px; background: #334155; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; border-bottom: 1px solid var(--border); }
        .profit-sec { border-top: 3px solid var(--blue); }
        .oee-sec { border-top: 3px solid var(--green); }
        .log-sec { border-top: 3px solid var(--yellow); flex-grow: 1; display: flex; flex-direction: column; min-height: 0; }

        .chart-container { height: 140px; padding: 8px; position: relative; }
        .log-box { flex-grow: 1; overflow-y: auto; padding: 0; }
        .log-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 0.75rem; }
        .log-table th { position: sticky; top: 0; background: #1e293b; padding: 8px; text-align: left; color: #9ca3af; }
        .log-table td { padding: 6px 8px; border-bottom: 1px solid #334155; }
        .text-ok { color: var(--green); } .text-fail { color: var(--red); font-weight: bold; }
        .warn-card { border: 1px solid var(--yellow) !important; color: var(--yellow) !important; }
    </style>
</head>
<body>
    <div class="header">
        <div class="kpi-card" id="oee_card"><div class="kpi-title">OEE Performance</div><div class="kpi-val" id="val_oee">0.0%</div></div>
        <div class="kpi-card"><div class="kpi-title">Status</div><div id="status_val" class="kpi-val" style="color:var(--red)">INIT</div></div>
        <div class="kpi-card"><div class="kpi-title">Net Profit</div><div class="kpi-val" id="val_profit">$0.00</div></div>
        <div class="kpi-card"><div class="kpi-title">Total Output</div><div class="kpi-val" id="val_total">0</div></div>
        <div class="kpi-card"><div class="kpi-title">Yield OK</div><div class="kpi-val" style="color:var(--green)" id="val_ok">0</div></div>
        <div class="kpi-card"><div class="kpi-title">Yield NOK</div><div class="kpi-val" style="color:var(--red)" id="val_nok">0</div></div>
    </div>
    <div class="control-bar">
        <button class="btn btn-start" onclick="sendCmd('START')">Start Cycle</button>
        <button class="btn btn-pause" onclick="sendCmd('PAUSE')">Pause System</button>
        <button class="btn btn-reset" onclick="sendCmd('RESET')">Master Reset</button>
        <button class="btn btn-estop" onclick="sendCmd('ESTOP')">Emergency Stop</button>
        <button class="btn btn-sim-fail" onclick="sendCmd('SIMULATE_FAIL')">Simulate Defect</button>
        <a href="/case-study" class="btn btn-case-study">Case Study</a>
        <a href="/rag" class="btn btn-rag">RAG Map</a>
        <a href="/api/export_report" class="btn btn-export">&#x1F4E5; Export Report</a>
    </div>
    <div class="main-grid">
        <div class="panel video-box" id="line-twin">
            <div class="twin-loading" id="twin-loading">Loading 3D line…</div>
            <img id="twin-fallback" alt="Simulated 2D inspection camera feed" hidden>
        </div>
        <div class="right-col">
            <div class="panel profit-sec"><div class="section-header">Trend Analysis</div><div class="chart-container"><canvas id="profitChart"></canvas></div></div>
            <div class="panel oee-sec"><div class="section-header">OEE Breakdown</div><div class="chart-container"><canvas id="oeeChart"></canvas></div></div>
            <div class="panel log-sec"><div class="section-header">Event Historian</div><div class="log-box"><table class="log-table"><thead><tr><th>Time</th><th>ID</th><th>Result</th></tr></thead><tbody id="log-tbody"></tbody></table></div></div>
        </div>
    </div>
<script>
    let profitChart = null;
    let oeeChart = null;

    // --- Chart.js Başlatma ---
    function initCharts() {
        try {
            if (typeof Chart === 'undefined') { console.error("Chart.js missing"); return; }
            const chartConfig = (type, color) => ({
                type: type,
                data: { labels: [], datasets: [{ data: [], borderColor: color, backgroundColor: color + '22', fill: true, tension: 0.3 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#334155' }, ticks: { color: '#94a3af', font: {size: 10} } } } }
            });
            profitChart = new Chart('profitChart', chartConfig('line', '#3b82f6'));
            oeeChart = new Chart('oeeChart', {
                type: 'bar',
                data: { labels: ['AVA', 'PER', 'QLY'], datasets: [{ data: [0,0,0], backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'] }] },
                options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: {display: false} }, scales: { x: { max: 1, ticks: { color: '#94a3af', callback: v => (v*100)+'%' } }, y: { ticks: { color: '#f3f4f6' } } } }
            });
        } catch (e) { console.warn("Chart Init Error", e); }
    }

    // --- AJAX Veri Çekme (1000ms Loop) ---
    function update() {
        fetch('/api/data')
            .then(r => {
                if (!r.ok) return r.text().then(text => { throw new Error(text) });
                return r.json();
            })
            .then(data => {
                // Durum Göstergesi
                const statusEl = document.getElementById('status_val');
                if(statusEl) {
                    statusEl.textContent = data.system_mode;
                    statusEl.style.color = data.system_mode === 'RUNNING' ? 'var(--green)' :
                                         (data.system_mode === 'ESTOP' ? 'var(--red)' : 'var(--yellow)');
                }

                // KPI Güncellemeleri
                document.getElementById('val_profit').textContent = `$${data.net_profit.toFixed(2)}`;
                document.getElementById('val_total').textContent = data.total_units;
                document.getElementById('val_ok').textContent = data.ok_units;
                document.getElementById('val_nok').textContent = data.nok_units;

                const oeeVal = (data.oee * 100).toFixed(1);
                document.getElementById('val_oee').textContent = oeeVal + '%';
                const oeeCard = document.getElementById('oee_card');
                if (oeeCard) oeeCard.className = oeeVal < 65 ? 'kpi-card warn-card' : 'kpi-card';

                // Log Tablosu Güncelleme
                const tbody = document.getElementById('log-tbody');
                if(tbody && data.recent_logs) {
                    tbody.innerHTML = data.recent_logs.map(log => `<tr><td>${log.time}</td><td>${log.id}</td><td class="${log.status=='OK'?'text-ok':'text-fail'}">${log.status}${log.defect ? ' · ' + log.defect.replace(/_/g, ' ') : ''}</td></tr>`).join('');
                }

                // Grafik Güncellemeleri
                if (profitChart && data.system_mode === 'RUNNING') {
                    profitChart.data.labels.push('');
                    profitChart.data.datasets[0].data.push(data.net_profit);
                    if (profitChart.data.labels.length > 30) { profitChart.data.labels.shift(); profitChart.data.datasets[0].data.shift(); }
                    profitChart.update('none');
                }
                if (oeeChart) {
                    oeeChart.data.datasets[0].data = [data.availability, data.performance, data.quality];
                    oeeChart.update();
                }
            })
            .catch(err => {
                console.error("Data Fetch Error:", err);
                const statusEl = document.getElementById('status_val');
                if(statusEl && statusEl.textContent !== "SERVER ERR") {
                    statusEl.textContent = "SERVER ERR";
                    statusEl.style.color = "var(--red)";
                }
            });
    }

    function sendCmd(cmd) {
        fetch('/api/control', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({command: cmd})
        }).then(() => setTimeout(update, 50));
    }

    // 3D hat ikizi yüklenemezse (WebGL yok / CDN erişilemiyor) eski 2D kamera akışına düş.
    window.showLineFallback = function () {
        const img = document.getElementById('twin-fallback');
        if (!img.src) img.src = '/video_feed';
        img.hidden = false;
        const loading = document.getElementById('twin-loading');
        if (loading) loading.hidden = true;
    };
    setTimeout(() => { if (!window.lineTwinReady) window.showLineFallback(); }, 8000);

    setInterval(update, 1000);
    window.onload = initCharts;
</script>
</body>
</html>
"""

# --- 2. BACKEND: İŞ MANTIĞI & DURUM YÖNETİMİ ---
# --- HAT SİMÜLASYONU (DURUMSUZ) ---
# Hattın tamamı line_sim içinde, ziyaretçinin imzalı session çerezindeki küçük bir kontrol
# durumundan deterministik olarak hesaplanır. Böylece her ziyaretçinin hattı bağımsızdır ve
# eşzamanlı istekleri farklı serverless instance'lar karşılasa bile aynı sayaçlar görülür.
ANIMATION_CYCLE = line_sim.CYCLE_SECONDS
DEFECT_CLASSES = line_sim.DEFECT_CLASSES


def line_state():
    state = session.get("line")
    if not line_sim.is_valid(state):
        state = line_sim.new_state(time.time())
        session["line"] = state
        session.permanent = True
    return state


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "SAMEORIGIN",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


@app.after_request
def add_security_headers(response):
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


# --- API ENDPOINTS (FRONTEND İLE İLETİŞİM) ---
@app.route('/api/control', methods=['POST'])
def control():
    """HMI butonlarından gelen komutları işler (yalnızca bu ziyaretçinin hattını etkiler)."""
    command = (request.get_json(silent=True) or {}).get('command')
    session["line"] = line_sim.apply_command(line_state(), command, time.time())
    return jsonify({"status": "ok"})


@app.route('/api/data')
def data():
    """Anlık sistem verilerini JSON olarak döner (3D ikiz saatini bununla senkronlar)."""
    return jsonify(line_sim.snapshot(line_state(), time.time()))


def generate_frame(state):
    """Yedek 2D kamera görüntüsü (WebGL olmayan tarayıcılar için OpenCV çizimi)."""
    snap = line_sim.snapshot(state, time.time())
    progress = (snap["sim_time"] % ANIMATION_CYCLE) / ANIMATION_CYCLE
    current_mode = snap["system_mode"]
    current_status = snap["current_unit_status"]

    # Görsel Çizim (OpenCV)
    w, h = 1280, 720
    frame = np.full((h, w, 3), (20, 25, 30), dtype=np.uint8)
    cv2.rectangle(frame, (0, h//2 - 130), (w, h//2 + 130), (40, 45, 50), -1)
    prod_x = int(w + 100 - (progress * (w + 400)))

    if -200 < prod_x < w:
        if current_mode == 'RUNNING' and current_status != "PENDING":
            is_ok = current_status == "OK"
            color = (129, 185, 16) if is_ok else (68, 68, 239)
            cv2.rectangle(frame, (prod_x, h//2-150), (prod_x+160, h//2+150), (160, 165, 170), -1)
            cv2.rectangle(frame, (prod_x+10, h//2-140), (prod_x+150, h//2+140), (10, 10, 10), -1)
            cv2.rectangle(frame, (prod_x-5, h//2-155), (prod_x+165, h//2+155), color, 2)
            cv2.putText(frame, f"{current_status}", (prod_x+20, h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, "CAM_01 > QC_SCAN", (prod_x, h//2-175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"MATCH: {99.8 if is_ok else 42.1}%", (prod_x, h//2+175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if current_mode != 'RUNNING':
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        msg = "SYSTEM " + current_mode
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        cv2.putText(frame, msg, ((w - text_size[0]) // 2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    return frame

def gen(state):
    while True:
        frame = generate_frame(state)
        (flag, encodedImage) = cv2.imencode('.jpg', frame)
        if not flag: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
        time.sleep(0.03)

# --- REPORTING ENGINE: PANDAS & XLSXWRITER (BUSINESS INTELLIGENCE) ---
@app.route('/api/export_report')
def export_report():
    try:
        # Ziyaretçinin hattındaki tamamlanmış üniteler (deterministik geçmiş -> Pandas DataFrame)
        df = pd.DataFrame(line_sim.history(line_state(), time.time()), columns=['timestamp', 'unit_id', 'status', 'defect', 'oee_score'])
        if df.empty:
            return "Raporlanacak veri yok: hattı başlatıp en az bir ünite tamamlanmasını bekleyin.", 404
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Excel oluşturma (Bellekte / In-Memory)
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
        filename = f"Uretim_Raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print(f"Export Hatası: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(gen(line_state()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
