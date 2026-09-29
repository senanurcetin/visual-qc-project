import os
import time
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request, send_file, session

import line_sim
from camera import gen, make_source
from case_study import case_study_bp
from classify import classify_bp
from hmi_page import HTML_TEMPLATE
from knowledge import knowledge_bp
from line_session import line_state
from rag_demo import rag_bp
from report import build_report
from review import review_bp
from spc import spc_bp

app = Flask(__name__, static_folder="web/line", static_url_path="/line/static")
# Session çerezi yalnızca demo hattının kontrol durumunu imzalar; gizli veri taşımaz.
app.secret_key = os.environ.get("SECRET_KEY", "visual-qc-public-demo")
app.permanent_session_lifetime = timedelta(days=7)
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                  # Polls must never re-send the cookie: a slow GET would overwrite a newer command.
                  SESSION_REFRESH_EACH_REQUEST=False)
app.register_blueprint(case_study_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(review_bp)
app.register_blueprint(spc_bp)
app.register_blueprint(classify_bp)
app.register_blueprint(knowledge_bp)

# --- 2. BACKEND: İŞ MANTIĞI & DURUM YÖNETİMİ ---
# --- HAT SİMÜLASYONU (DURUMSUZ) ---
# Hattın tamamı line_sim içinde, ziyaretçinin imzalı session çerezindeki küçük bir kontrol
# durumundan deterministik olarak hesaplanır. Böylece her ziyaretçinin hattı bağımsızdır ve
# eşzamanlı istekleri farklı serverless instance'lar karşılasa bile aynı sayaçlar görülür.
ANIMATION_CYCLE = line_sim.CYCLE_SECONDS
DEFECT_CLASSES = line_sim.DEFECT_CLASSES


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


# --- REPORTING: PANDAS & XLSXWRITER (workbook built in report.py) ---
@app.route('/api/export_report')
def export_report():
    try:
        # Ziyaretçinin hattındaki tamamlanmış üniteler (deterministik geçmiş -> Pandas DataFrame)
        df = pd.DataFrame(line_sim.history(line_state(), time.time()), columns=['timestamp', 'unit_id', 'status', 'defect', 'oee_score'])
        if df.empty:
            return "Raporlanacak veri yok: hattı başlatıp en az bir ünite tamamlanmasını bekleyin.", 404
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        filename = f"Uretim_Raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        return send_file(
            build_report(df),
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
    return Response(gen(make_source(os.environ.get('FRAME_SOURCE'), line_state())), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
