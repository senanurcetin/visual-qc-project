"""HMI page (HTML/CSS/JS) served at `/`."""

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

        .control-bar { flex-shrink: 0; padding: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; background: #111827; border-bottom: 1px solid var(--border); }
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
        .review-sec { border-top: 3px solid var(--purple); max-height: 150px; }
        .review-list { overflow-y: auto; font-family: 'Roboto Mono', monospace; font-size: 0.75rem; }
        .review-row { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-bottom: 1px solid #334155; }
        .review-row .who { flex: 1; } .review-row .done { color: var(--green); }
        .review-row button, .review-row select { background: #334155; color: var(--text); border: 1px solid var(--border); border-radius: 3px; font: inherit; padding: 2px 6px; cursor: pointer; }
        .review-empty { padding: 8px; color: #9ca3af; }
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
        <a href="/spc" class="btn btn-case-study">SPC Chart</a>
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
            <div class="panel review-sec"><div class="section-header">Review Queue &middot; <a href="/api/review/export.csv" style="color:#a78bfa">corrections CSV</a></div><div class="review-list" id="review-list"><div class="review-empty">No rejected units yet.</div></div></div>
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


    // --- Review queue: confirm or correct the predicted defect of rejected units ---
    function loadReview() {
        fetch('/api/review-queue').then(r => r.json()).then(q => {
            const list = document.getElementById('review-list');
            if (!list) return;
            if (!q.items.length) { list.innerHTML = '<div class="review-empty">No rejected units yet.</div>'; return; }
            list.innerHTML = q.items.map(i => {
                const options = q.classes.map(c => `<option value="${c}">${c.replace(/_/g, ' ')}</option>`).join('');
                const state = i.decision
                    ? `<span class="done">${i.decision === 'confirm' ? '&#10003; confirmed' : '&#9998; ' + i.operator_label.replace(/_/g, ' ')}</span>`
                    : `<button onclick="submitReview('${i.unit_id}','${i.predicted_defect}')">Confirm</button>
                       <select onchange="if(this.value)submitReview('${i.unit_id}',this.value)"><option value="">Correct&hellip;</option>${options}</select>`;
                return `<div class="review-row"><span class="who">${i.unit_id} &middot; ${i.predicted_defect.replace(/_/g, ' ')}</span>${state}</div>`;
            }).join('');
        }).catch(() => {});
    }

    function submitReview(unitId, label) {
        fetch('/api/review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({unit_id: unitId, label: label})
        }).then(loadReview);
    }

    setInterval(loadReview, 4000);
    loadReview();

    setInterval(update, 1000);
    window.onload = initCharts;
</script>
</body>
</html>
"""
