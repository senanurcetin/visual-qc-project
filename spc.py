"""Statistical process control for the live line: a p-chart of the reject rate per batch."""
from __future__ import annotations

import math
import time

from flask import Blueprint, jsonify

import line_sim
from line_session import line_state

spc_bp = Blueprint("spc", __name__)
BATCH_SIZE = 10
MAX_BATCHES = 30


def p_chart(rows: list[dict], batch_size: int = BATCH_SIZE, max_batches: int = MAX_BATCHES) -> dict:
    """p-chart over complete batches of consecutive units.

    Batch k holds units [k*batch_size, (k+1)*batch_size). Incomplete batches are dropped, the
    centre line is the pooled reject rate of the batches shown and the limits are 3-sigma
    binomial limits clamped to [0, 1].
    """
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["unit_id"][2:]) // batch_size, []).append(row)
    complete = sorted(k for k, g in groups.items() if len(g) == batch_size)[-max_batches:]
    if not complete:
        return {"batch_size": batch_size, "batches": [], "center": None, "ucl": None, "lcl": None}
    rejects = [sum(r["status"] != "OK" for r in groups[k]) for k in complete]
    center = sum(rejects) / (batch_size * len(complete))
    sigma = math.sqrt(center * (1 - center) / batch_size)
    ucl, lcl = min(1.0, center + 3 * sigma), max(0.0, center - 3 * sigma)
    batches = [
        {"batch": k, "p": r / batch_size, "rejects": r, "out_of_control": r / batch_size > ucl or r / batch_size < lcl}
        for k, r in zip(complete, rejects, strict=True)
    ]
    return {"batch_size": batch_size, "batches": batches, "center": center, "ucl": ucl, "lcl": lcl}


@spc_bp.route("/api/spc")
def spc_data():
    rows = line_sim.history(line_state(), time.time(), BATCH_SIZE * (MAX_BATCHES + 1))
    return jsonify(p_chart(rows))


@spc_bp.route("/spc")
def spc_page():
    return SPC_PAGE


SPC_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Line SPC</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { margin: 0; background: #111827; color: #f3f4f6; font-family: Inter, system-ui, sans-serif; }
  main { max-width: 960px; margin: 0 auto; padding: 16px; }
  a { color: #a78bfa; } .box { background: #1f293b; border: 1px solid #374151; border-radius: 4px; padding: 12px; height: 340px; }
  #status { margin: 12px 0; font-family: 'Roboto Mono', monospace; font-size: .85rem; color: #9ca3af; }
  .bad { color: #ef4444; font-weight: 600; }
</style></head>
<body><main>
  <p><a href="/">&larr; Back to HMI</a></p>
  <h1>Reject rate p-chart</h1>
  <p>Share of rejected units per batch of 10, with 3-sigma control limits from the batches shown. A point outside the limits is a signal to investigate the process, not noise.</p>
  <div class="box"><canvas id="chart"></canvas></div>
  <div id="status">Start the line on the HMI to collect batches.</div>
</main>
<script>
  let chart = null;
  function draw(d) {
    const status = document.getElementById('status');
    if (!d.batches.length) { status.textContent = 'Not enough units yet: a batch needs ' + d.batch_size + ' completed units.'; return; }
    const labels = d.batches.map(b => 'B' + b.batch);
    const line = v => d.batches.map(() => v);
    const out = d.batches.filter(b => b.out_of_control).length;
    status.innerHTML = 'Centre ' + (d.center * 100).toFixed(1) + '% &middot; UCL ' + (d.ucl * 100).toFixed(1) + '% &middot; LCL ' + (d.lcl * 100).toFixed(1) + '% &middot; ' +
      (out ? '<span class="bad">' + out + ' batch(es) out of control</span>' : 'in control');
    const data = {
      labels,
      datasets: [
        { label: 'Reject rate', data: d.batches.map(b => b.p * 100), borderColor: '#3b82f6', pointBackgroundColor: d.batches.map(b => b.out_of_control ? '#ef4444' : '#3b82f6'), pointRadius: 4, tension: 0 },
        { label: 'Centre', data: line(d.center * 100), borderColor: '#10b981', borderDash: [6, 4], pointRadius: 0 },
        { label: 'UCL', data: line(d.ucl * 100), borderColor: '#f59e0b', borderDash: [2, 3], pointRadius: 0 },
        { label: 'LCL', data: line(d.lcl * 100), borderColor: '#f59e0b', borderDash: [2, 3], pointRadius: 0 },
      ],
    };
    if (chart) { chart.data = data; chart.update('none'); return; }
    chart = new Chart('chart', { type: 'line', data, options: { responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, title: { display: true, text: '% rejected', color: '#9ca3af' }, grid: { color: '#334155' }, ticks: { color: '#94a3af' } }, x: { ticks: { color: '#94a3af' } } },
      plugins: { legend: { labels: { color: '#f3f4f6' } } } } });
  }
  function load() { fetch('/api/spc').then(r => r.json()).then(draw).catch(() => {}); }
  load(); setInterval(load, 3000);
</script></body></html>
"""
