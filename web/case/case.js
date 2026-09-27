// Visual QC case study: renders the NEU-CLS benchmark from /api/case-study as an interactive page.
const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const nice = (s) => s.replace(/_/g, " ");
const COLORS = ["#f472b6", "#fb923c", "#a3e635", "#60a5fa", "#c084fc", "#2dd4bf"];
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function countUp(el, target, fmt) {
    if (reduced) { el.textContent = fmt(target); return; }
    const start = performance.now();
    const step = (t) => {
        const k = Math.min(1, (t - start) / 1100);
        el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
        if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function onVisible(el, fn) {
    el.classList.add("reveal");
    new IntersectionObserver((entries, io) => {
        if (entries.some((e) => e.isIntersecting)) { el.classList.add("in"); fn?.(); io.disconnect(); }
    }, { threshold: 0.25 }).observe(el);
}

function bar(label, value, color, text, max = 1) {
    return `<div class="bar-row"><span>${label}</span><span class="bar-track"><span class="bar-fill" style="width:${(value / max) * 100}%;background:${color}"></span></span><span class="bar-val">${text}</span></div>`;
}

function render(d) {
    const s = d.summary, fm = s.final_model;
    const errors = s.pareto_summary.total_errors, holdout = s.dataset.evaluation_split.holdout_samples;
    const top = d.review_queue.review_budgets.at(-1);
    const kpis = [
        ["Accuracy", fm.accuracy, (v) => pct(v), "Random Forest, HOG + Gabor + grid"],
        ["Macro F1", fm.macro_f1, (v) => v.toFixed(3), "balanced across 6 classes"],
        ["Mistakes", errors, (v) => `${Math.round(v)}`, `of ${holdout} holdout images`],
        [`Caught at ${pct(top.review_fraction, 0)} review`, top.error_capture_rate, (v) => pct(v), `${top.yield_lift_vs_random.toFixed(1)}× random review`],
    ];
    $("kpis").innerHTML = kpis.map(([k, , , sub], i) => `<div class="kpi"><dt>${k}</dt><dd id="kpi-${i}">0</dd><dt style="margin-top:4px">${sub}</dt></div>`).join("");
    kpis.forEach(([, v, fmt], i) => countUp($(`kpi-${i}`), v, fmt));

    // Defect classes
    const f1 = Object.fromEntries(d.class_metrics.map((m) => [m.class_name, m.f1]));
    $("defects").innerHTML = d.taxonomy.map((t, i) => `<article class="defect" style="--c:${COLORS[i]}">
        <span class="f1">F1 ${f1[t.class_name]?.toFixed(2) ?? "—"}</span><h3>${esc(nice(t.class_name))}</h3><p>${esc(t.description)}</p></article>`).join("");
    onVisible($("defects"));

    // Model race
    const names = { dummy_baseline: "Dummy baseline", logistic_regression: "Logistic regression", random_forest: "Random Forest" };
    const best = Math.max(...d.benchmarks.map((b) => b.macro_f1));
    $("race").innerHTML = d.benchmarks.map((b) => `<div class="race-row ${b.macro_f1 === best ? "win" : ""}">
        <span class="name">${names[b.model] || b.model}</span><span class="race-track"><span class="race-fill" data-w="${b.accuracy}" style="background:${b.macro_f1 === best ? "linear-gradient(90deg,#34d399,#6ee7b7)" : "#3b82f6"}"></span></span>
        <span class="val">${pct(b.accuracy)}</span></div>`).join("") + `<p class="card-note" style="margin:4px 0 0">Accuracy on the holdout. ${esc(d.model_selection.selection_reason || "")}</p>`;
    onVisible($("race"), () => document.querySelectorAll(".race-fill").forEach((el) => { el.style.width = `${el.dataset.w * 100}%`; }));

    // Confusion matrix
    const { labels, matrix } = d.confusion;
    const maxOff = Math.max(...matrix.flatMap((r, i) => r.filter((_, j) => j !== i)));
    let html = `<span></span>${labels.map((l) => `<span class="lab col">${esc(nice(l))}</span>`).join("")}`;
    matrix.forEach((row, i) => {
        html += `<span class="lab">${esc(nice(labels[i]))}</span>`;
        row.forEach((v, j) => {
            const bg = i === j ? `rgba(110,231,183,${0.15 + 0.6 * v / 60})` : v ? `rgba(248,113,113,${0.25 + 0.65 * v / maxOff})` : "rgba(33,65,102,.25)";
            html += `<button class="cell" data-i="${i}" data-j="${j}" style="background:${bg}" aria-label="${nice(labels[i])} predicted as ${nice(labels[j])}: ${v}">${v || ""}</button>`;
        });
    });
    $("cm").innerHTML = html;
    const readout = (i, j) => {
        const v = matrix[i][j];
        $("cm-readout").innerHTML = i === j
            ? `<strong>${v}/${row(i)}</strong>${esc(nice(labels[i]))} correctly recognised.`
            : `<strong>${v}</strong>${esc(nice(labels[i]))} ${v === 1 ? "image was" : "images were"} read as <b>${esc(nice(labels[j]))}</b>.`;
        document.querySelectorAll(".cm .cell").forEach((c) => c.classList.toggle("on", +c.dataset.i === i && +c.dataset.j === j));
    };
    const row = (i) => matrix[i].reduce((a, b) => a + b, 0);
    $("cm").addEventListener("pointerover", (e) => { const c = e.target.closest(".cell"); if (c) readout(+c.dataset.i, +c.dataset.j); });
    $("cm").addEventListener("focusin", (e) => { const c = e.target.closest(".cell"); if (c) readout(+c.dataset.i, +c.dataset.j); });
    const hot = d.confusion.hotspots[0];
    readout(labels.indexOf(hot.actual), labels.indexOf(hot.predicted));

    const misses = labels.map((l, i) => [l, row(i) - matrix[i][i]]).filter(([, m]) => m).sort((a, b) => b[1] - a[1]);
    let cum = 0;
    $("pareto").innerHTML = `<div class="pareto">${misses.map(([l, m]) => { cum += m; return bar(nice(l), m, COLORS[labels.indexOf(l)], `${m} · ${pct(cum / errors, 0)}`, misses[0][1]); }).join("")}</div>`;
    onVisible(document.querySelector(".cm-layout"));

    // Review queue
    const budgets = d.review_queue.review_budgets;
    $("budget").innerHTML = budgets.map((b, i) => `<button type="button" data-i="${i}" aria-pressed="false">Review ${pct(b.review_fraction, 0)}</button>`).join("");
    $("waffle").innerHTML = Array.from({ length: errors }, () => "<i></i>").join("");
    const pick = (i) => {
        const b = budgets[i];
        document.querySelectorAll("#budget button").forEach((x) => x.setAttribute("aria-pressed", String(+x.dataset.i === i)));
        countUp($("q-capture"), b.error_capture_rate, (v) => pct(v, 0));
        $("q-caption").textContent = `${b.captured_errors} of ${errors} misclassifications in ${b.reviewed_samples} reviewed images`;
        document.querySelectorAll("#waffle i").forEach((el, k) => el.classList.toggle("hit", k < b.captured_errors));
        $("q-bars").innerHTML = bar("Entropy queue", b.review_yield, "#6ee7b7", `${pct(b.review_yield)} of reviews find an error`, 0.3)
            + bar("Random", b.random_review_yield, "#8da4c4", `${pct(b.random_review_yield)} · ${b.yield_lift_vs_random.toFixed(1)}× lift`, 0.3);
    };
    $("budget").addEventListener("click", (e) => { const x = e.target.closest("button"); if (x) pick(+x.dataset.i); });
    onVisible(document.querySelector(".queue"), () => pick(budgets.length - 1));

    // Cost
    const maxCost = Math.max(...d.class_metrics.map((m) => m.fn_cost_per_missed));
    $("cost-classes").innerHTML = `<h3>False-negative cost per missed part</h3><div class="bars" style="margin-top:12px">${d.class_metrics
        .slice().sort((a, b) => b.fn_cost_per_missed - a.fn_cost_per_missed)
        .map((m) => bar(nice(m.class_name), m.fn_cost_per_missed, COLORS[labels.indexOf(m.class_name)], `$${m.fn_cost_per_missed}`, maxCost)).join("")}</div>
        <p class="card-foot">Inclusion is both the least precise class and the most expensive miss, so it drives the review queue, not the class with the lowest F1.</p>`;
    const c = s.cost_model_summary;
    $("cost-total").innerHTML = `<p class="panel-label">Illustrative cost on the holdout</p><span class="big" id="cost-save">0%</span>
        <p class="card-note" style="margin:0">lower than shipping without a model: $${c.total_model_cost.toLocaleString()} vs $${c.naive_no_model_cost.toLocaleString()}.</p>`;
    onVisible($("cost-total"), () => countUp($("cost-save"), c.cost_savings_share, (v) => pct(v)));

    $("limits").innerHTML = s.limitations.map((l) => `<li>${esc(l)}</li>`).join("");
}

fetch(document.body.dataset.src).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then(render)
    .catch((err) => { console.error(err); $("kpis").innerHTML = `<p class="card-note">Case-study artifacts could not be loaded. Run python analysis/run_neu_case_study.py.</p>`; });
