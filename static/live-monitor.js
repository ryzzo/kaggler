/*
 * live-monitor.js — live hardware chart + full activity log for long-running jobs.
 * Needs Plotly. Exposes window.LiveMonitor:
 *
 *   const mon = LiveMonitor.mount(host);   // renders the panel into `host`
 *   mon.start(); mon.stop();               // begin/stop sampling /resources (~1.5 s)
 *   mon.log(level, text);                  // level: info | ok | warn | error | detail
 *   mon.attach(eventSource, names?);       // log every named SSE event via describe()
 *
 * The hardware chart shows CPU %, GPU %, VRAM % and RAM % over the last few minutes,
 * so you can see at a glance whether a run is really using the GPU.
 */
(function () {
  'use strict';

  const T = { surface: '#111827', grid: '#1f2937', axis: '#475569', tick: '#94a3b8', text: '#e2e8f0', muted: '#64748b' };
  const LINES = [
    { key: 'cpu',  name: 'CPU',  color: '#3987e5' },
    { key: 'gpu',  name: 'GPU',  color: '#199e70' },
    { key: 'vram', name: 'VRAM', color: '#c98500' },
    { key: 'ram',  name: 'RAM',  color: '#9085e9' },
  ];
  const LEVEL_COLOR = { info: '#cbd5e1', ok: '#6ee7b7', warn: '#fbbf24', error: '#fca5a5', detail: '#64748b' };
  const MAX_SAMPLES = 160;   // ~4 min at 1.5 s
  const MAX_LOG = 1500;

  const style = document.createElement('style');
  style.textContent = `
    .lm-card { background:${T.surface}; border:1px solid #1e2030; border-radius:12px; padding:12px 14px; margin:12px 0; }
    .lm-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
    .lm-title { font-size:.82rem; color:#cbd5e1; font-weight:600; }
    .lm-live { font-size:.68rem; font-weight:700; padding:2px 8px; border-radius:999px; background:#2a2f3a; color:#94a3b8; }
    .lm-live.on { background:#064e3b; color:#6ee7b7; }
    .lm-now { display:flex; gap:14px; flex-wrap:wrap; font-size:.74rem; color:${T.muted}; font-variant-numeric:tabular-nums; margin-left:auto; }
    .lm-now b { color:#e2e8f0; font-weight:600; }
    .lm-now i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }
    .lm-log { height:240px; overflow-y:auto; background:#0b1220; border-radius:8px; padding:8px 10px;
              font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
    .lm-line { white-space:pre-wrap; word-break:break-word; }
    .lm-line .t { color:#475569; margin-right:8px; }
    .lm-tools { display:flex; gap:12px; align-items:center; margin:6px 0; font-size:.72rem; color:${T.muted}; }
    .lm-tools label { cursor:pointer; display:flex; gap:5px; align-items:center; }
    .lm-tools button { background:none; border:1px solid #334155; color:#94a3b8; border-radius:6px; padding:2px 10px; font-size:.72rem; cursor:pointer; }
    .lm-tools button:hover { border-color:#6366f1; color:#e2e8f0; }
  `;
  document.head.appendChild(style);

  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const fmt = v => (typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4)) : String(v));
  const secs = v => (v == null ? '' : ` (${v}s)`);

  /** Turn one worker event into [level, text]; null = don't log. */
  function describe(name, d) {
    d = d || {};
    const m = d.model || d.ensemble || '';
    switch (name) {
      case 'sample_ready': return ['info', `Sample ready: ${d.n_sample?.toLocaleString()} of ${d.n_total?.toLocaleString()} rows (${Math.round((d.sample_frac || 0) * 100)}%)`];
      case 'gpu_info': {
        const parts = Object.entries(d).filter(([k, v]) => typeof v === 'boolean').map(([k, v]) => `${k}=${v ? 'yes' : 'no'}`);
        return [d.warning ? 'warn' : 'info', `Compute: ${d.device ? 'device=' + d.device + ' · ' : ''}${parts.join(' · ')}${d.warning ? ' — ' + d.warning : ''}`];
      }
      case 'compare_start': return ['info', `Comparing ${d.total} models: ${(d.models || []).join(', ')}`];
      case 'model_start': return ['info', `▶ ${m} — starting (${(d.index ?? 0) + 1}/${d.total ?? '?'})`];
      case 'fold_result': return ['detail', `   ${m} fold ${d.fold}/${d.folds}: ${Object.entries(d.metrics || {}).map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')}` +
        (d.running ? `   | running mean ${Object.entries(d.running).map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')}` : '') + secs(d.elapsed_s)];
      case 'compare_result': return d.error ? ['error', `✗ ${m} failed: ${d.error}`]
        : ['ok', `✓ ${m}: ${Object.entries(d.metrics || {}).slice(0, 4).map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')}${secs(d.elapsed_s)}`];
      case 'leaderboard_done': return ['ok', `Leaderboard complete — top 3: ${(d.top3 || []).join(', ')}`];
      case 'best_model_fit_start': return ['info', `Fitting best model ${m} on the full dataset…`];
      case 'fit_progress': return ['detail', d.iteration != null ? `   ${m} fit ${d.iteration}/${d.total} iterations${secs(d.elapsed_s)}` : `   ${m} still fitting…${secs(d.elapsed_s)}`];
      case 'best_model_saved': return d.error ? ['error', `Could not save ${m}: ${d.error}`] : ['ok', `Saved ${m} → ${d.filename}${secs(d.elapsed_s)}`];
      case 'search_start': return ['info', `▶ Search ${m}: ${d.n_trials} trials${d.stages > 1 ? ` over ${d.stages} stages` : ''}`];
      case 'stage_start': return ['info', `  Stage ${d.stage}/${d.stages}: ${Math.round((d.sample_frac || 0) * 100)}% sample (${d.n_sample?.toLocaleString()} rows)`];
      case 'trial_start': return ['detail', `   trial ${d.trial + 1} ${m} params ${JSON.stringify(d.params)}`];
      case 'trial_result': return ['info', `   trial ${d.trial + 1}/${d.n_trials} ${m}: ${d.value == null ? 'failed' : fmt(d.value)} · best ${fmt(d.best_value)}${secs(d.elapsed_s)}`];
      case 'stage_done': return ['ok', `  Stage ${d.stage} done — best ${fmt(d.best_value)}`];
      case 'search_done': return d.error ? ['error', `✗ Search ${m} failed: ${d.error}`] : ['ok', `✓ Search ${m} done — best ${fmt(d.best_value)} ${JSON.stringify(d.best_params || {})}`];
      case 'full_setup': return ['info', `Full-dataset setup: ${d.n_train?.toLocaleString()} train / ${d.n_test?.toLocaleString()} test rows`];
      case 'full_refit_result': return d.error ? ['error', `✗ Refit ${m}: ${d.error}`] : ['ok', `✓ Refit ${m} on full data${secs(d.elapsed_s)}`];
      case 'baseline_start': return ['info', `Baseline: ${m} with default hyperparameters`];
      case 'baseline_result': return d.error ? ['error', `Baseline failed: ${d.error}`] : ['ok', `Baseline ${m} scored`];
      case 'full_cv_start': return ['info', `Cross-validating ${m} on the full dataset (${d.folds} folds)`];
      case 'cv_progress': return ['detail', `   cross-validating ${m}…${secs(d.elapsed_s)}`];
      case 'full_cv_result': return d.error ? ['error', `CV ${m} failed: ${d.error}`] : ['ok', `CV ${m}: mean ${fmt(d.mean)} ± ${fmt(d.std)}`];
      case 'ensemble_start': return ['info', `Building ensemble ${m} from ${(d.members || []).join(', ')}`];
      case 'ensemble_result': return d.error ? ['error', `Ensemble ${m} failed: ${d.error}`] : ['ok', `Ensemble ${m} scored`];
      case 'saved': return ['ok', `Saved ${(d.saved_models || []).length} model file(s)`];
      case 'done': return ['ok', 'Run complete.'];
      case 'error': return ['error', `Error: ${d.detail || 'unknown'}`];
      default: return null;
    }
  }

  const ALL_EVENTS = ['sample_ready', 'gpu_info', 'compare_start', 'model_start', 'fold_result', 'compare_result', 'leaderboard_done',
    'best_model_fit_start', 'fit_progress', 'best_model_saved', 'search_start', 'stage_start', 'trial_start', 'trial_result',
    'stage_done', 'search_done', 'full_setup', 'full_refit_result', 'baseline_start', 'baseline_result', 'full_cv_start',
    'cv_progress', 'full_cv_result', 'ensemble_start', 'ensemble_result', 'saved', 'done'];

  function mount(host, opts) {
    opts = opts || {};
    const card = el('div', 'lm-card');
    const head = el('div', 'lm-head');
    head.appendChild(el('span', 'lm-title', opts.title || 'Live hardware & activity'));
    const badge = el('span', 'lm-live', 'idle'); head.appendChild(badge);
    const now = el('div', 'lm-now'); head.appendChild(now);
    card.appendChild(head);

    const plot = el('div'); plot.style.height = '170px'; card.appendChild(plot);

    const tools = el('div', 'lm-tools');
    const detailLbl = el('label'); const detailCb = document.createElement('input'); detailCb.type = 'checkbox'; detailCb.checked = true;
    detailLbl.appendChild(detailCb); detailLbl.appendChild(document.createTextNode('Show fine detail (folds, trials, iterations)'));
    const copyBtn = el('button', null, 'Copy log');
    tools.appendChild(el('span', null, 'Activity log')); tools.appendChild(detailLbl); tools.appendChild(copyBtn);
    card.appendChild(tools);
    const logBox = el('div', 'lm-log'); card.appendChild(logBox);
    host.appendChild(card);

    const S = { t: [], cpu: [], gpu: [], vram: [], ram: [] };
    let timer = null, busy = false, lines = [], lastFit = 0;

    function draw() {
      if (!window.Plotly) return;
      const traces = LINES.filter(l => S[l.key].some(v => v != null)).map(l => ({
        type: 'scatter', mode: 'lines', name: l.name, x: S.t, y: S[l.key], connectgaps: false,
        line: { color: l.color, width: 2 }, hovertemplate: `${l.name} %{y:.0f}%<extra></extra>`,
      }));
      Plotly.react(plot, traces, {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: { l: 40, r: 10, t: 8, b: 28 },
        font: { size: 11, color: T.tick }, hovermode: 'x unified',
        xaxis: { gridcolor: T.grid, zeroline: false, type: 'date', tickformat: '%H:%M:%S' },
        yaxis: { range: [0, 100], gridcolor: T.grid, zeroline: false, ticksuffix: '%' },
        legend: { orientation: 'h', x: 0, y: 1.2, font: { color: T.tick } },
      }, { displayModeBar: false, responsive: true });
    }

    async function sample() {
      if (busy) return; busy = true;
      try {
        const d = await fetch('/resources').then(r => r.json());
        const g = d.gpu || {};
        S.t.push(new Date()); S.cpu.push(d.cpu_pct);
        S.gpu.push(d.gpu_available && g.util_pct != null ? g.util_pct : null);
        S.vram.push(d.gpu_available && g.vram_pct != null ? g.vram_pct : null);
        S.ram.push(d.ram && d.ram.percent != null ? d.ram.percent : null);
        Object.keys(S).forEach(k => { if (S[k].length > MAX_SAMPLES) S[k].shift(); });
        const last = k => S[k][S[k].length - 1];
        now.innerHTML = LINES.filter(l => last(l.key) != null)
          .map(l => `<span><i style="background:${l.color}"></i>${l.name} <b>${Math.round(last(l.key))}%</b></span>`).join('') +
          (d.gpu_available ? '' : '<span>No NVIDIA GPU</span>');
        draw();
      } catch (e) { /* server busy — skip this sample */ } finally { busy = false; }
    }

    function render(entry) {
      const row = el('div', 'lm-line'); row.dataset.level = entry.level;
      row.style.color = LEVEL_COLOR[entry.level] || LEVEL_COLOR.info;
      row.appendChild(el('span', 't', entry.time)); row.appendChild(document.createTextNode(entry.text));
      if (entry.level === 'detail' && !detailCb.checked) row.style.display = 'none';
      const stick = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 30;
      logBox.appendChild(row);
      while (logBox.childElementCount > MAX_LOG) logBox.removeChild(logBox.firstChild);
      if (stick) logBox.scrollTop = logBox.scrollHeight;
    }

    const api = {
      log(level, text) {
        const entry = { level, text, time: new Date().toLocaleTimeString([], { hour12: false }) };
        lines.push(entry); if (lines.length > MAX_LOG) lines.shift();
        render(entry);
      },
      start() { badge.textContent = 'running'; badge.classList.add('on'); sample(); if (!timer) timer = setInterval(sample, 1500); },
      stop() { badge.textContent = 'finished'; badge.classList.remove('on'); if (timer) { clearInterval(timer); timer = null; } },
      attach(es, names) {
        es.addEventListener('error', e => {   // server-sent errors carry data; plain connection drops don't
          let d = null; try { d = JSON.parse(e.data); } catch (x) { /* connection error */ }
          if (d && d.detail) api.log('error', 'Error: ' + d.detail);
        });
        (names || ALL_EVENTS).forEach(n => es.addEventListener(n, e => {
          let d = null; try { d = JSON.parse(e.data); } catch (x) { /* ignore */ }
          if (n === 'fit_progress') { const t = Date.now(); if (t - lastFit < 1500) return; lastFit = t; }
          const r = describe(n, d); if (r) api.log(r[0], r[1]);
        }));
      },
    };
    detailCb.addEventListener('change', () => logBox.querySelectorAll('[data-level="detail"]').forEach(r => { r.style.display = detailCb.checked ? '' : 'none'; }));
    copyBtn.addEventListener('click', () => {
      const text = lines.map(l => `${l.time} [${l.level}] ${l.text}`).join('\n');
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => { copyBtn.textContent = 'Copied'; setTimeout(() => { copyBtn.textContent = 'Copy log'; }, 1500); }).catch(() => {});
    });
    draw();
    return api;
  }

  window.LiveMonitor = { mount, describe };
})();
