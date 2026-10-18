/*
 * progress-charts.js — live progress charts for training and hyperparameter search.
 * Needs Plotly (already loaded by the pages). Exposes window.ProgressCharts:
 *
 *   leaderboard(el, opts)  Data Training: models ranked by score as they finish
 *   trialChart(el, opts)   Hyperparameter Search: one model's trials + best-so-far
 *   autopilot(el, opts)    Autopilot: best-over-time, current-round search, score by round
 *
 * Every chart takes the SSE payloads the workers already emit, so a page just
 * forwards events. Redraws are throttled (a fast model can emit hundreds of trials
 * a second) and use Plotly.react, so charts update in place.
 *
 * Colour: categorical hues come from the validated dark palette, assigned to an
 * ENTITY (a model id) in first-seen order and remembered for the browser session —
 * a model keeps its colour in every chart on every page, whatever else is showing.
 * Past 8 entities the rest share a neutral grey. Anything in the plot area that is
 * text uses text tokens, never a series colour.
 */
(function () {
  'use strict';

  const T = {
    surface: '#111827', grid: '#1f2937', axis: '#475569',
    tick: '#94a3b8', text: '#e2e8f0', muted: '#64748b',
  };
  // Validated categorical order (dark mode) — see the dataviz palette reference.
  const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
  const OTHER = '#64748b';
  const ORDER_KEY = 'chartEntityOrder';
  const CONFIG = { displayModeBar: false, responsive: true };

  let entityOrder = [];
  try { entityOrder = JSON.parse(sessionStorage.getItem(ORDER_KEY)) || []; } catch (e) { entityOrder = []; }

  function colorFor(entity) {
    let i = entityOrder.indexOf(entity);
    if (i < 0) {
      entityOrder.push(entity);
      i = entityOrder.length - 1;
      try { sessionStorage.setItem(ORDER_KEY, JSON.stringify(entityOrder)); } catch (e) { /* storage unavailable */ }
    }
    return i < SERIES.length ? SERIES[i] : OTHER;
  }

  function nameOf(id) {
    try { if (typeof MODEL_LABELS !== 'undefined' && MODEL_LABELS[id]) return MODEL_LABELS[id]; } catch (e) { /* not defined */ }
    return id;
  }

  // ── styles (injected once) ──
  const style = document.createElement('style');
  style.textContent = `
    .pc-card { background:${T.surface}; border:1px solid #1e2030; border-radius:12px; padding:12px 14px; margin:12px 0; }
    .pc-title { font-size:.82rem; color:#cbd5e1; margin-bottom:2px; }
    .pc-sub { font-size:.72rem; color:${T.muted}; margin-bottom:6px; }
    .pc-cap { font-size:.74rem; color:${T.muted}; margin-top:4px; font-variant-numeric:tabular-nums; }
    .pc-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:12px 0; }
    .pc-kpi { background:${T.surface}; border:1px solid #1e2030; border-radius:10px; padding:10px 12px; }
    .pc-kpi b { display:block; font-size:1.2rem; color:${T.text}; font-variant-numeric:tabular-nums; font-weight:600; }
    .pc-kpi span { font-size:.7rem; color:${T.muted}; }
    .pc-table { width:100%; border-collapse:collapse; font-size:.76rem; margin-top:8px; font-variant-numeric:tabular-nums; }
    .pc-table th, .pc-table td { text-align:left; padding:4px 8px; border-bottom:1px solid #1e2030; color:${T.tick}; }
    .pc-table th { color:${T.muted}; font-weight:500; }
    details.pc-details summary { cursor:pointer; font-size:.76rem; color:#a5b4fc; margin-top:8px; }
  `;
  document.head.appendChild(style);

  // ── helpers ──
  function mergeAxis(base, extra) { return Object.assign({}, base, extra || {}); }

  function layout(extra) {
    extra = extra || {};
    const out = Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif', size: 12, color: T.tick },
      margin: { l: 56, r: 24, t: 52, b: 44 },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: '#0b1220', bordercolor: '#334155', font: { color: T.text, size: 12 } },
      legend: { orientation: 'h', x: 0, y: 1.27, font: { color: T.tick }, bgcolor: 'rgba(0,0,0,0)' },
    }, extra);
    out.xaxis = mergeAxis({ gridcolor: T.grid, zeroline: false, linecolor: T.axis, tickcolor: T.axis, automargin: true }, extra.xaxis);
    out.yaxis = mergeAxis({ gridcolor: T.grid, zeroline: false, automargin: true }, extra.yaxis);
    return out;
  }

  /** Throttled, in-place redraw. `build()` returns {data, layout}. */
  function live(el, build) {
    let pending = false;
    function draw() {
      pending = false;
      if (!window.Plotly) return;
      const fig = build();
      if (fig) window.Plotly.react(el, fig.data, fig.layout, CONFIG);
    }
    return { redraw() { if (!pending) { pending = true; setTimeout(draw, 250); } }, now: draw };
  }

  /** y-range that ignores a few terrible outliers (failed/awful trials) at the bottom. */
  function yRange(values, lowQuantile) {
    const v = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return undefined;
    const lo = v[Math.floor((v.length - 1) * (lowQuantile || 0))];
    const hi = v[v.length - 1];
    const span = Math.max(hi - lo, 1e-3);
    return [lo - span * 0.08, hi + span * 0.15];
  }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function card(host, title, sub) {
    const c = el('div', 'pc-card');
    c.appendChild(el('div', 'pc-title', title));
    if (sub) c.appendChild(el('div', 'pc-sub', sub));
    host.appendChild(c);
    return c;
  }

  function unavailable(host) {
    host.appendChild(el('div', 'pc-cap', 'Charts unavailable (Plotly did not load).'));
  }

  function fmtAgo(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '—';
    if (seconds < 90) return 'just now';
    const m = Math.round(seconds / 60);
    if (m < 90) return `${m} min ago`;
    const h = seconds / 3600;
    return h < 48 ? `${h.toFixed(1)} h ago` : `${(h / 24).toFixed(1)} d ago`;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Data Training — live leaderboard
  // ═══════════════════════════════════════════════════════════════════════
  function leaderboard(host, opts) {
    opts = opts || {};
    const metricKey = opts.metricKey || 'Accuracy';
    const metricLabel = opts.metricLabel || metricKey;
    const rows = new Map();     // model -> {value, elapsed}
    const failed = [];
    let total = 0;
    let top3 = null;

    const c = card(host, 'Model comparison — live', `${metricLabel} on the cross-validated sample, best at the top`);
    const plot = el('div'); c.appendChild(plot);
    const cap = el('div', 'pc-cap'); c.appendChild(cap);
    if (!window.Plotly) { unavailable(c); }

    const L = live(plot, () => {
      const sorted = [...rows.entries()].sort((a, b) => a[1].value - b[1].value);   // ascending: Plotly draws the first row at the bottom
      const n = sorted.length;
      plot.style.height = `${Math.max(180, 30 * Math.max(n, total ? Math.min(total, 20) : 1) + 70)}px`;
      const blue = SERIES[0];
      const colors = sorted.map(([m]) => (top3 && !top3.includes(m)) ? 'rgba(57,135,229,0.38)' : blue);
      return {
        data: [{
          type: 'bar', orientation: 'h',
          y: sorted.map(([m]) => nameOf(m)), x: sorted.map(([, r]) => r.value),
          customdata: sorted.map(([, r]) => r.elapsed),
          marker: { color: colors, line: { color: T.surface, width: 2 } },
          text: sorted.map(([, r]) => r.value.toFixed(4)), textposition: 'outside', cliponaxis: false,
          textfont: { color: T.text, size: 11 },
          hovertemplate: `%{y}<br>${metricLabel} %{x:.4f}<br>%{customdata}s to fit<extra></extra>`,
        }],
        layout: layout({
          hovermode: 'closest', showlegend: false,
          margin: { l: 120, r: 60, t: 8, b: 40 },
          xaxis: { title: { text: metricLabel }, rangemode: 'tozero' },
          yaxis: { gridcolor: 'rgba(0,0,0,0)', automargin: true },
          bargap: 0.3,
        }),
      };
    });

    function caption() {
      const parts = [`${rows.size + failed.length}${total ? ' / ' + total : ''} models compared`];
      if (top3) parts.push('top 3 highlighted');
      if (failed.length) parts.push(`${failed.length} failed: ${failed.map(nameOf).join(', ')}`);
      cap.textContent = parts.join(' · ');
    }

    return {
      setTotal(n) { total = n; caption(); L.redraw(); },
      add(d) {
        if (d.error || !d.metrics || typeof d.metrics[metricKey] !== 'number') failed.push(d.model);
        else rows.set(d.model, { value: d.metrics[metricKey], elapsed: d.elapsed_s != null ? d.elapsed_s : '—' });
        caption(); L.redraw();
      },
      addMany(map) { Object.entries(map).forEach(([m, v]) => this.add(v && v.error ? { model: m, error: v.error } : { model: m, metrics: v })); },
      setTop3(ids) { top3 = ids; caption(); L.redraw(); },
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Hyperparameter Search — one model's trials
  // ═══════════════════════════════════════════════════════════════════════
  function trialChart(host, opts) {
    opts = opts || {};
    const scoreLabel = opts.scoreLabel || 'Score';
    const color = colorFor(opts.model || 'model');
    const st = { x: [], y: [], bx: [], by: [], failed: 0, stages: [], perStage: null, n: 0 };

    const c = card(host, 'Search progress — live',
      'Each dot is one trial (cross-validated); the line is the best score found so far');
    const plot = el('div'); plot.style.height = '260px'; c.appendChild(plot);
    const cap = el('div', 'pc-cap'); c.appendChild(cap);
    if (!window.Plotly) unavailable(c);

    const L = live(plot, () => {
      const lastX = st.bx.length ? st.bx[st.bx.length - 1] : null;
      const lastY = st.by.length ? st.by[st.by.length - 1] : null;
      return {
        data: [
          {
            type: 'scatter', mode: 'markers', name: 'Trial score', x: st.x, y: st.y,
            marker: { color, size: 8, opacity: 0.6, line: { color: T.surface, width: 1 } },
            hovertemplate: 'trial %{x}: %{y:.4f}<extra></extra>',
          },
          {
            type: 'scatter', mode: 'lines', name: 'Best so far', x: st.bx, y: st.by,
            line: { color: T.text, width: 2, shape: 'hv' },
            hovertemplate: 'best %{y:.4f}<extra></extra>',
          },
        ],
        layout: layout({
          xaxis: { title: { text: 'Trial' }, rangemode: 'tozero' },
          yaxis: { title: { text: scoreLabel }, range: yRange(st.y.concat(st.by), 0.05) },
          hovermode: 'closest',
          shapes: st.stages.map(s => ({ type: 'line', xref: 'x', yref: 'paper', x0: s.x, x1: s.x, y0: 0, y1: 1,
                                        line: { color: T.axis, width: 1, dash: 'dot' } })),
          annotations: st.stages.map(s => ({ xref: 'x', yref: 'paper', x: s.x, y: 1, xanchor: 'left', yanchor: 'bottom',
                                             text: s.label, showarrow: false, font: { color: T.muted, size: 10 } }))
            .concat(lastX == null ? [] : [{ xref: 'x', yref: 'y', x: lastX, y: lastY, xanchor: 'right', yanchor: 'bottom', yshift: 6,
                                            text: lastY.toFixed(4), showarrow: false, font: { color: T.text, size: 11 } }]),
        }),
      };
    });

    function caption() {
      const best = st.by.length ? st.by[st.by.length - 1].toFixed(4) : '—';
      cap.textContent = `${st.n} trials · best ${scoreLabel} ${best}` +
        (st.failed ? ` · ${st.failed} failed trial${st.failed > 1 ? 's' : ''} not plotted` : '');
    }

    return {
      setPerStage(n) { st.perStage = n; },
      markStage(d) {
        if (!st.perStage) return;
        const label = d.stages > 1 ? `Stage ${d.stage}/${d.stages} · ${Math.round(d.sample_frac * 100)}% data` : '';
        if (label) st.stages.push({ x: (d.stage - 1) * st.perStage + 0.5, label });
        L.redraw();
      },
      addTrial(d) {
        const x = d.trial + 1;
        st.n += 1;
        const bad = d.value == null || d.value <= -1e8 || (opts.classification && d.value === 0);
        if (bad) st.failed += 1; else { st.x.push(x); st.y.push(d.value); }
        st.bx.push(x); st.by.push(d.best_value);
        caption(); L.redraw();
      },
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Autopilot — three linked views + headline numbers
  // ═══════════════════════════════════════════════════════════════════════
  function autopilot(host, opts) {
    opts = opts || {};
    const scoreLabel = opts.scoreLabel || 'Score';
    const bests = new Map();      // `${round}|${model}|${score}` -> {ts, score, model, round}
    const results = new Map();    // `${round}|${model}` -> {round, model, score, level, frac, mode}
    const rounds = new Map();     // round -> {round, mode, level, frac, start_ts, end_ts}
    let cur = null;               // current round_start payload
    let curModel = null;
    const curTrials = new Map();  // model -> {bx, by, px, py}
    let lastTs = 0;

    // headline numbers
    const kpis = el('div', 'pc-kpis');
    const kBest = el('div', 'pc-kpi'); const kRounds = el('div', 'pc-kpi');
    const kNow = el('div', 'pc-kpi'); const kImp = el('div', 'pc-kpi');
    [kBest, kRounds, kNow, kImp].forEach(k => { k.append(el('b', null, '—'), el('span', null, '')); kpis.appendChild(k); });
    host.appendChild(kpis);
    function setKpi(k, big, small) { k.firstChild.textContent = big; k.lastChild.textContent = small; }

    const c1 = card(host, 'Best hold-out score over time',
      `Each step is a new best model (hold-out ${scoreLabel}). Shaded bands are rounds.`);
    const p1 = el('div'); p1.style.height = '280px'; c1.appendChild(p1);
    const c2 = card(host, 'Current round — search progress',
      'Best cross-validated score so far for each model, by trial. Dots: the model being searched now.');
    const p2 = el('div'); p2.style.height = '280px'; c2.appendChild(p2);
    const cap2 = el('div', 'pc-cap'); c2.appendChild(cap2);
    const c3 = card(host, 'Hold-out score by round',
      'Each model\'s refit score per round. Axis labels: complexity level (L) and share of training data searched.');
    const p3 = el('div'); p3.style.height = '300px'; c3.appendChild(p3);
    const det = el('details', 'pc-details'); det.appendChild(el('summary', null, 'Data table'));
    const tableHost = el('div'); det.appendChild(tableHost); c3.appendChild(det);
    if (!window.Plotly) unavailable(host);

    const sortedBests = () => [...bests.values()].sort((a, b) => a.ts - b.ts);
    const phaseOf = (r) => rounds.get(r) || {};

    const L1 = live(p1, () => {
      const b = sortedBests();
      const x = b.map(r => new Date(r.ts * 1000));
      const y = b.map(r => r.score);
      const end = Math.max(lastTs, b.length ? b[b.length - 1].ts : 0);
      if (b.length) { x.push(new Date(end * 1000)); y.push(y[y.length - 1]); }
      const rs = [...rounds.values()].sort((p, q) => p.round - q.round);
      const label = rs.length <= 24;
      const shapes = rs.filter(r => r.start_ts).map((r, i) => ({
        type: 'rect', xref: 'x', yref: 'paper', layer: 'below', line: { width: 0 },
        x0: new Date(r.start_ts * 1000), x1: new Date((r.end_ts || end || r.start_ts) * 1000), y0: 0, y1: 1,
        fillcolor: i % 2 ? 'rgba(148,163,184,0.00)' : 'rgba(148,163,184,0.07)',
      }));
      const notes = label ? rs.filter(r => r.start_ts).map(r => ({
        xref: 'x', yref: 'paper', x: new Date(r.start_ts * 1000), y: 1, xanchor: 'left', yanchor: 'bottom',
        text: `R${r.round + 1}`, showarrow: false, font: { color: T.muted, size: 10 },
      })) : [];
      if (b.length) {
        const lastB = b[b.length - 1];
        notes.push({ xref: 'x', yref: 'y', x: x[x.length - 1], y: lastB.score, xanchor: 'right', yanchor: 'bottom', yshift: 8,
                     text: `${nameOf(lastB.model)} ${lastB.score.toFixed(4)}`, showarrow: false, font: { color: T.text, size: 11 } });
      }
      return {
        data: [{
          type: 'scatter', mode: 'lines+markers', name: `Best ${scoreLabel}`, x, y,
          customdata: b.map(r => `${nameOf(r.model)} · round ${r.round + 1}`).concat(b.length ? [''] : []),
          line: { color: SERIES[0], width: 2, shape: 'hv' },
          marker: { color: SERIES[0], size: 8, line: { color: T.surface, width: 2 } },
          hovertemplate: `%{y:.4f}<br>%{customdata}<extra></extra>`,
        }],
        layout: layout({
          showlegend: false, hovermode: 'closest', shapes, annotations: notes,
          xaxis: { type: 'date' },
          yaxis: { title: { text: `Hold-out ${scoreLabel}` }, range: yRange(y, 0) },
        }),
      };
    });

    const L2 = live(p2, () => {
      const data = []; const notes = [];
      const entries = [...curTrials.entries()].filter(([, t]) => t.bx.length);
      const direct = entries.length <= 4;
      entries.forEach(([m, t]) => {
        const col = colorFor(m);
        data.push({ type: 'scatter', mode: 'lines', name: nameOf(m), x: t.bx, y: t.by,
                    line: { color: col, width: 2, shape: 'hv' }, hovertemplate: '%{y:.4f}<extra>' + nameOf(m) + '</extra>' });
        if (m === curModel && t.px.length) {
          data.push({ type: 'scatter', mode: 'markers', x: t.px, y: t.py, showlegend: false, hoverinfo: 'skip',
                      marker: { color: col, size: 6, opacity: 0.35 } });
        }
        if (direct) notes.push({ xref: 'x', yref: 'y', x: t.bx[t.bx.length - 1], y: t.by[t.by.length - 1], xanchor: 'left', xshift: 6,
                                 text: nameOf(m), showarrow: false, font: { color: T.tick, size: 11 } });
      });
      const all = entries.flatMap(([, t]) => t.by);
      return {
        data,
        layout: layout({
          hovermode: 'x unified', annotations: notes, margin: { l: 56, r: direct ? 110 : 24, t: 52, b: 44 },
          xaxis: { title: { text: 'Trial' }, rangemode: 'tozero' },
          yaxis: { title: { text: `CV ${scoreLabel}` }, range: yRange(all, 0) },
        }),
      };
    });

    const L3 = live(p3, () => {
      const byModel = new Map();
      [...results.values()].forEach(r => { if (!byModel.has(r.model)) byModel.set(r.model, []); byModel.get(r.model).push(r); });
      const direct = byModel.size <= 4;
      const data = []; const notes = [];
      byModel.forEach((rs, m) => {
        rs.sort((a, b) => a.round - b.round);
        const col = colorFor(m);
        data.push({ type: 'scatter', mode: 'lines+markers', name: nameOf(m), x: rs.map(r => r.round + 1), y: rs.map(r => r.score),
                    line: { color: col, width: 2 }, marker: { color: col, size: 7, line: { color: T.surface, width: 2 } },
                    hovertemplate: '%{y:.4f}<extra>' + nameOf(m) + '</extra>' });
        const last = rs[rs.length - 1];
        if (direct) notes.push({ xref: 'x', yref: 'y', x: last.round + 1, y: last.score, xanchor: 'left', xshift: 8,
                                 text: nameOf(m), showarrow: false, font: { color: T.tick, size: 11 } });
      });
      const roundNums = [...new Set([...results.values()].map(r => r.round))].sort((a, b) => a - b);
      const step = Math.max(1, Math.ceil(roundNums.length / 24));
      const ticks = roundNums.filter((_, i) => i % step === 0);
      const info = (r) => { const p = phaseOf(r); const lv = p.level || (results.get([...results.keys()].find(k => k.startsWith(r + '|'))) || {}).level;
                            const fr = p.frac != null ? p.frac : (results.get([...results.keys()].find(k => k.startsWith(r + '|'))) || {}).frac;
                            return `${r + 1}<br>L${lv ?? '?'} · ${fr != null ? Math.round(fr * 100) : '?'}%`; };
      const shapes = [];
      if (roundNums.some(r => r >= 6)) {
        shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: 6.5, x1: 6.5, y0: 0, y1: 1, line: { color: T.axis, width: 1, dash: 'dot' } });
        notes.push({ xref: 'x', yref: 'paper', x: 6.5, y: 1, xanchor: 'left', yanchor: 'bottom', text: 'full data from here', showarrow: false, font: { color: T.muted, size: 10 } });
      }
      const all = [...results.values()].map(r => r.score);
      return {
        data,
        layout: layout({
          annotations: notes, shapes, margin: { l: 56, r: direct ? 110 : 24, t: 52, b: 60 },
          xaxis: { title: { text: 'Round' }, tickmode: 'array', tickvals: ticks.map(r => r + 1), ticktext: ticks.map(info) },
          yaxis: { title: { text: `Hold-out ${scoreLabel}` }, range: yRange(all, 0) },
        }),
      };
    });

    function renderTable() {
      const rowsArr = [...results.values()].sort((a, b) => b.round - a.round || b.score - a.score).slice(0, 200);
      const t = el('table', 'pc-table');
      const head = el('tr'); ['Round', 'Model', `Hold-out ${scoreLabel}`, 'Complexity', 'Search data'].forEach(h => head.appendChild(el('th', null, h)));
      t.appendChild(head);
      rowsArr.forEach(r => {
        const tr = el('tr');
        [r.round + 1, nameOf(r.model), r.score.toFixed(4), r.level != null ? `L${r.level}` : '—', r.frac != null ? `${Math.round(r.frac * 100)}%` : '—']
          .forEach(v => tr.appendChild(el('td', null, String(v))));
        t.appendChild(tr);
      });
      tableHost.replaceChildren(t);
    }

    function renderKpis() {
      const b = sortedBests();
      const top = b[b.length - 1];
      setKpi(kBest, top ? top.score.toFixed(4) : '—', top ? `best hold-out ${scoreLabel} · ${nameOf(top.model)}` : 'best hold-out score');
      const done = [...rounds.values()].filter(r => r.end_ts).length;
      setKpi(kRounds, String(done), 'rounds completed');
      setKpi(kNow, cur ? `Round ${cur.round + 1} · ${cur.mode}` : '—',
        cur ? `complexity ${cur.level}/6 · ${Math.round(cur.frac * 100)}% data · ${cur.trials} trials/model` : 'current round');
      const ref = Math.max(lastTs, Date.now() / 1000 - 0);   // wall clock so a stalled search shows its age
      setKpi(kImp, top ? fmtAgo(ref - top.ts) : '—', 'since last improvement');
    }

    function redrawAll() { renderKpis(); L1.redraw(); L2.redraw(); L3.redraw(); renderTable(); }
    setInterval(renderKpis, 30000);

    const api = {
      load(h) {
        (h.bests || []).forEach(r => bests.set(`${r.round}|${r.model}|${r.score}`, r));
        (h.results || []).forEach(r => results.set(`${r.round}|${r.model}`, r));
        (h.rounds || []).forEach(r => rounds.set(r.round, r));
        const all = [].concat(h.bests || [], h.results || []).map(r => r.ts || 0);
        lastTs = Math.max(lastTs, ...all, 0);
        redrawAll();
      },
      ingest(event, d) {
        if (d.ts) lastTs = Math.max(lastTs, d.ts);
        switch (event) {
          case 'round_start':
            cur = d; curModel = null; curTrials.clear();
            rounds.set(d.round, { round: d.round, mode: d.mode, level: d.level, frac: d.frac, start_ts: d.ts, end_ts: null });
            cap2.textContent = '';
            break;
          case 'model_start':
            curModel = d.model;
            if (!curTrials.has(d.model)) curTrials.set(d.model, { bx: [], by: [], px: [], py: [] });
            cap2.textContent = `Searching ${nameOf(d.model)}…`;
            break;
          case 'trial_result': {
            const t = curTrials.get(d.model) || curTrials.set(d.model, { bx: [], by: [], px: [], py: [] }).get(d.model);
            t.bx.push(d.trial + 1); t.by.push(d.best_value);
            const bad = d.value == null || d.value <= -1e8 || (opts.classification && d.value === 0);
            if (!bad) { t.px.push(d.trial + 1); t.py.push(d.value); }
            cap2.textContent = `${nameOf(d.model)}: trial ${d.trial + 1}/${d.n_trials} · best CV ${scoreLabel} ${Number(d.best_value).toFixed(4)}`;
            break;
          }
          case 'model_result':
            if (!d.error && d.score != null) {
              const p = rounds.get(d.round) || {};
              results.set(`${d.round}|${d.model}`, { round: d.round, model: d.model, score: d.score,
                                                     level: p.level, frac: p.frac, mode: p.mode, ts: d.ts });
            }
            break;
          case 'new_best':
            bests.set(`${d.round}|${d.model}|${Math.round(d.score * 1e4) / 1e4}`, { ts: d.ts || d.saved_at, score: Math.round(d.score * 1e4) / 1e4, model: d.model, round: d.round });
            break;
          case 'round_done': {
            const r = rounds.get(d.round);
            if (r) r.end_ts = d.ts;
            break;
          }
          default: return;
        }
        redrawAll();
      },
      reset() { bests.clear(); results.clear(); rounds.clear(); curTrials.clear(); cur = null; curModel = null; lastTs = 0; redrawAll(); },
    };
    renderKpis();
    return api;
  }

  window.ProgressCharts = { leaderboard, trialChart, autopilot, colorFor };
})();
