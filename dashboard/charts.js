(function () {
  const LC = 'Lewis & Clark';
  const failed = (el, d) => { el.replaceChildren(); el.textContent = d.status === 'error' ? 'This chart could not load.' : '—'; };
  const measure = (svg, labels, cls) => {
    let w = 0;
    labels.forEach((t) => { const n = svg.append('text').attr('class', cls || '').text(t); w = Math.max(w, n.node().getComputedTextLength()); n.remove(); });
    return w;
  };
  const tipAt = (el, tip, ev, lines) => {
    tip.replaceChildren();
    lines.forEach((l, i) => { const div = document.createElement('div'); div.textContent = l; if (i === 0) div.style.fontWeight = '600'; tip.appendChild(div); });
    tip.hidden = false;
    const [x, y] = d3.pointer(ev, el);
    const tw = tip.offsetWidth, W = el.clientWidth;
    tip.style.left = Math.max(0, Math.min(W - tw, x + 12)) + 'px';
    tip.style.top = (y + 14) + 'px';
  };
  const mkTip = (el) => { const t = document.createElement('div'); t.className = 'cr-tip'; t.hidden = true; el.appendChild(t); return t; };

  // ---------- current risk bars
  dash.draw('bars', (el) => {
    const d = dash.data('current_risk'), b = dash.data('benchmarks');
    if (d.status !== 'ok') return failed(el, d);
    el.replaceChildren();
    const rows = d.data.map((r) => ({ ...r, v: +r.risk_3yr_pct })).sort((a, c) => c.v - a.v);
    const medRow = b.status === 'ok' ? b.data.find((r) => r.key === 'np4_median_3yr') : null;
    const med = medRow ? +medRow.value : NaN;
    const W = Math.max(300, el.clientWidth), rowH = 20, top = 22;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('role', 'img')
      .attr('aria-label', 'Bar chart of each college’s three-year closure score, with the national median marked');
    const left = measure(svg, rows.map((r) => r.college)) + 10;
    const right = measure(svg, ['0.00%']) + 12;
    const H = top + rows.length * rowH + 8;
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
    const x = d3.scaleLinear().domain([0, Math.max(d3.max(rows, (r) => r.v), isNaN(med) ? 0 : med) * 1.08]).range([left, W - right]);
    const y = (i) => top + i * rowH;
    const g = svg.append('g');
    g.selectAll('text.lab').data(rows).join('text').attr('x', left - 8).attr('y', (r, i) => y(i) + rowH / 2 + 4).attr('text-anchor', 'end')
      .attr('class', (r) => r.college === LC ? 'cr-strong' : null).text((r) => r.college);
    g.selectAll('rect.bar').data(rows).join('rect').attr('x', left).attr('y', (r, i) => y(i) + 3).attr('height', rowH - 6)
      .attr('width', (r) => Math.max(1, x(r.v) - left))
      .style('fill', (r) => r.college === LC ? dash.colors[0] : 'var(--cds-chart-muted)');
    g.selectAll('text.val').data(rows).join('text').attr('x', (r) => x(r.v) + 4).attr('y', (r, i) => y(i) + rowH / 2 + 4)
      .attr('class', (r) => r.college === LC ? 'cr-strong' : null).text((r) => d3.format('.2f')(r.v) + '%');
    if (!isNaN(med)) {
      g.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', top - 4).attr('y2', H - 4)
        .style('stroke', 'var(--cds-chart-reference)').style('stroke-dasharray', '4 3');
      const lab = g.append('text').attr('y', top - 8).text('Median nonprofit 4-yr ' + d3.format('.2f')(med) + '%');
      const lw = lab.node().getComputedTextLength();
      lab.attr('x', Math.max(left, Math.min(x(med) - lw / 2, W - lw)));
    }
    const tip = mkTip(el);
    g.selectAll('rect.hit').data(rows).join('rect').attr('x', 0).attr('y', (r, i) => y(i)).attr('width', W).attr('height', rowH)
      .style('fill', 'transparent')
      .on('mousemove', (ev, r) => tipAt(el, tip, ev, [r.college, r.group, 'Chance of closing by 2027: ' + d3.format('.2f')(r.v) + '%', 'Percentile: ' + r.percentile, 'This year alone: ' + d3.format('.2f')(+r.risk_this_year_pct) + '%']))
      .on('mouseleave', () => { tip.hidden = true; });
  });

  // ---------- history lines
  dash.draw('lines', (el) => {
    const d = dash.data('risk_history');
    if (d.status !== 'ok') return failed(el, d);
    el.replaceChildren();
    const order = ['Lewis & Clark', 'Group 1 median', 'Group 2 median', 'Group 3 median'];
    const rows = d.data.map((r) => ({ year: +r.year, series: r.series, v: +r.percentile }));
    const byS = d3.group(rows, (r) => r.series);
    const years = Array.from(new Set(rows.map((r) => r.year))).sort((a, b) => a - b);
    const W = Math.max(300, el.clientWidth), H = 260, m = { t: 10, r: 14, b: 26 };
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
      .attr('role', 'img').attr('aria-label', 'Line chart of Lewis & Clark’s closure-risk percentile and the three peer-group medians, 2004 to 2025');
    const ymax = Math.max(50, Math.ceil(d3.max(rows, (r) => r.v) / 10) * 10);
    m.l = measure(svg, [String(ymax)]) + 10;
    const x = d3.scaleLinear().domain(d3.extent(years)).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, ymax]).range([H - m.b, m.t]);
    svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(W / 90))).tickFormat(d3.format('d')).tickSizeOuter(0))
      .call((g) => g.select('.domain').style('stroke', 'var(--cds-chart-axis)')).call((g) => g.selectAll('line').style('stroke', 'var(--cds-chart-axis)'));
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)))
      .call((g) => g.select('.domain').remove()).call((g) => g.selectAll('line').style('stroke', 'var(--cds-chart-grid)'));
    const line = d3.line().x((r) => x(r.year)).y((r) => y(r.v));
    order.forEach((s, i) => {
      const pts = (byS.get(s) || []).sort((a, b) => a.year - b.year);
      svg.append('path').datum(pts).attr('d', line).style('fill', 'none').style('stroke', dash.colors[i]).style('stroke-width', i === 0 ? 3 : 1.5);
    });
    const tip = mkTip(el);
    const guide = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-reference)').style('visibility', 'hidden');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b).style('fill', 'transparent')
      .on('mousemove', (ev) => {
        const [px] = d3.pointer(ev);
        const yr = years.reduce((a, b) => Math.abs(x(b) - px) < Math.abs(x(a) - px) ? b : a);
        guide.attr('x1', x(yr)).attr('x2', x(yr)).style('visibility', 'visible');
        tipAt(el, tip, ev, [String(yr)].concat(order.map((s) => { const r = (byS.get(s) || []).find((q) => q.year === yr); return s + ': ' + (r ? d3.format('.0f')(r.v) : '—'); })));
      })
      .on('mouseleave', () => { tip.hidden = true; guide.style('visibility', 'hidden'); });
  });

  // ---------- scenario dumbbell
  dash.draw('dumbbell', (el) => {
    const d = dash.data('scenarios_2031'), md = dash.data('scenario_medians_2031');
    if (d.status !== 'ok') return failed(el, d);
    el.replaceChildren();
    const rows = d.data.map((r) => ({ ...r, b: +r.baseline, g: +r.cliff_gradual, f: +r.cliff_gradual_fixed50 })).sort((a, c) => c.f - a.f);
    const medR = md.status === 'ok' ? md.data.find((r) => r.scenario === 'cliff_gradual_fixed50') : null;
    const med = medR ? +medR.np4_median_pct : NaN;
    const W = Math.max(300, el.clientWidth), rowH = 20, top = 22;
    const svg = d3.select(el).append('svg').attr('width', '100%').attr('role', 'img')
      .attr('aria-label', 'Dot chart of each college’s 2031 closure score under flat enrollment, a gradual 15% decline, and the decline with half of costs fixed');
    const left = measure(svg, rows.map((r) => r.college)) + 10, right = measure(svg, ['0.00%']) + 12;
    const H = top + rows.length * rowH + 8;
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
    const x = d3.scaleLinear().domain([0, Math.max(d3.max(rows, (r) => Math.max(r.b, r.g, r.f)), isNaN(med) ? 0 : med) * 1.08]).range([left, W - right]);
    const y = (i) => top + i * rowH + rowH / 2;
    const g = svg.append('g');
    if (!isNaN(med)) {
      g.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', top - 4).attr('y2', H - 4).style('stroke', 'var(--cds-chart-reference)').style('stroke-dasharray', '4 3');
      const lab = g.append('text').attr('y', top - 8).text('Median nonprofit 4-yr, fixed-cost case ' + d3.format('.2f')(med) + '%');
      const lw = lab.node().getComputedTextLength();
      lab.attr('x', Math.max(left, Math.min(x(med) - lw / 2, W - lw)));
    }
    g.selectAll('text.lab').data(rows).join('text').attr('x', left - 8).attr('y', (r, i) => y(i) + 4).attr('text-anchor', 'end')
      .attr('class', (r) => r.college === LC ? 'cr-strong' : null).text((r) => r.college);
    g.selectAll('line.rng').data(rows).join('line').attr('x1', (r) => x(Math.min(r.b, r.g, r.f))).attr('x2', (r) => x(Math.max(r.b, r.g, r.f)))
      .attr('y1', (r, i) => y(i)).attr('y2', (r, i) => y(i)).style('stroke', 'var(--cds-chart-grid)').style('stroke-width', 2);
    const dot = (key, fill, rad) => g.selectAll('circle.' + key).data(rows).join('circle').attr('cx', (r) => x(r[key])).attr('cy', (r, i) => y(i)).attr('r', rad).style('fill', fill);
    dot('b', 'var(--cds-chart-muted)', 4); dot('g', dash.colors[1], 4); dot('f', dash.colors[0], 5);
    g.selectAll('text.val').data(rows).join('text').attr('x', (r) => x(Math.max(r.b, r.g, r.f)) + 8).attr('y', (r, i) => y(i) + 4)
      .attr('class', (r) => r.college === LC ? 'cr-strong' : null).text((r) => d3.format('.2f')(r.f) + '%');
    const tip = mkTip(el);
    g.selectAll('rect.hit').data(rows).join('rect').attr('x', 0).attr('y', (r, i) => y(i) - rowH / 2).attr('width', W).attr('height', rowH).style('fill', 'transparent')
      .on('mousemove', (ev, r) => tipAt(el, tip, ev, [r.college + ' (' + r.group + '), 2031',
        'Flat enrollment: ' + d3.format('.2f')(r.b) + '%', '2019-23 decline persists: ' + d3.format('.2f')(+r.persist) + '%',
        '15% drop in 2025: ' + d3.format('.2f')(+r.cliff_onetime) + '%', '15% gradual decline: ' + d3.format('.2f')(r.g) + '%',
        'Gradual decline, half of costs fixed: ' + d3.format('.2f')(r.f) + '%']))
      .on('mouseleave', () => { tip.hidden = true; });
  });

  // ---------- tables
  const table = (name, id, cols, opts) => dash.draw(name, (tb) => {
    const d = dash.data(id);
    tb.replaceChildren();
    if (d.status !== 'ok') {
      const tr = document.createElement('tr'), td = document.createElement('td');
      td.colSpan = cols.length; td.textContent = d.status === 'error' ? 'This table could not load.' : '—'; tr.appendChild(td); tb.appendChild(tr); return;
    }
    d.data.forEach((r) => {
      const tr = document.createElement('tr');
      if (r.college === LC) tr.className = 'cr-lc';
      cols.forEach(([c, f, shade]) => {
        const td = document.createElement('td');
        const v = r[c];
        td.textContent = f ? crFmt(v, f) : (v == null ? '—' : String(v));
        if (shade && v != null && v !== '') td.style.background = `color-mix(in srgb, var(--color-bad) ${Math.round(+v * 0.45)}%, transparent)`;
        if (opts && opts.left && opts.left.includes(c)) td.className = 'cr-left';
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  });
  table('distress_rows', 'distress_outlook', [['college'], ['group'], ['revenue_down_10_pct', 'd', 1], ['neg_margin_3_of_5_pct', 'd', 1], ['enrollment_down_10_pct', 'd', 1], ['enrollment_slide_3yr_pct', 'd', 1]]);
  table('metric_rows', 'metrics_2023', [['college'], ['group'], ['fall_2023_enrollment', ',d'], ['fy2023_margin_pct', '.1f'], ['tuition_share_pct', 'd'], ['days_cash_proxy', ',d'], ['endowment_m', ',d'], ['plant_debt_m', ',d']]);
  table('auc_rows', 'model_check', [['model'], ['description'], ['auc_this_year', '.1f'], ['paper_auc_this_year', '.1f'], ['auc_this_year_np4', '.1f'], ['auc_three_year', '.1f'], ['paper_auc_three_year', '.1f']], { left: ['description'] });
  table('oos_rows', 'later_closures_check', [['institution'], ['event'], ['rank', ',d'], ['of', ',d'], ['risk_3yr_pct', '.1f'], ['top_decile']], { left: ['event'] });
})();
