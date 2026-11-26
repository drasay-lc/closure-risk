(function () {
  const LC = 'Lewis & Clark';
  window.crFmt = (v, p) => (v == null || v === '' || isNaN(+v)) ? '—' : d3.format(p || ',.2f')(+v);
  window.crRow = (id, col, val) => { const d = dash.data(id); return d.status === 'ok' ? d.data.find(r => r[col] === val) : null; };

  dash.calc('peers_below_median', {
    inputs: ['current_risk', 'benchmarks'],
    title: 'Colleges below the national median',
    description: 'Count of the comparison colleges (L&C and its peers) whose 3-year closure score is below the median private nonprofit four-year college.',
    fn: (current_risk, benchmarks) => {
      const m = +((benchmarks.find(r => r.key === 'np4_median_3yr') || {}).value);
      return current_risk.filter(r => +r.risk_3yr_pct < m).length;
    }
  });
  dash.calc('oos_hits', {
    inputs: ['later_closures_check'], title: 'Later closures in the riskiest tenth',
    description: 'Count of later closures whose 2025 score ranked in the top decile.',
    fn: (later_closures_check) => later_closures_check.filter(r => r.top_decile === 'Yes').length
  });
  dash.calc('oos_n', {
    inputs: ['later_closures_check'], title: 'Later closures checked',
    description: 'Number of later closures found in the 2025 scores.',
    fn: (later_closures_check) => later_closures_check.length
  });

  const text = (name, id, col, val, field, p) => dash.draw(name, (el) => {
    const r = crRow(id, col, val);
    el.textContent = r ? crFmt(r[field], p) : '—';
  });
  text('lc_risk', 'current_risk', 'college', LC, 'risk_3yr_pct', '.2f');
  text('lc_pct', 'current_risk', 'college', LC, 'percentile', 'd');
  text('np4_n', 'benchmarks', 'key', 'np4_count', 'value', ',d');
  text('np4_med', 'benchmarks', 'key', 'np4_median_3yr', 'value', '.2f');
  text('tile_med', 'benchmarks', 'key', 'np4_median_3yr', 'value', '.2f');
  text('tile_pct', 'current_risk', 'college', LC, 'percentile', 'd');
  text('tile_base', 'scenarios_2031', 'college', LC, 'baseline', '.2f');
  text('lc_enr10', 'distress_outlook', 'college', LC, 'enrollment_down_10_pct', 'd');
  text('lc_rev10', 'distress_outlook', 'college', LC, 'revenue_down_10_pct', 'd');
  text('cl_base', 'scenarios_2031', 'college', LC, 'baseline', '.2f');
  text('cl_grad', 'scenarios_2031', 'college', LC, 'cliff_gradual', '.2f');
  text('cl_fix', 'scenarios_2031', 'college', LC, 'cliff_gradual_fixed50', '.2f');
  text('lc_margin', 'metrics_2023', 'college', LC, 'fy2023_margin_pct', '.1f');
  text('lc_tuit', 'metrics_2023', 'college', LC, 'tuition_share_pct', 'd');
  text('sys_base', 'system_closures_2027_2031', 'scenario', 'baseline', 'closures_per_year_lasso', 'd');
  text('sys_grad', 'system_closures_2027_2031', 'scenario', 'cliff_gradual', 'closures_per_year_lasso', 'd');
  text('sys_fix', 'system_closures_2027_2031', 'scenario', 'cliff_gradual_fixed50', 'closures_per_year_lasso', 'd');
  text('sys_per', 'system_closures_2027_2031', 'scenario', 'persist', 'closures_per_year_lasso', 'd');
  text('np4_fix', 'scenario_medians_2031', 'scenario', 'cliff_gradual_fixed50', 'np4_median_pct', '.2f');
  dash.draw('tile_risk', (el) => { const r = crRow('current_risk', 'college', LC); el.textContent = r ? crFmt(r.risk_3yr_pct, '.2f') + '%' : '—'; });
  dash.draw('tile_cliff', (el) => { const r = crRow('scenarios_2031', 'college', LC); el.textContent = r ? crFmt(r.cliff_gradual_fixed50, '.2f') + '%' : '—'; });
  for (const k of ['peers_below', 'oos_hits', 'oos_n']) {
    const src = k === 'peers_below' ? 'peers_below_median' : k;
    dash.draw(k, (el) => { const d = dash.data(src); el.textContent = d.status === 'ok' && d.data != null ? String(d.data) : '—'; });
  }

  // legend swatches
  const sw = () => document.querySelectorAll('.cr [data-sw]').forEach((s) => {
    const k = s.getAttribute('data-sw');
    s.style.background = k === 'muted' ? 'var(--cds-chart-muted)' : dash.colors[+k];
  });
  dash.onData(sw);
})();
