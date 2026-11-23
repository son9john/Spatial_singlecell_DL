"""Cluster bootstrap intervals for calibration rates and close AUROC comparisons (own code, archived results only)."""
import os
import numpy as np, pandas as pd
R = os.environ.get('JBENCH_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
rng = np.random.default_rng(20261122); NB = 10000
DON0 = {**{s: 'D1' for s in range(151507, 151511)}, **{s: 'D2' for s in range(151669, 151673)}, **{s: 'D3' for s in range(151673, 151677)}}
DON = {**DON0, **{str(k): v for k, v in DON0.items()}}
def hier(diff_by_unit, groups):
    """Resample groups (donors), then units within each drawn group; mean of unit values."""
    g = pd.Series(diff_by_unit).groupby(np.asarray(groups)); arrs = [v.values for _, v in g]; out = np.empty(NB)
    for b in range(NB):
        pick = rng.integers(0, len(arrs), len(arrs))
        out[b] = np.mean(np.concatenate([arrs[i][rng.integers(0, len(arrs[i]), len(arrs[i]))] for i in pick]))
    return np.percentile(out, [2.5, 97.5])
rows = []
def add(name, a, b, units, groups):
    d = (a - b); lo, hi = hier(d.values, groups)
    gm = d.groupby(groups.values).mean()
    rows.append(dict(comparison=name, mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), lo=lo, hi=hi,
                     n_units=len(d), groups_same_sign=f"{int((np.sign(gm) == np.sign(d.mean())).sum())}/{len(gm)}",
                     group_means=';'.join(f'{v:+.4f}' for v in gm)))
# 1-2 regional J (v5 metrics)
m = pd.read_csv(f'{R}/revisions/v5_2026-11-22_104348_EST/results/metrics_by_unit_2026-11-22_104504_EST.csv')
c = m[(m.design == 'B1') & (m.task == 'touch_apart')].pivot(index='unit', columns='method', values='auroc')
grp = pd.Series(c.index.map(DON), index=c.index)
add('Cortex adjacent vs distant: regional J - matched-scale Pearson', c.J_region, c.smooth_pearson, c.index, grp)
add('Cortex adjacent vs distant: regional J - Hotspot', c.J_region, c.hotspot_Z, c.index, grp)
e = m[(m.design == 'C1') & (m.task == 'touch_apart')].pivot(index='unit', columns='method', values='auroc')
add('Embryo contact enrichment: regional J - Giotto', e.J_region, e.giotto_spatial_cor, e.index, pd.Series(e.index, index=e.index))
# 3 touching vs same (Fig 9 source)
f = pd.read_csv(f'{R}/revisions/v13_2026-11-22_175222_EST/results/figure9_boundary_method_auroc_2026-11-22_175222_EST_source.csv')
p = f[f.design == 'B1'].pivot(index='unit', columns='method', values='auroc_as_plotted'); g = pd.Series(p.index.map(DON), index=p.index)
add('Cortex touching vs same-domain: J4 - 16-step Pearson (lower first)', p.J4, p.smooth_pearson_s16, p.index, g)
add('Cortex touching vs same-domain: J1 - Pearson (lower first)', p.J1, p.pearson, p.index, g)
q = f[f.design == 'C1'].pivot(index='unit', columns='method', values='auroc_as_plotted')
add('Embryo touching vs same-type: J4 - Pearson (lower first)', q.J4, q.pearson, q.index, pd.Series(q.index, index=q.index))
# 4 combined model
k = pd.read_csv(f'{R}/results/v3/complementarity_auroc.csv')
for co in ('hotspot_Z', 'lee_l'):
    for task in ('touch_vs_apart_or_same', 'touch_vs_apart'):
        d = k[(k.dataset == 'DLPFC') & (k.task == task) & (k.colocalization == co)].set_index('unit')
        add(f'Cortex {task}: J + {co} - {co} alone', d.auc_combined_lr, d.auc_colocalization, d.index, pd.Series(d.index.astype(int).map(DON), index=d.index))
out = pd.DataFrame(rows); out.to_csv('auroc_cluster_ci.csv', index=False)
pd.set_option('display.width', 250); print(out.round(4).to_string())
# 5 calibration: resample section pairs, then x genes and y genes within the pair
a = pd.read_csv(f'{R}/results/v3/gold_calibration_all.csv.gz', low_memory=False); a = a[a.axis == 'A2']
cal = []
for mode in ('rotated', 'aligned'):
    d = a[a.setting == mode].copy(); d['gx'] = d.pair // 50; d['gy'] = d.pair % 50
    for col in ('J1_spectral', 'lee_perm', 'hotspot_z', 'pearson_t'):
        mats = [(d[d.unit == u].pivot(index='gx', columns='gy', values=col).values < .05) for u in d.unit.unique()]
        bs = np.empty(4000)
        for b in range(4000):
            tot = []
            for i in rng.integers(0, 6, 6):
                M = mats[i]; tot.append(M[np.ix_(rng.integers(0, 50, 50), rng.integers(0, 50, 50))].mean())
            bs[b] = 100*np.mean(tot)
        n = len(d); kk = (d[col] < .05).sum(); ph = kk/n; z = 1.96
        w = (ph + z*z/(2*n)) / (1 + z*z/n); hw = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / (1 + z*z/n)
        cal.append(dict(control=mode, test=col, rate=100*ph, wilson=f'{100*(w-hw):.1f}-{100*(w+hw):.1f}',
                        cluster=f'{np.percentile(bs, 2.5):.1f}-{np.percentile(bs, 97.5):.1f}',
                        unit_range=f"{min(m.mean() for m in mats)*100:.1f}-{max(m.mean() for m in mats)*100:.1f}"))
cal = pd.DataFrame(cal); cal.to_csv('calibration_cluster_ci.csv', index=False); print(cal.round(2).to_string())
