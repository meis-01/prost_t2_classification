"""Reproducible factorial analysis from the completed metrics CSV."""
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'results' / 'test_metrics_all_runs.csv'
REPORT = ROOT / 'report'
OUT = REPORT / 'derived'
PAPER = REPORT / 'paper'
OUT.mkdir(parents=True, exist_ok=True)

def decode(label):
    if label == 'real':
        return dict(mode='real', input_domain='image', pooling='none', normalization='batchnorm', convolution='real', streams='magnitude_only', interaction='none', activation='silu')
    aliases = {'img':'image','ksp':'kspace','med':'median','avg':'average','cbn':'batchnorm','std':'standard','wl':'widely_linear','single':'complex_only','gate':'modulus_gate','holo':'holographic','magsilu':'magnitude_silu','card':'cardioid'}
    if label in {'complex_modrelu','complex_modrelu_median_pool','complex_modrelu_average_pool'}:
        return dict(mode='complex', input_domain='image', pooling={'complex_modrelu':'max','complex_modrelu_median_pool':'median','complex_modrelu_average_pool':'average'}[label], normalization='rms', convolution='standard', streams='complex_only', interaction='none', activation='modrelu')
    vals = label.removeprefix('cx_').split('_')
    # domain, pooling, normalization, convolution, streams, interaction, activation
    dom, pool, norm, conv, stream = vals[:5]
    activation = next((a for a in ('magnitude_silu','modrelu','crelu','cardioid') if label.endswith({'magnitude_silu':'magsilu','modrelu':'modrelu','crelu':'crelu','cardioid':'card'}[a])), 'modrelu')
    interaction = 'holographic' if '_holo_' in label else ('modulus_gate' if '_gate_' in label else 'none')
    return dict(mode='complex', input_domain=aliases.get(dom,dom), pooling=aliases.get(pool,pool), normalization=aliases.get(norm,norm), convolution=aliases.get(conv,conv), streams=aliases.get(stream,stream), interaction=interaction, activation=activation)

def main():
    d = pd.read_csv(CSV)
    d = d[(d.phase == 'phase1') & d.metrics_available].copy()
    factors = pd.DataFrame([decode(x) for x in d.run_label])
    d = pd.concat([d.reset_index(drop=True), factors], axis=1)
    real = float(d.loc[d.run_label == 'real', 'test_auc'].iloc[0])
    cx = d[d['mode'] == 'complex'].copy()
    overall = {k: float(cx[k].mean()) for k in ['test_auc','test_average_precision']}
    summary = {
        'n_phase1': int(len(d)), 'n_complex': int(len(cx)), 'real_auc': real,
        'complex_mean_auc': overall['test_auc'], 'complex_mean_ap': overall['test_average_precision'],
        'complex_min_auc': float(cx.test_auc.min()), 'complex_max_auc': float(cx.test_auc.max()),
        'complex_beating_real': int((cx.test_auc > real).sum()), 'complex_beating_real_fraction': float((cx.test_auc > real).mean()),
        'best_label': str(cx.loc[cx.test_auc.idxmax(),'run_label']), 'best_auc': float(cx.test_auc.max()),
        'canonical': {str(r.run_label): {'auc':float(r.test_auc), 'ap':float(r.test_average_precision)} for _,r in d[d.run_label.isin(['complex_modrelu','complex_modrelu_median_pool','complex_modrelu_average_pool','real'])].iterrows()},
    }
    means = cx.groupby(['input_domain','streams','interaction'], as_index=False)[['test_auc','test_average_precision']].mean()
    norm = cx.groupby(['input_domain','normalization'], as_index=False)[['test_auc','test_average_precision']].mean()
    conv = cx.groupby('convolution', as_index=False)[['test_auc','test_average_precision']].mean()
    effects = []
    mu = cx.test_auc.mean()
    for factor in ['input_domain','pooling','normalization','convolution','activation','streams','interaction']:
        for level, g in cx.groupby(factor): effects.append({'factor':factor,'level':level,'effect_auc':float(g.test_auc.mean()-mu),'mean_auc':float(g.test_auc.mean()),'n':int(len(g))})
    d.to_csv(OUT/'phase1_annotated.csv', index=False)
    means.to_csv(OUT/'domain_stream_interactions.csv', index=False); norm.to_csv(OUT/'domain_normalization.csv', index=False); conv.to_csv(OUT/'convolution.csv', index=False)
    pd.DataFrame(effects).to_csv(OUT/'main_effects.csv', index=False)
    (OUT/'phase1_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    tex = {'real_auc':f'{real:.4f}', 'real_ap':f'{d.loc[d.run_label=="real","test_average_precision"].iloc[0]:.4f}', 'win_count':f'{summary["complex_beating_real"]}', 'image_auc':f'{cx[cx.input_domain=="image"].test_auc.mean():.4f}', 'image_ap':f'{cx[cx.input_domain=="image"].test_average_precision.mean():.4f}', 'kspace_auc':f'{cx[cx.input_domain=="kspace"].test_auc.mean():.4f}', 'kspace_ap':f'{cx[cx.input_domain=="kspace"].test_average_precision.mean():.4f}'}
    for (domain, norm), g in cx.groupby(['input_domain','normalization']): tex[f'{domain}_{norm}_auc'] = f'{g.test_auc.mean():.4f}'
    for (domain, streams, interaction), g in cx.groupby(['input_domain','streams','interaction']): tex[f'{domain}_{streams}_{interaction}_auc'] = f'{g.test_auc.mean():.4f}'
    for (conv,), g in cx.groupby(['convolution']): tex[f'conv_{conv}_auc'] = f'{g.test_auc.mean():.4f}'
    for (act,), g in cx.groupby(['activation']): tex[f'act_{act}_auc'] = f'{g.test_auc.mean():.4f}'
    tex['complex_mean_auc'] = f'{cx.test_auc.mean():.4f}'; tex['complex_mean_ap'] = f'{cx.test_average_precision.mean():.4f}'
    tex['complex_min_auc'] = f'{cx.test_auc.min():.4f}'; tex['complex_max_auc'] = f'{cx.test_auc.max():.4f}'
    tex['best_label'] = summary['best_label'].replace('_','\\_'); tex['best_ap'] = f'{cx.loc[cx.test_auc.idxmax(),"test_average_precision"]:.4f}'
    canonical_labels = ['real','complex_modrelu','complex_modrelu_median_pool','complex_modrelu_average_pool']
    for label, macro in [('complex_modrelu','canonical_max_auc'),('complex_modrelu_median_pool','canonical_median_auc'),('complex_modrelu_average_pool','canonical_average_auc')]:
        tex[macro] = f'{d.loc[d.run_label==label,"test_auc"].iloc[0]:.4f}'
    rows=[]
    for label in canonical_labels:
        r=d[d.run_label==label].iloc[0]; name={'real':'Real magnitude baseline','complex_modrelu':'Canonical max','complex_modrelu_median_pool':'Canonical median','complex_modrelu_average_pool':'Canonical average'}[label]
        rows.append(f'{name} & {r.test_auc:.4f} & {r.test_average_precision:.4f} & {r.test_balanced_accuracy:.4f} & {r.test_sensitivity:.4f} & {r.test_specificity:.4f} \\\\')
    canonical_rows = '\n'.join(rows)
    factor_table=[]
    for factor in ['input_domain','pooling','normalization','convolution','activation']:
        for level,g in cx.groupby(factor):
            factor_table.append(f'{factor.replace("_"," ").title()} & {str(level).replace("_"," ")} & {len(g)} & {g.test_auc.mean():.4f} & {(g.test_auc.mean()-real):+.4f} \\\\')
    stream_levels = [
        ('complex_only', 'none', 'complex-only / none'),
        ('complex_only', 'holographic', 'complex-only / holographic'),
        ('dual', 'none', 'dual / none'),
        ('dual', 'modulus_gate', 'dual / modulus-gate'),
        ('dual', 'holographic', 'dual / holographic'),
    ]
    for streams, interaction, label in stream_levels:
        g = cx[(cx.streams == streams) & (cx.interaction == interaction)]
        factor_table.append(f'Stream/interaction & {label} & {len(g)} & {g.test_auc.mean():.4f} & {(g.test_auc.mean()-real):+.4f} \\\\')
    factor_rows = '\n'.join(factor_table)
    top=cx.nlargest(5,'test_auc')
    top_rows=[]
    for rank,(_,r) in enumerate(top.iterrows(),1):
        stream_interaction = f'({str(r.streams).replace("_", "-")} / {str(r.interaction).replace("_", " ")})'
        factors_text = ' / '.join(str(r[k]).replace('_','\\_') for k in ['input_domain','pooling','normalization','convolution','activation']) + f' / {stream_interaction}'
        top_rows.append(f'{rank} & {str(r.run_label).replace("_","\\_")} & {r.test_auc:.4f} & {r.test_average_precision:.4f} & {factors_text} \\\\')
    top_rows_text = '\n'.join(top_rows)

    # One human-readable scalar include replaces the former one-file-per-value
    # outputs.  Values remain generated directly from the supplied CSV.
    macro_names = {
        'real_auc':'RealAUC', 'real_ap':'RealAP', 'image_auc':'ImageAUC',
        'image_ap':'ImageAP', 'kspace_auc':'FourierAUC', 'kspace_ap':'FourierAP',
        'complex_mean_auc':'ComplexMeanAUC', 'complex_mean_ap':'ComplexMeanAP',
        'complex_min_auc':'ComplexMinAUC', 'complex_max_auc':'BestAUC',
        'best_ap':'BestAP', 'best_label':'BestLabel', 'win_count':'ComplexWins',
        'canonical_max_auc':'CanonicalMaxAUC', 'canonical_median_auc':'CanonicalMedianAUC',
        'canonical_average_auc':'CanonicalAverageAUC',
    }
    macro_lines = [
        '% Generated by report/scripts/analyze_results.py; do not edit by hand.',
        '% All scalar values are regenerated from results/test_metrics_all_runs.csv.',
        '\\newcommand{\\ComplexTotal}{%d\\xspace}' % summary['n_complex'],
    ]
    for key, macro in macro_names.items():
        macro_lines.append('\\newcommand{\\%s}{%s\\xspace}' % (macro, tex[key]))
    # Factor-level values used in the Results prose.
    for key, value in tex.items():
        if key in macro_names or key in {'real_auc','real_ap','image_auc','image_ap','kspace_auc','kspace_ap','complex_mean_auc','complex_mean_ap','complex_min_auc','complex_max_auc','best_ap','best_label','canonical_max_auc','canonical_median_auc','canonical_average_auc'}:
            continue
        macro = ''.join(part.capitalize() for part in key.split('_'))
        if macro.endswith('Auc'):
            macro = macro[:-3] + 'AUC'
        elif macro.endswith('Ap'):
            macro = macro[:-2] + 'AP'
        macro_lines.append('\\newcommand{\\%s}{%s\\xspace}' % (macro, value))
    (PAPER / 'generated_metrics.tex').write_text('\n'.join(macro_lines) + '\n')

    # Table row macros keep the manuscript source readable while preserving
    # generated tabular content in one logically grouped file.
    table_text = '\n'.join([
        '% Generated by report/scripts/analyze_results.py; do not edit by hand.',
        '\\newcommand{\\CanonicalTableRows}{%', canonical_rows, '}',
        '\\newcommand{\\FactorTableRows}{%', factor_rows, '}',
        '\\newcommand{\\TopTableRows}{%', top_rows_text, '}',
        '',
    ])
    (PAPER / 'generated_tables.tex').write_text(table_text)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
