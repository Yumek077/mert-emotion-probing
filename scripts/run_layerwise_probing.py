"""Module D: all-level selection, then one gated held-out sweep (no extraction)."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from run_basic_probing import atomic_write_csv, atomic_write_json, file_identity, utc_now
from mert_emotion_probing import probing as p

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/results'
CACHE = ROOT / 'outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt'
LABELS = ROOT / 'data/raw/deam/verification/deam_item_mapping.csv'
SPLIT = ROOT / 'data/metadata/deam_primary_split_seed42.csv'
SELECTION = OUT / 'module_d_stage1_validation.json'
VAL_PRED = OUT / 'module_d_stage1_validation_predictions.csv'
GATE = OUT / 'module_d_stage1_pre_test_verification.json'
TEST = OUT / 'module_d_stage1_test.json'
TEST_PRED = OUT / 'module_d_stage1_test_predictions.csv'
STARTED = OUT / 'module_d_stage1_test_started.json'
FINAL_VERIFY = OUT / 'module_d_stage1_final_verification.json'
TABLE = OUT / 'module_d_stage1_layerwise_results.csv'
FIGURE = ROOT / 'outputs/figures/module_d_stage1_test_r2_trajectory.png'
C3 = OUT / 'module_c_stage3_validation.json'
C4 = OUT / 'module_c_stage4_test.json'
C4_PRED = OUT / 'module_c_stage4_test_predictions.csv'
PROTOCOL = {
    'rq': 'How does linear Valence and Arousal decodability vary across the 13 frozen MERT representation levels?',
    'levels': list(p.EXPECTED_LEVEL_NAMES), 'targets': p.TARGET_COLUMNS,
    'population': 1744, 'excluded_metadata_full_songs': 58,
    'split_counts': p.EXPECTED_SPLIT_COUNTS, 'identity_key': 'sample_id',
    'probe': 'Ridge', 'solver': 'cholesky', 'fit_intercept': True,
    'input_scaler': 'StandardScaler fit on Train only', 'target_standardization': False,
    'alpha_grid': list(p.ALPHA_GRID), 'selection': 'maximum Validation R2; exact tie larger alpha',
    'final_fit': 'train_only', 'baseline': 'constant Train target mean',
    'metrics': ['mae', 'r2', 'pearson_r'], 'uncertainty': 'point estimates only',
    'test_policy': 'all 26 selections verified and frozen before unified Test sweep; no post-Test tuning',
    'layer12': 'reuse authoritative Module C RQ1 Test metrics and ID-aligned predictions; not new untouched evidence',
    'interpretation': 'complete depth trajectory primary; numerical maxima secondary; no significance or universal best-layer claims',
    'figure': {'x': 'Pre-Transformer then Layer 1 through Layer 12', 'y': 'Test R2',
               'series': ['valence', 'arousal'], 'zero_reference': True,
               'style': 'unsmoothed connected point estimates; no error bars; identify reused Layer 12',
               'formats': ['png', 'svg']},
    'test_integrity_access': 'whole-cache loading and ID/finite checks only before gate; no Test fitting, selection, predictions or metrics',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identities():
    paths = {'canonical_cache': CACHE, 'labels': LABELS, 'fixed_split': SPLIT,
             'module_c_validation': C3, 'module_c_test': C4, 'module_c_test_predictions': C4_PRED,
             'probing_source': ROOT / 'src/mert_emotion_probing/probing.py',
             'runner_source': Path(__file__),
             'verifier_source': ROOT / 'scripts/verify_layerwise_probing_results.py',
             'serialization_source': ROOT / 'scripts/run_basic_probing.py'}
    return {k: file_identity(v) for k, v in paths.items()}


def dataset(level):
    return p.assemble_probing_dataset(CACHE, LABELS, SPLIT, representation_level=p.EXPECTED_LEVEL_NAMES[level])


def rows(data, level, target, alpha, pred, base, split):
    return pd.DataFrame({'sample_id': getattr(data, split + '_ids'), 'split': split,
                         'target': target, 'representation_level': p.EXPECTED_LEVEL_NAMES[level],
                         'representation_index': level, 'selected_alpha': alpha,
                         'y_true': getattr(data, 'y_' + split), 'prediction': pred,
                         'baseline_prediction': base})


def select():
    if any(path.exists() for path in (SELECTION, GATE, STARTED, TEST)):
        raise RuntimeError('Selection artifacts already exist; refusing silent overwrite or post-Test reselection')
    before = identities()
    c3 = read(C3)
    for name in ('canonical_cache', 'labels', 'fixed_split'):
        assert before[name]['sha256'] == c3['inputs'][name]['sha256']
    configurations, tables, checks = [], [], []
    for level in range(13):
        ds = dataset(level)
        checks.append(ds.verification)
        for target in p.TARGET_COLUMNS:
            data = p.train_validation_view(ds, target)
            result, pred, base = p.fit_select_validate(data)
            configurations.append({'level': level, 'representation_level': ds.representation_level,
                                   'target': target, **result})
            tables.append(rows(data, level, target, result['selected_alpha'], pred, base, 'validation'))
        print(f'Selected level {level}; Test gate closed', flush=True)
    assert before == identities()
    atomic_write_csv(VAL_PRED, pd.concat(tables, ignore_index=True))
    atomic_write_json(SELECTION, {
        'created_utc': utc_now(), 'protocol': PROTOCOL, 'inputs_and_code': before,
        'configurations': configurations, 'data_assembly_verification': checks,
        'validation_predictions': file_identity(VAL_PRED),
        'test_predictions_generated': False, 'test_metrics_computed': False,
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'sklearn': sklearn.__version__, 'torch': torch.__version__},
    })


def plot(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for target, color, marker in [('valence', '#2274A5', 'o'), ('arousal', '#C65D21', 's')]:
        data = table.loc[table.target == target].sort_values('level')
        ax.plot(data.level, data.test_r2, color=color, marker=marker, linewidth=2, label=target.title())
        ax.scatter([12], data.loc[data.level == 12, 'test_r2'], s=100, facecolors='white', edgecolors=color, zorder=4)
    ax.axhline(0, color='0.45', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(13), ['Pre-\nTransformer'] + [f'Layer {i}' for i in range(1, 13)], rotation=35, ha='right')
    ax.set(xlabel='Frozen MERT representation level', ylabel='Test R²',
           title='Linear emotion decodability across frozen MERT depth')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='lower right')
    fig.text(0.5, 0.025, 'Fixed DEAM split · Train-only Ridge · Point estimates only\nOpen markers at Layer 12: authoritative Module C Test values reused; no new untouched evidence.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for suffix in ('.png', '.svg'):
        fig.savefig(FIGURE.with_suffix(suffix), dpi=180)
    plt.close(fig)


def evaluate():
    from verify_layerwise_probing_results import verify_selection
    gate = read(GATE)
    assert gate['passed'] and gate['selection_sha256'] == file_identity(SELECTION)['sha256']
    assert gate['validation_predictions_sha256'] == file_identity(VAL_PRED)['sha256']
    selection = read(SELECTION)
    assert selection['protocol'] == PROTOCOL and selection['inputs_and_code'] == identities()
    verify_selection(repeat_fits=False)
    if STARTED.exists() or TEST.exists():
        raise RuntimeError('Unified Test sweep already started; correction requires a traceable invalidating-error record')
    c4 = read(C4)
    c4pred = pd.read_csv(C4_PRED, float_precision='round_trip')
    for name in ('canonical_cache', 'labels', 'fixed_split'):
        assert c4['provenance']['inputs'][name]['sha256'] == selection['inputs_and_code'][name]['sha256']
    atomic_write_json(STARTED, {'started_utc': utc_now(), 'gate': file_identity(GATE),
                              'selection': file_identity(SELECTION), 'protocol': PROTOCOL})
    results, tables, flat = [], [], []
    for level in range(13):
        ds = dataset(level)
        for target in p.TARGET_COLUMNS:
            selected = next(c for c in selection['configurations'] if c['level'] == level and c['target'] == target)
            alpha = selected['selected_alpha']
            data = p.train_test_view(ds, target)
            if level == 12:
                result = c4['targets'][target]
                assert alpha == result['frozen_alpha']
                saved = c4pred.loc[c4pred.target == target].set_index('sample_id', verify_integrity=True).loc[data.test_ids]
                np.testing.assert_array_equal(saved.y_true, data.y_test)
                assert set(saved.representation_index) == {12}
                assert set(saved.representation_level) == {p.PRIMARY_LEVEL_NAME}
                assert set(saved.frozen_alpha) == {alpha}
                pred, base = saved.prediction.to_numpy(), saved.baseline_prediction.to_numpy()
                np.testing.assert_array_equal(base, np.full(261, np.mean(data.y_train)))
                source = 'reused_authoritative_module_c_rq1'
            else:
                result, pred, base = p.fit_train_evaluate_test(data, alpha=alpha)
                source = 'module_d_unified_held_out_sweep'
            results.append({'level': level, 'target': target, 'source': source, **result})
            tables.append(rows(data, level, target, alpha, pred, base, 'test'))
            flat.append({'level': level, 'representation_level': ds.representation_level,
                         'target': target, 'selected_alpha': alpha, 'test_source': source,
                         **{'validation_' + k: selected['selected_validation_metrics'][k] for k in PROTOCOL['metrics']},
                         **{'test_' + k: result['test_metrics'][k] for k in PROTOCOL['metrics']},
                         **{'baseline_validation_' + k: selected['baseline']['validation_metrics'][k] for k in PROTOCOL['metrics']},
                         **{'baseline_test_' + k: result['baseline']['test_metrics'][k] for k in PROTOCOL['metrics']}})
        print(f'Held-out level {level} saved in memory' + (' (Module C reused)' if level == 12 else ''), flush=True)
    assert selection['inputs_and_code'] == identities()
    atomic_write_csv(TEST_PRED, pd.concat(tables, ignore_index=True))
    table = pd.DataFrame(flat)
    atomic_write_csv(TABLE, table)
    atomic_write_json(TEST, {'created_utc': utc_now(), 'protocol': PROTOCOL,
                            'selection': file_identity(SELECTION), 'gate': file_identity(GATE),
                            'test_started': file_identity(STARTED), 'configurations': results,
                            'predictions': file_identity(TEST_PRED), 'table': file_identity(TABLE),
                            'new_test_configurations': 24, 'reused_module_c_configurations': 2,
                            'post_test_tuning': False})
    plot(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['select', 'test'])
    args = parser.parse_args()
    (select if args.phase == 'select' else evaluate)()
