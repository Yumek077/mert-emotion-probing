"""Verify Module D selection before Test, or final saved artifacts without refitting Test."""
from __future__ import annotations

import argparse
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_layerwise_probing as d
from run_basic_probing import atomic_write_json, file_identity, utc_now
from mert_emotion_probing import probing as p


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def metrics(y, pred, saved):
    # Independent formulas, rather than calling the production metric helper.
    y, pred = np.asarray(y), np.asarray(pred)
    close(np.mean(np.abs(y - pred)), saved['mae'])
    close(1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2), saved['r2'])
    if np.all(pred == pred[0]):
        assert saved['pearson_r'] is None
        assert saved['pearson_r_status'] == 'undefined_constant_prediction'
    else:
        close(np.corrcoef(y, pred)[0, 1], saved['pearson_r'])


def verify_predictions(table, configurations, split):
    mapping = pd.read_csv(d.LABELS).set_index('song_id', verify_integrity=True)
    roles = pd.read_csv(d.SPLIT).set_index('sample_id', verify_integrity=True)
    expected_ids = set(roles.index[roles.split == split])
    train_ids = roles.index[roles.split == 'train']
    assert len(table) == 26 * p.EXPECTED_SPLIT_COUNTS[split]
    assert set(table.split) == {split}
    assert not table.duplicated(['sample_id', 'target', 'representation_index']).any()
    assert set(zip(table.representation_index, table.target)) == {(l, t) for l in range(13) for t in p.TARGET_COLUMNS}
    for config in configurations:
        level, target = config['level'], config['target']
        selected = table.loc[(table.representation_index == level) & (table.target == target)]
        assert set(selected.sample_id) == expected_ids and len(selected) == len(expected_ids)
        assert set(selected.representation_level) == {p.EXPECTED_LEVEL_NAMES[level]}
        alpha = config['selected_alpha'] if split == 'validation' else config['frozen_alpha']
        assert set(selected.selected_alpha) == {alpha}
        true = mapping.loc[selected.sample_id, p.TARGET_COLUMNS[target]].to_numpy()
        np.testing.assert_array_equal(selected.y_true, true)
        metric_key = 'selected_validation_metrics' if split == 'validation' else 'test_metrics'
        metrics(true, selected.prediction, config[metric_key])
        mean = float(mapping.loc[train_ids, p.TARGET_COLUMNS[target]].mean())
        close(mean, config['baseline']['train_target_mean'])
        close(selected.baseline_prediction, mean)
        assert selected.baseline_prediction.nunique() == 1
        metrics(true, selected.baseline_prediction, config['baseline'][split + '_metrics'])
        assert config['standardization']['fit_split'] == 'train'
        assert config['standardization']['fit_sample_count'] == 1221
        assert config['standardization']['target_standardized'] is False
        assert config['standardization']['train_mean_verified']
        assert config['standardization']['train_variance_verified']
        assert config['ridge']['fit_split'] == 'train' and config['ridge']['solver'] == 'cholesky'


def verify_selection(repeat_fits=True):
    saved = d.read(d.SELECTION)
    assert saved['protocol'] == d.PROTOCOL
    assert saved['inputs_and_code'] == d.identities()
    assert not saved['test_predictions_generated'] and not saved['test_metrics_computed']
    assert saved['validation_predictions'] == file_identity(d.VAL_PRED)
    configs = saved['configurations']
    assert len(configs) == 26
    assert {(c['level'], c['target']) for c in configs} == {(l, t) for l in range(13) for t in p.TARGET_COLUMNS}
    predictions = pd.read_csv(d.VAL_PRED, float_precision='round_trip')
    verify_predictions(predictions, configs, 'validation')
    for c in configs:
        candidates = c['candidate_validation_r2']
        assert [r['alpha'] for r in candidates] == list(p.ALPHA_GRID)
        assert c['selected_alpha'] == max(candidates, key=lambda row: (row['validation_r2'], row['alpha']))['alpha']
        assert c['best_validation_r2'] == max(r['validation_r2'] for r in candidates)
        assert c['selected_validation_metrics']['r2'] == c['best_validation_r2']
        assert c['representation_level'] == p.EXPECTED_LEVEL_NAMES[c['level']]
    for score, expected in [(0.5, 10), (np.nextafter(0.5, -np.inf), 1)]:
        selected, _, _ = p.select_candidate_by_validation_r2([
            {'alpha': 1., 'validation_r2': 0.5}, {'alpha': 10., 'validation_r2': score}])
        assert selected['alpha'] == expected
    c3 = d.read(d.C3)
    for c in configs[-2:]:
        assert c['level'] == 12
        original = c3['targets'][c['target']]
        assert c['selected_alpha'] == original['selected_alpha']
        for actual, expected in zip(c['candidate_validation_r2'], original['candidate_validation_r2']):
            assert actual['alpha'] == expected['alpha']
            close(actual['validation_r2'], expected['validation_r2'])
        for key in d.PROTOCOL['metrics']:
            close(c['selected_validation_metrics'][key], original['selected_validation_metrics'][key])
    if repeat_fits:
        raw = torch.load(d.CACHE, map_location='cpu', weights_only=False)
        # Deliberately reversed cache identities: no assumption that cache rows are sorted.
        reverse_ids = raw['sample_ids'].numpy()[::-1]
        cache_lookup = {int(sid): i for i, sid in enumerate(reverse_ids)}
        reversed_features = raw['representations'].numpy()[::-1]
        with patch.object(p, 'train_test_view', side_effect=AssertionError('Test view forbidden before gate')), \
             patch.object(p, 'fit_train_evaluate_test', side_effect=AssertionError('Test evaluation forbidden before gate')):
            for level in range(13):
                ds = d.dataset(level)
                assert ds.representation_index == level
                assert ds.verification['split_counts'] == p.EXPECTED_SPLIT_COUNTS
                assert ds.verification['excluded_full_song_rows'] == 58
                expected = reversed_features[[cache_lookup[int(sid)] for sid in ds.sample_ids], level, :]
                np.testing.assert_array_equal(ds.features, expected)
                # Poison all Test values after integrity checks. Selection must remain identical.
                poisoned_x = ds.features.copy()
                poisoned_x[ds.split == 'test'] = np.nan
                poisoned_y = {t: v.copy() for t, v in ds.targets.items()}
                for values in poisoned_y.values():
                    values[ds.split == 'test'] = np.nan
                poisoned = replace(ds, features=poisoned_x, targets=poisoned_y)
                for target in p.TARGET_COLUMNS:
                    data = p.train_validation_view(poisoned, target)
                    result, pred, _ = p.fit_select_validate(data)
                    c = next(c for c in configs if c['level'] == level and c['target'] == target)
                    assert result == {k: v for k, v in c.items() if k not in ('level', 'representation_level', 'target')}
                    rows = predictions.loc[(predictions.representation_index == level) & (predictions.target == target)]
                    rows = rows.set_index('sample_id', verify_integrity=True).loc[data.validation_ids]
                    np.testing.assert_array_equal(pred, rows.prediction)
                print(f'Verified level {level}: alignment, deterministic repeat, poisoned-Test isolation', flush=True)
        # Exercise actual assembly with reversed cache, labels and split rows using existing loaders.
        original_cache, original_labels, original_split = p._load_cache, p._load_labels, p._load_split
        def reverse_cache(*args, **kwargs):
            ids, features, index, checks = original_cache(*args, **kwargs)
            return ids[::-1], features[::-1], index, checks
        def reverse_labels(*args):
            table, checks = original_labels(*args)
            return table.iloc[::-1], checks
        def reverse_split(*args):
            table, checks = original_split(*args)
            return table.iloc[::-1], checks
        regular = d.dataset(0)
        with patch.object(p, '_load_cache', side_effect=reverse_cache), \
             patch.object(p, '_load_labels', side_effect=reverse_labels), \
             patch.object(p, '_load_split', side_effect=reverse_split):
            reversed_ds = d.dataset(0)
        np.testing.assert_array_equal(regular.features, reversed_ds.features)
        np.testing.assert_array_equal(regular.split, reversed_ds.split)
        for target in p.TARGET_COLUMNS:
            np.testing.assert_array_equal(regular.targets[target], reversed_ds.targets[target])
    return {'passed': True, 'created_utc': utc_now(), 'configurations': 26,
            'selection_sha256': file_identity(d.SELECTION)['sha256'],
            'validation_predictions_sha256': file_identity(d.VAL_PRED)['sha256'],
            'validation_prediction_rows': len(predictions),
            'metrics_independently_recomputed': True, 'baseline_recomputed_from_train_labels': True,
            'layer12_module_c_validation_compatible_atol': 1e-12,
            'all_levels_raw_cache_alignment_verified': repeat_fits,
            'reversed_source_rows_invariance_verified': repeat_fits,
            'all_26_deterministic_repeats_with_poisoned_test_values': repeat_fits,
            'exact_and_near_tie_verified': True,
            'new_test_predictions_or_metrics_computed': False,
            'test_integrity_loading_only_before_gate': True}


def verify_final():
    selection = d.read(d.SELECTION)
    test = d.read(d.TEST)
    assert selection['inputs_and_code'] == d.identities()
    assert test['protocol'] == d.PROTOCOL and test['post_test_tuning'] is False
    for key, path in [('selection', d.SELECTION), ('gate', d.GATE), ('test_started', d.STARTED),
                      ('predictions', d.TEST_PRED), ('table', d.TABLE)]:
        assert test[key] == file_identity(path)
    assert d.read(d.GATE)['passed']
    assert selection['created_utc'] < d.read(d.GATE)['created_utc'] < d.read(d.STARTED)['started_utc'] < test['created_utc']
    assert len(test['configurations']) == 26
    assert {(c['level'], c['target']) for c in test['configurations']} == {(l, t) for l in range(13) for t in p.TARGET_COLUMNS}
    predictions = pd.read_csv(d.TEST_PRED, float_precision='round_trip')
    verify_predictions(predictions, test['configurations'], 'test')
    c4, c4pred = d.read(d.C4), pd.read_csv(d.C4_PRED, float_precision='round_trip')
    flat = pd.read_csv(d.TABLE, float_precision='round_trip')
    assert len(flat) == 26 and not flat.duplicated(['level', 'target']).any()
    for c in test['configurations']:
        selected = next(s for s in selection['configurations'] if s['level'] == c['level'] and s['target'] == c['target'])
        assert c['frozen_alpha'] == selected['selected_alpha']
        assert c['standardization']['validation_in_final_refit'] is False
        assert c['ridge']['fit_sample_count'] == 1221
        row = flat.loc[(flat.level == c['level']) & (flat.target == c['target'])].iloc[0]
        assert row.selected_alpha == selected['selected_alpha'] and row.test_source == c['source']
        for prefix, values in [('validation_', selected['selected_validation_metrics']),
                               ('test_', c['test_metrics']),
                               ('baseline_validation_', selected['baseline']['validation_metrics']),
                               ('baseline_test_', c['baseline']['test_metrics'])]:
            for metric in d.PROTOCOL['metrics']:
                assert (pd.isna(row[prefix + metric]) if values[metric] is None else np.isclose(row[prefix + metric], values[metric], rtol=0, atol=1e-12))
        if c['level'] == 12:
            assert c['source'] == 'reused_authoritative_module_c_rq1'
            assert c['test_metrics'] == c4['targets'][c['target']]['test_metrics']
            assert c['baseline'] == c4['targets'][c['target']]['baseline']
            actual = predictions.loc[(predictions.representation_index == 12) & (predictions.target == c['target'])].set_index('sample_id')
            original = c4pred.loc[c4pred.target == c['target']].set_index('sample_id').loc[actual.index]
            np.testing.assert_array_equal(actual[['y_true', 'prediction', 'baseline_prediction']], original[['y_true', 'prediction', 'baseline_prediction']])
        else:
            assert c['source'] == 'module_d_unified_held_out_sweep'
    for suffix in ('.png', '.svg'):
        assert d.FIGURE.with_suffix(suffix).stat().st_size > 1000
    from PIL import Image
    with Image.open(d.FIGURE) as figure:
        figure.verify()
    summary = {'passed': True, 'created_utc': utc_now(), 'configurations': 26,
               'new_test_configurations': 24, 'reused_layer12_configurations': 2,
               'test_prediction_rows': len(predictions), 'unique_test_ids': predictions.sample_id.nunique(),
               'independent_metric_formulas_verified': True, 'target_mapping_verified': True,
               'frozen_alpha_and_train_only_fit_verified': True, 'baseline_train_means_verified': True,
               'layer12_metrics_and_predictions_exactly_equal_to_module_c': True,
               'gate_chronology_and_hashes_verified': True, 'flat_table_verified': True,
               'figure_files_verified': [file_identity(d.FIGURE.with_suffix(s)) for s in ('.png', '.svg')],
               'post_test_tuning': False}
    atomic_write_json(d.FINAL_VERIFY, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['selection', 'final'])
    args = parser.parse_args()
    if args.phase == 'selection':
        if d.GATE.exists() or d.STARTED.exists() or d.TEST.exists():
            raise RuntimeError('Cannot overwrite a freeze gate or recreate it after Test')
        summary = verify_selection()
        atomic_write_json(d.GATE, summary)
    else:
        summary = verify_final()
    print(summary)
