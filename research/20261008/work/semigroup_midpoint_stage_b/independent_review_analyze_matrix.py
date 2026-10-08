#!/usr/bin/env python3
"""Independent arithmetic/control fixtures. No model loading or real run reads."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'analyze_matrix.py'
spec = importlib.util.spec_from_file_location('reviewed_matrix_analyzer', TARGET)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append({'name': name, 'passed': True})


def matrix(error=(1., 1., 1.), time=(.75, .75, .75), mechanism=(.85, .85, .85)):
    rows = []
    for arm in m.ARMS:
        for seed in m.SEEDS:
            value = {'gdsg500': 1., 'single500': error[seed] / mechanism[seed],
                     'midpoint500': error[seed], 'gdsg333': 1.3}[arm]
            seconds = time[seed] * 100 if arm == 'midpoint500' else 100.
            rows.append({'arm': arm, 'seed': seed, 'status': m.SUCCESS,
                         'development_joint_mean': value, 'training_seconds': seconds,
                         'test_joint_mean': value * 2, 'test_final_step_mean': value * 3,
                         'whole_process_wall_seconds': seconds + 10,
                         'cumulative_wall_seconds': seconds + 20})
    return rows


def gates(rows):
    return m.paired_summaries(rows)[1]


base = matrix()
check('nominal_all_registered_gates', gates(base)['all_registered_gates_pass'])
check('row_order_invariant', m.paired_summaries(base) == m.paired_summaries(base[::-1]))
for label, error, seconds, key in [
    ('utility_error_at_inclusive_boundary', (1.1,) * 3, (.8,) * 3, True),
    ('utility_error_one_ulp_above', (math.nextafter(1.1, math.inf),) * 3, (.8,) * 3, False),
    ('utility_time_one_ulp_above', (1.,) * 3, (math.nextafter(.8, math.inf),) * 3, False),
    ('utility_two_directions_and_gm', (.9, .9, 1.3), (.75,) * 3, True),
    ('utility_one_direction_despite_good_gm', (.1, 1.2, 1.2), (.75,) * 3, False),
]:
    check(label, gates(matrix(error, seconds))['practical_utility']['passed'] is key)

for name, value, expected in [
    ('mechanism_at_inclusive_boundary', (.9,) * 3, True),
    ('mechanism_one_ulp_above', (math.nextafter(.9, math.inf),) * 3, False),
    ('mechanism_two_strict_wins', (.8, .8, 1.), True),
    ('mechanism_one_strict_win_despite_good_gm', (.5, 1., 1.), False),
    ('mechanism_ties_are_not_strict_wins', (1., 1., 1.), False),
]:
    # Set candidate error directly to avoid generating a threshold by a rounded reciprocal.
    rows = matrix()
    for row in rows:
        if row['arm'] == 'midpoint500': row['development_joint_mean'] = value[row['seed']]
        if row['arm'] == 'single500': row['development_joint_mean'] = 1.
    check(name, gates(rows)['midpoint_mechanism']['passed'] is expected)

check('utility_and_mechanism_are_separate',
      gates(matrix(mechanism=(1.,) * 3))['practical_utility']['passed'] and
      not gates(matrix(mechanism=(1.,) * 3))['midpoint_mechanism']['passed'])

failure_states = ('MISSING_TRAINING', 'INCOMPLETE_TRAINING', 'TRAINING_FAILED',
                  'NONFINITE_OR_UNDEFINED_DEVELOPMENT', 'MISSING_EVALUATION',
                  'INCOMPLETE_EVALUATION', 'EVALUATION_FAILED', 'NONFINITE_OR_UNDEFINED_TEST',
                  'NONFINITE_FINAL_DIAGNOSTIC', 'INVALID_EVIDENCE', 'PAIRED_INPUT_INTEGRITY_FAILED')
for status in failure_states:
    for seed in m.SEEDS:
        rows = matrix()
        next(r for r in rows if r['arm'] == 'midpoint500' and r['seed'] == seed)['status'] = status
        comparisons, result = m.paired_summaries(rows)
        check(f'{status}_seed{seed}_not_omitted', not result['all_registered_gates_pass'] and
              not result['practical_utility']['passed'] and not result['midpoint_mechanism']['passed'] and
              all(c['development_joint_mean_ratio_geometric_mean'] is None for c in comparisons.values()) and
              all(len(c['pairs']) == 3 for c in comparisons.values()))
for control in m.CONTROLS:
    rows = matrix()
    next(r for r in rows if r['arm'] == control and r['seed'] == 2)['status'] = 'TRAINING_FAILED'
    comparisons, result = m.paired_summaries(rows)
    check(control + '_failure_retained', not result['all_registered_gates_pass'] and
          comparisons[control]['development_joint_mean_ratio_geometric_mean'] is None)
    if control == 'gdsg333':
        check('gdsg333_failure_does_not_change_two_other_comparisons',
              result['practical_utility']['passed'] and result['midpoint_mechanism']['passed'])

for bad in (base[:-1], base[:-1] + [base[0]], base + [base[0]]):
    try:
        m.paired_summaries(bad)
    except ValueError:
        checks.append({'name': 'invalid_12_cell_identity_' + str(len(checks)), 'passed': True})
    else:
        raise AssertionError('Incomplete/duplicate identities accepted')
for values, expected in [([.55, 1.1, 2.2], 1.1), ([.45, .9, 1.8], .9),
                         ([.4, .8, 1.6], .8), ([1e-300, 1e300, 1.], 1.), ([0., 1., 2.], 0.)]:
    check('geometric_mean_' + repr(values), m.geometric_mean(values) == expected)
for values in ([], [None, 1., 1.], [math.nan, 1., 1.], [math.inf, 1., 1.], [-1., 1., 1.]):
    check('undefined_gm_' + repr(values), m.geometric_mean(values) is None)
check('zero_numerator_allowed', m.ratio(0., 1.) == 0.)
check('zero_denominator_not_regularized', m.ratio(1., 0.) is None and m.ratio(0., 0.) is None)
rows = matrix()
next(r for r in rows if r['arm'] == 'gdsg500' and r['seed'] == 0)['development_joint_mean'] = 0.
comparisons, result = m.paired_summaries(rows)
check('undefined_comparator_blocks_utility', not result['practical_utility']['passed'] and
      comparisons['gdsg500']['development_joint_mean_ratio_geometric_mean'] is None)

truth = np.array([1, 10, 1, 1], dtype=np.float32).reshape(2, 1, 1, 1, 2)
prediction = truth + np.array([0, 1, 1, 0], dtype=np.float32).reshape(truth.shape)
summary, raw = m.metrics_from_predictions(prediction, truth)
expected = np.array([1 / math.sqrt(101), 1 / math.sqrt(2)])
check('joint_mean_is_mean_of_ratios', np.allclose(raw['joint_per_trajectory'], expected, rtol=1e-15) and
      summary['joint_mean'] == expected.mean())
check('per_time_reduction', np.array_equal(raw['relative_l2_by_snapshot'], [[0., .1], [1., 0.]]))
check('joint_differs_from_mean_time_ratio', summary['joint_mean'] != raw['relative_l2_by_snapshot'].mean())
check('initial_snapshot_in_joint_denominator', raw['joint_per_trajectory'][0] != .1)
for order in ('C', 'F'):
    s, r = m.metrics_from_predictions(np.array(prediction, order=order), np.array(truth, order=order))
    check('memory_layout_' + order, s['joint_mean'] == summary['joint_mean'] and
          np.array_equal(r['relative_l2_by_snapshot'], raw['relative_l2_by_snapshot']))
zero = truth.copy(); zero[0] = 0
summary_zero, raw_zero = m.metrics_from_predictions(prediction, zero)
check('zero_truth_retains_undefined_not_dropped', not summary_zero['valid'] and
      summary_zero['joint_mean'] is None and math.isnan(raw_zero['joint_per_trajectory'][0]) and
      summary_zero['zero_joint_denominators'] == 1 and summary_zero['zero_spatial_denominators'] == 2)
zero_time = truth.copy(); zero_time[0, ..., 0] = 0
summary_zero, raw_zero = m.metrics_from_predictions(prediction, zero_time)
check('single_zero_time_not_hidden_by_joint_norm', not summary_zero['valid'] and
      math.isfinite(raw_zero['joint_per_trajectory'][0]) and math.isnan(raw_zero['relative_l2_by_snapshot'][0, 0]))
for value in (math.nan, math.inf):
    bad = prediction.copy(); bad.flat[-1] = value
    with np.errstate(invalid='ignore'):
        s, r = m.metrics_from_predictions(bad, truth)
    check('nonfinite_prediction_' + repr(value), not s['valid'] and s['joint_mean'] is None and
          s['nonfinite_prediction_values'] == 1 and len(r['joint_per_trajectory']) == 2)
large_truth = np.array([1e-30, 1e-30], dtype=np.float32).reshape(1, 1, 1, 1, 2)
large_prediction = np.array([1e30, 1e30], dtype=np.float32).reshape(large_truth.shape)
s, r = m.metrics_from_predictions(large_prediction, large_truth)
check('finite_float32_states_use_float64_norms', s['valid'] and s['joint_mean'] > 1e59)
try:
    m.metrics_from_predictions(prediction.astype(np.float64), truth)
except ValueError:
    check('wrong_state_dtype_rejected', True)
else:
    raise AssertionError('Float64 input accepted')

audits = {}
for row in base:
    seed, arm = row['seed'], row['arm']
    count = 2 if arm == 'gdsg333' else 3
    audits[(arm, seed)] = {'input_by_name': {'train.py': 'frozen'}, 'initial_rng': {'shuffle': str(seed)},
        'collocations': {'u': str(seed)}, 'environment': {'initial_parameter_sha256': str(seed)},
        'trace': {'indices': ['i' + str(seed) + str(i) for i in range(count)], 'coins': ['c' + str(i) for i in range(count)]},
        'history': [{'permutation_sha256': str(i)} for i in range(count)],
        'diagnostic_common_inputs': {'z': 'common'}}
check('matching_pairs_and_short_budget_prefix', m.pairing_audit(base, audits)['passed'])
for name in ('input_by_name', 'initial_rng', 'collocations', 'environment', 'indices', 'coins', 'history', 'diagnostic_common_inputs'):
    changed = copy.deepcopy(audits)
    audit = changed[('midpoint500', 1)]
    if name in ('indices', 'coins'):
        audit['trace'][name][0] = 'altered'
    elif name == 'history':
        audit[name][0]['permutation_sha256'] = 'altered'
    elif name == 'environment':
        audit[name]['initial_parameter_sha256'] = 'altered'
    else:
        audit[name] = {'changed': True}
    check('pairing_detects_' + name, not m.pairing_audit(base, changed)['passed'])

report = {'status': 'PASS', 'scope': 'Pure synthetic arithmetic, failure handling and pairing checks; no real predictions, checkpoints, network calls or training.',
          'checks': checks, 'check_count': len(checks), 'analyzer_sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
          'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'network_calls': 0, 'optimizer_updates': 0, 'real_model_loads': 0}
destination = HERE / 'analyze_matrix_independent_self_test.json'
with destination.open('x') as f:
    json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps({'status': report['status'], 'check_count': len(checks), 'analyzer_sha256': report['analyzer_sha256'], 'report': str(destination)}))
