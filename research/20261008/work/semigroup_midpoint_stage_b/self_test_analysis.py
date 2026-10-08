#!/usr/bin/env python3
"""Small synthetic tests of matrix arithmetic/gates/plotting; no model evidence."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('analyze_matrix.py')
    spec = importlib.util.spec_from_file_location('matrix_under_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checks = []
    def passed(name, **detail): checks.append({'name': name, 'passed': True, **detail})
    truth = np.zeros((2, 2, 1, 1, 3), dtype=np.float32)
    truth[0, 0, 0, 0] = [1, 1, 10]; truth[1, 0, 0, 0] = [1, 1, 2]
    pred = truth.copy(); pred[0, 0, 0, 0, 1:] += [1, 1]; pred[1, 0, 0, 0, 2] += 2
    result, raw = module.metrics_from_predictions(pred, truth)
    expected = np.array([math.sqrt(2 / 102), math.sqrt(4 / 6)])
    np.testing.assert_allclose(raw['joint_per_trajectory'], expected, rtol=1e-15)
    np.testing.assert_array_equal(raw['relative_l2_by_snapshot'], [[0, 1, .1], [0, 0, 1]])
    assert math.isclose(result['joint_mean'], expected.mean(), rel_tol=1e-15)
    passed('per_trajectory_joint_then_mean_with_initial_snapshot')
    assert not math.isclose(result['joint_mean'], raw['relative_l2_by_snapshot'].mean())
    passed('joint_error_is_not_mean_of_snapshot_ratios')
    large = np.full((1, 1, 1, 1, 2), np.float32(3e38), dtype=np.float32)
    large_result, _ = module.metrics_from_predictions(-large, large)
    assert large_result['valid'] and large_result['joint_mean'] == 2.0
    with np.errstate(over='ignore'): assert np.isinf((-large) - large).all()
    passed('cast_float32_states_before_subtraction_avoids_float32_overflow')
    invalid, _ = module.metrics_from_predictions(np.zeros_like(truth), np.zeros_like(truth))
    assert not invalid['valid'] and invalid['joint_mean'] is None and invalid['zero_joint_denominators'] == 2
    passed('zero_denominator_is_undefined_without_epsilon')
    bad = pred.copy(); bad[0, 0, 0, 0, 2] = np.nan
    invalid, _ = module.metrics_from_predictions(bad, truth)
    assert not invalid['valid'] and invalid['joint_mean'] is None and invalid['nonfinite_prediction_values'] == 1
    passed('nonfinite_trajectory_not_dropped')
    assert math.isclose(module.geometric_mean([1, 4, 16]), 4, rel_tol=1e-15)
    assert module.geometric_mean([0, 1, 2]) == 0 and module.ratio(0, 1) == 0
    assert module.ratio(0, 0) is None and module.geometric_mean([None, 1, 2]) is None
    passed('geometric_mean_and_zero_ratio_contract')
    def matrix():
        rows = []
        for arm in module.ARMS:
            for seed in module.SEEDS:
                error = {'gdsg500': 1., 'single500': 1.3, 'midpoint500': .9, 'gdsg333': .85}[arm]
                seconds = {'gdsg500': 1000., 'single500': 700., 'midpoint500': 700., 'gdsg333': 666.}[arm]
                rows.append({'arm': arm, 'seed': seed, 'status': module.SUCCESS,
                             'development_joint_mean': error, 'training_seconds': seconds,
                             'test_joint_mean': 2 * error, 'test_final_step_mean': 3 * error,
                             'whole_process_wall_seconds': 1.1 * seconds, 'cumulative_wall_seconds': 1.1 * seconds})
        return rows
    def edit(rows, arm, seed=None, **updates):
        for row in rows:
            if row['arm'] == arm and (seed is None or row['seed'] == seed): row.update(updates)
    rows = matrix(); comparisons, gates = module.paired_summaries(rows)
    assert gates['all_registered_gates_pass'] and set(comparisons) == set(module.CONTROLS)
    assert comparisons['gdsg333']['development_joint_mean_ratio_geometric_mean'] > 1
    passed('complete_three_seed_gates_and_unhidden_better_budget_control')
    boundary = matrix(); edit(boundary, 'midpoint500', development_joint_mean=1.1, training_seconds=800.)
    _, gate = module.paired_summaries(boundary)
    assert gate['practical_utility']['passed'] and gate['practical_utility']['error_ratio_gm'] == 1.1 and gate['practical_utility']['training_time_ratio_gm'] == .8
    passed('inclusive_practical_boundaries_1_10_and_0_80')
    boundary = matrix(); edit(boundary, 'single500', development_joint_mean=1.)
    _, gate = module.paired_summaries(boundary)
    assert gate['midpoint_mechanism']['passed'] and gate['midpoint_mechanism']['error_ratio_gm'] == .9
    passed('inclusive_mechanism_boundary_0_90')
    for change, expected_gate in [({'development_joint_mean': 1.10001}, 'practical_utility'), ({'training_seconds': 800.01}, 'practical_utility')]:
        case = matrix(); edit(case, 'midpoint500', **change)
        assert not module.paired_summaries(case)[1][expected_gate]['passed']
    case = matrix(); edit(case, 'single500', development_joint_mean=1.); edit(case, 'midpoint500', development_joint_mean=.90001)
    assert not module.paired_summaries(case)[1]['midpoint_mechanism']['passed']
    passed('just_above_each_registered_bound_rejected')
    case = matrix()
    for seed, value in enumerate([.3, 1.3, 1.3]): edit(case, 'midpoint500', seed, development_joint_mean=value)
    gate = module.paired_summaries(case)[1]['practical_utility']
    assert gate['error_ratio_gm'] < 1.1 and gate['seeds_with_error_ratio_at_most_1_10'] == 1 and not gate['passed']
    passed('low_gm_cannot_replace_two_of_three_practical_direction')
    case = matrix(); edit(case, 'single500', development_joint_mean=1.)
    for seed, value in enumerate([.3, 1.1, 1.1]): edit(case, 'midpoint500', seed, development_joint_mean=value)
    gate = module.paired_summaries(case)[1]['midpoint_mechanism']
    assert gate['error_ratio_gm'] < .9 and gate['seeds_with_strictly_lower_error'] == 1 and not gate['passed']
    passed('low_gm_cannot_replace_two_of_three_mechanism_direction')
    case = matrix()
    for seed, value in enumerate([100., 1400., 1400.]): edit(case, 'midpoint500', seed, training_seconds=value)
    gate = module.paired_summaries(case)[1]['practical_utility']
    assert gate['passed'] and gate['training_time_ratio_gm'] < .8
    assert np.mean([.1, 1.4, 1.4]) > .8
    passed('time_gate_uses_paired_three_seed_geometric_mean')
    for status in ('TRAINING_FAILED', 'MISSING_TRAINING', 'EVALUATION_FAILED', 'NONFINITE_OR_UNDEFINED_TEST'):
        case = matrix(); edit(case, 'midpoint500', 1, status=status)
        comp, gate = module.paired_summaries(case)
        assert not gate['all_registered_gates_pass'] and not gate['practical_utility']['passed'] and not gate['midpoint_mechanism']['passed']
        assert len(comp['gdsg500']['pairs']) == 3 and comp['gdsg500']['development_joint_mean_ratio_geometric_mean'] is None
        assert comp['gdsg500']['pairs'][1]['development_joint_mean_ratio'] is None
    passed('candidate_failed_missing_nonfinite_seed_never_omitted_for_success')
    case = matrix(); edit(case, 'single500', 0, status='MISSING_EVALUATION')
    _, gate = module.paired_summaries(case)
    assert gate['practical_utility']['passed'] and not gate['midpoint_mechanism']['passed'] and not gate['all_registered_gates_pass']
    passed('separate_utility_mechanism_gates_when_one_control_missing')
    case = matrix(); edit(case, 'gdsg333', 0, status='TRAINING_FAILED')
    assert not module.paired_summaries(case)[1]['all_registered_gates_pass']
    passed('missing_budget_control_prevents_complete_matrix_success')
    for invalid in (matrix()[:-1], matrix()[:-1] + [copy.deepcopy(matrix()[0])]):
        try: module.paired_summaries(invalid)
        except ValueError: pass
        else: raise AssertionError('Incomplete/duplicate matrix was accepted')
    passed('fixed_twelve_unique_identities_required')
    case = matrix(); edit(case, 'midpoint500', test_joint_mean=1e20, test_final_step_mean=1e20)
    assert module.paired_summaries(case)[1]['all_registered_gates_pass']
    passed('finite_exploratory_test_does_not_change_registered_development_thresholds')
    partial = out / 'synthetic_failed_run_trace'; partial.mkdir()
    records = [{'update': 1, 'forward_calls': 6, 'training_seconds': 1.0},
               {'update': 2, 'forward_calls': 6, 'training_seconds': 1.5}]
    (partial / 'updates.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n{"update":')
    costs = module.partial_resources(partial, {'updates_completed': 3, 'whole_process_wall_seconds': 9.0})
    assert costs['logged_updates'] == 2 and costs['actual_updates'] == 3
    assert costs['training_forward_calls'] == 12 and costs['training_seconds'] == 2.5
    assert costs['partial_trace_truncated_or_discontinuous'] is True and costs['whole_process_wall_seconds'] == 9.0
    assert 'lower bounds' in costs['resource_evidence_scope'] and 'cuda_peak_allocated_bytes' not in costs
    passed('failed_run_partial_costs_retained_as_lower_bounds_without_invented_gpu_peak')
    for number, invalid_record in enumerate([{'update': 2, 'training_seconds': 5.0},
                                           {'update': 2, 'training_seconds': 5.0, 'forward_calls': 6.5},
                                           {'update': 2, 'training_seconds': 5.0, 'forward_calls': 0}]):
        folder = out / ('synthetic_invalid_calls_' + str(number)); folder.mkdir()
        (folder / 'updates.jsonl').write_text(json.dumps(records[0]) + '\n' + json.dumps(invalid_record) + '\n')
        invalid_costs = module.partial_resources(folder)
        assert invalid_costs['logged_updates'] == 1 and invalid_costs['training_forward_calls'] == 6 and invalid_costs['training_seconds'] == 1.0
        assert invalid_costs['partial_trace_truncated_or_discontinuous']
    passed('incomplete_json_records_cannot_increment_logged_updates_or_cost')
    plot_raw = {(row['arm'], row['seed']): {'test': {'relative_l2_by_snapshot': np.tile(np.linspace(.0001, row['test_joint_mean'], 100), (2, 1))}} for row in rows}
    module.make_figures(out, rows, plot_raw, comparisons, fixture=True)
    assert (out / 'matrix_overview.png').stat().st_size > 1000
    assert (out / 'matrix_overview.pdf').read_bytes().startswith(b'%PDF')
    passed('scientific_png_pdf_render_explicitly_labelled_synthetic_fixture')
    report = {'status': 'SYNTHETIC_MATRIX_LOGIC_TESTS_PASS', 'checks': checks, 'checks_passed': len(checks),
              'analysis_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'training_runs_loaded': 0, 'real_predictions_loaded': 0, 'network_calls': 0, 'optimizer_updates': 0,
              'scope': 'Synthetic arithmetic/control/plot fixtures only; no Stage B experiment result or conclusion.'}
    module.write_json(out / 'SYNTHETIC_SELF_TEST.json', report)
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks), 'output': str(out)}, indent=2))


if __name__ == '__main__': main()
