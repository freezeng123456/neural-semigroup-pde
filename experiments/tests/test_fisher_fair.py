import pytest

from experiments.run_fisher_fair import build_parser, rollout_steps_for_horizon


def test_common_variable_tau_horizon_is_exactly_aligned():
    assert [
        rollout_steps_for_horizon(1.2, tau)
        for tau in (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    ] == [48, 24, 16, 12, 8, 6]
    with pytest.raises(ValueError, match="not an integer multiple"):
        rollout_steps_for_horizon(1.2, 0.25)


def test_parser_has_parameter_matched_defaults():
    args = build_parser().parse_args(
        [
            "--regime",
            "variable",
            "--output-dir",
            "out",
            "--data-cache",
            "data.pt",
        ]
    )
    assert args.train_taus == (0.025, 0.05, 0.1, 0.2)
    assert args.eval_taus == (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    assert args.validation_interval == 5
