"""Regression coverage for the Allen--Cahn runner's canonical output contract."""

import csv
import json
import math
import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_allen_cahn_fair as runner  # noqa: E402
import verify_allen_cahn_artifacts as artifact_verifier  # noqa: E402


def _tiny_runner_argv(tmp_path):
    return [
        "--regime",
        "fixed",
        "--models",
        "latent_periodic_decoded_interaction",
        "--output-dir",
        str(tmp_path / "out"),
        "--data-cache",
        str(tmp_path / "allen_cahn_data.pt"),
        "--device",
        "cpu",
        "--N",
        "8",
        "--L",
        str(2.0 * math.pi),
        "--reference-dt",
        "0.01",
        "--fixed-tau",
        "0.02",
        "--eval-horizon",
        "0.04",
        "--n-train",
        "1",
        "--n-val",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--beta-v-floor",
        "0",
        "--deterministic",
        "--no-resume",
    ]


def test_allen_cahn_runner_writes_and_verifies_canonical_artifacts(tmp_path):
    """A successful run supplies the portable artifacts expected by launchers."""
    argv = _tiny_runner_argv(tmp_path)
    args = runner.build_parser().parse_args(argv)

    # The production runner intentionally requires a frozen cache for the
    # training stage.  Prepare the tiny cache through its existing preparation
    # API; this does not train a model or create a checkpoint.
    runner.load_or_generate_data(args, allow_generate=True)
    runner.main(argv)

    output_dir = Path(args.output_dir)
    model_name = "latent_periodic_decoded_interaction"
    required_artifacts = (
        output_dir / "summary.json",
        output_dir / "provenance.json",
        output_dir / "metrics.csv",
        output_dir / model_name / "checkpoints" / f"{model_name}_best.pt",
        output_dir / model_name / "checkpoints" / f"{model_name}_final.pt",
        output_dir / model_name / "checkpoints" / f"{model_name}_history.pt",
        output_dir / model_name / "result.json",
    )

    missing = [str(path.relative_to(output_dir)) for path in required_artifacts if not path.is_file()]
    assert not missing, f"Allen--Cahn runner is missing required artifacts: {missing}"

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (output_dir / model_name / "result.json").read_text(encoding="utf-8")
    )

    assert summary["experiment"] == "allen_cahn_parameter_matched_fair"
    assert summary["results"][model_name]["model"] == model_name
    assert provenance["experiment"] == summary["experiment"]
    assert result["model"] == model_name

    with (output_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["model"] == model_name
    assert rows[0]["tau"] == "0.02"
    assert float(rows[0]["rollout_mse_mean"]) >= 0.0

    receipt = artifact_verifier.validate_artifacts(output_dir, [model_name])
    assert receipt["status"] == "passed"
    assert receipt["metrics_rows"] == 1
