#!/usr/bin/env python3
"""Stable entry point for the exploratory Fisher mechanism evaluator.

The implementation lives in :mod:`evaluate_fisher_mechanism`; this short
wrapper keeps the descriptive ``*_diagnostics.py`` command name while making
the implementation easy to import in tests and on SCNet.
"""

from __future__ import annotations

if __package__:
    from .evaluate_fisher_mechanism import (  # noqa: F401
        EVALUATION_MODE,
        EXPLORATORY_SCHEMA_VERSION,
        EXPLORATORY_TRACK,
        build_parser,
        checkpoint_specs_from_args,
        evaluate_diagnostics,
        main,
        normalize_diagnostic_args,
        validate_args,
    )
else:  # pragma: no cover - direct ``python path/to/script.py``
    # The torch 1.12 compatibility launcher executes this file with
    # ``runpy.run_path``.  In that mode Python does not add the target script's
    # directory to sys.path, so the sibling implementation would otherwise be
    # invisible when the launcher is invoked from a different working
    # directory (as on SCNet).
    import sys
    from pathlib import Path

    _SCRIPT_DIR = str(Path(__file__).resolve().parent)
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from evaluate_fisher_mechanism import (  # noqa: F401
        EVALUATION_MODE,
        EXPLORATORY_SCHEMA_VERSION,
        EXPLORATORY_TRACK,
        build_parser,
        checkpoint_specs_from_args,
        evaluate_diagnostics,
        main,
        normalize_diagnostic_args,
        validate_args,
    )


if __name__ == "__main__":
    main()
