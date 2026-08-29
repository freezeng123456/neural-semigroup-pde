#!/usr/bin/env python3
"""Stable entry point for the exploratory Fisher mechanism evaluator.

The implementation lives in :mod:`evaluate_fisher_mechanism`; this short
wrapper keeps the descriptive ``*_diagnostics.py`` command name while making
the implementation easy to import in tests and on SCNet.
"""

from __future__ import annotations

try:
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
except ImportError:  # pragma: no cover - direct ``python path/to/script.py``
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
