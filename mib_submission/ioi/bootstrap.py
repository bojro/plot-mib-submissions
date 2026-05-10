"""IOI linear-parameter bootstrap.

The IOI causal model's ``logit_diff`` mechanism reads three per-model
scalars — ``{bias, token_coeff, position_coeff}`` — that decompose the
output logit difference into bias + token-signal + position-signal terms.
These can't be derived analytically; the harness fits them by linear
regression on actual model logit-diffs across four counterfactual splits
(`same`, `s1_io_flip`, `s2_io_flip`, `s1_ioi_flip_s2_ioi_flip`).

We delegate the actual fitting to the harness's
`baselines/ioi_baselines/ioi_learn_linear_params.py` script (one-shot, no
PLOT logic) and expose helpers for:

- ``bootstrap_linear_params(model_short)``: run the harness script,
  capture its JSON output to ``submissions/plot/ioi_linear_params.json``,
  return the parsed dict for that model.
- ``load_linear_params(json_path, model_short)``: read the JSON and
  return the ``{bias, token_coeff, position_coeff}`` dict for the
  requested model.

Output JSON shape (matches the harness script — keyed by short model
name, with ``model_class`` injected at the top level for the harness
eval to pick up):

    {
      "gemma": {"bias": 0.05, "token_coeff": 0.77, "position_coeff": 2.00,
                "score": 0.95, "model_name": "google/gemma-2-2b"},
      "model_class": "Gemma2ForCausalLM"
    }
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..pipeline import REPO_ROOT, MIB_TRACK


LINEAR_PARAMS_FILENAME = "ioi_linear_params.json"


# Map from our short model identifiers (matching ioi_utils.get_model_config)
# to the HF model class names used elsewhere in our pipeline.
_SHORT_TO_CLASS_NAME = {
    "gpt2": "GPT2LMHeadModel",
    "qwen": "Qwen2ForCausalLM",
    "gemma": "Gemma2ForCausalLM",
    "llama": "LlamaForCausalLM",
}

# And the inverse, for callers that have an HF class name and want the
# short identifier the harness script expects.
_CLASS_NAME_TO_SHORT = {v: k for k, v in _SHORT_TO_CLASS_NAME.items()}


@dataclass(frozen=True)
class LinearParams:
    """The three per-model scalars + provenance."""

    bias: float
    token_coeff: float
    position_coeff: float
    score: Optional[float] = None
    model_name: Optional[str] = None  # full HF path, e.g. "google/gemma-2-2b"

    def as_dict(self) -> dict:
        """Return the {bias, token_coeff, position_coeff} dict the IOI
        causal model expects when called as ``get_causal_model(params)``.
        """
        return {
            "bias": self.bias,
            "token_coeff": self.token_coeff,
            "position_coeff": self.position_coeff,
        }


def model_short_name(model_class_name: str) -> str:
    """Convert ``Gemma2ForCausalLM`` → ``gemma`` etc. The harness scripts
    consistently use these short names; we keep both representations for
    interop (HF class name in our run.py, short name to the harness)."""
    if model_class_name not in _CLASS_NAME_TO_SHORT:
        raise ValueError(
            f"Unknown model class {model_class_name!r}. "
            f"Expected one of {sorted(_CLASS_NAME_TO_SHORT)}."
        )
    return _CLASS_NAME_TO_SHORT[model_class_name]


def model_class_name(short: str) -> str:
    """Inverse of ``model_short_name``."""
    if short not in _SHORT_TO_CLASS_NAME:
        raise ValueError(
            f"Unknown short model name {short!r}. "
            f"Expected one of {sorted(_SHORT_TO_CLASS_NAME)}."
        )
    return _SHORT_TO_CLASS_NAME[short]


def default_output_path() -> Path:
    """Where the bootstrap writes ``ioi_linear_params.json`` by default —
    next to the per-cell submission folders so the harness eval can find it."""
    return REPO_ROOT / "submissions" / "plot" / LINEAR_PARAMS_FILENAME


def bootstrap_linear_params(
    model_short: str,
    *,
    output_path: Optional[Path] = None,
    heads_list: Optional[list[tuple[int, int]]] = None,
    quick_test: bool = False,
    eval_batch_size: Optional[int] = None,
    use_gpu1: bool = False,
    extra_env: Optional[dict] = None,
    inline: bool = True,
) -> LinearParams:
    """Run the harness's ``ioi_learn_linear_params.py`` script and parse
    the JSON it writes.

    Parameters
    ----------
    model_short : str
        One of "gpt2", "qwen", "gemma", "llama".
    output_path : Path or None
        Where to write the JSON. Defaults to
        ``submissions/plot/ioi_linear_params.json``.
    heads_list : list of (layer, head) or None
        Attention heads to use for the regression. ``None`` keeps the
        harness default ``[(7,3), (7,9), (8,6), (8,10)]``.
    quick_test : bool
        Pass ``--quick_test`` to the harness script (uses size=10, single
        head). For pipeline smoke testing only — the resulting params are
        not usable for real submissions.
    eval_batch_size : int or None
        Forwarded to the harness script.
    use_gpu1 : bool
        Forwarded to the harness script.
    extra_env : dict or None
        Extra environment variables for the subprocess (e.g. ``HF_TOKEN``).
    inline : bool
        If True (default), run the bootstrap in-process via
        ``mib_submission.ioi._runner.run_inline`` so we can apply the
        runtime monkey-patches needed for GPT-2 (``position_ids`` fix)
        and Qwen (``head_dim`` injection). If False, shell out to the
        harness's ``ioi_learn_linear_params.py`` script — useful only when
        the harness's environment is fully compatible (Gemma-only).

    Returns
    -------
    LinearParams
        The fitted scalars plus regression score.

    Raises
    ------
    RuntimeError
        If the harness script exits non-zero or produces malformed JSON.
    """
    if model_short not in _SHORT_TO_CLASS_NAME:
        raise ValueError(
            f"Unknown short model name {model_short!r}; "
            f"expected one of {sorted(_SHORT_TO_CLASS_NAME)}."
        )

    if output_path is None:
        output_path = default_output_path()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if inline:
        from . import _runner
        _runner.run_inline(
            model_short,
            output_path=output_path,
            heads_list=heads_list,
            quick_test=quick_test,
            eval_batch_size=eval_batch_size,
        )
        return load_linear_params(output_path, model_short=model_short)

    script = MIB_TRACK / "baselines" / "ioi_baselines" / "ioi_learn_linear_params.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Harness bootstrap script missing at {script}. "
            "Confirm MIB submodule is initialised."
        )

    cmd: list[str] = [
        sys.executable,
        "-u",  # unbuffered stdout/stderr so live progress is visible in our log
        str(script),
        "--model", model_short,
        "--output_file", str(output_path),
    ]
    if heads_list is not None:
        # The harness script consumes "--heads_list '(7,3)' '(7,9)' ..."
        # via ``type=lambda s: eval(s)``. Pass each tuple as its repr.
        cmd.append("--heads_list")
        for L, H in heads_list:
            cmd.append(f"({int(L)},{int(H)})")
    if quick_test:
        cmd.append("--quick_test")
    if eval_batch_size is not None:
        cmd.extend(["--eval_batch_size", str(int(eval_batch_size))])
    if use_gpu1:
        cmd.append("--use_gpu1")

    env = None
    if extra_env:
        import os
        env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}

    print(f"[ioi-bootstrap] $ {' '.join(cmd)}")
    # The harness script imports relative to its parent dir; mirror what
    # it does at the top (sys.path appends are inside the __main__ block).
    cwd = MIB_TRACK / "baselines" / "ioi_baselines"
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ioi_learn_linear_params.py exited {result.returncode}; "
            "see stdout/stderr above."
        )
    if not output_path.is_file():
        raise RuntimeError(
            f"Harness script returned 0 but did not write {output_path}."
        )
    return load_linear_params(output_path, model_short=model_short)


def load_linear_params(
    json_path: Path,
    *,
    model_short: Optional[str] = None,
    model_class_name_filter: Optional[str] = None,
) -> LinearParams:
    """Load a previously-bootstrapped linear-params JSON file.

    Parameters
    ----------
    json_path : Path
        Path to ``ioi_linear_params.json``.
    model_short : str or None
        Top-level key to read. Pass exactly one of ``model_short`` or
        ``model_class_name_filter``.
    model_class_name_filter : str or None
        HF model class name (e.g. ``"Gemma2ForCausalLM"``) — translated
        to the short name internally.

    Returns
    -------
    LinearParams
    """
    if (model_short is None) == (model_class_name_filter is None):
        raise ValueError(
            "Pass exactly one of model_short or model_class_name_filter."
        )
    if model_short is None:
        assert model_class_name_filter is not None  # for type checker
        model_short = model_short_name(model_class_name_filter)

    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Linear params file missing at {json_path}. "
            f"Run bootstrap_linear_params({model_short!r}) first."
        )
    with json_path.open() as f:
        blob = json.load(f)

    if not isinstance(blob, dict):
        raise ValueError(f"{json_path} must contain a JSON object, got {type(blob).__name__}")

    # The harness script writes {model_short: {bias,...,score,model_name}}.
    # The example notebook also injects "model_class" at the top level.
    # Tolerate both shapes.
    entry = blob.get(model_short)
    if entry is None:
        raise KeyError(
            f"{json_path} has no entry for model {model_short!r}; "
            f"keys present: {sorted(k for k in blob if k != 'model_class')}"
        )
    required = ("bias", "token_coeff", "position_coeff")
    missing = [k for k in required if k not in entry]
    if missing:
        raise KeyError(
            f"{json_path}[{model_short!r}] missing required keys: {missing}"
        )
    return LinearParams(
        bias=float(entry["bias"]),
        token_coeff=float(entry["token_coeff"]),
        position_coeff=float(entry["position_coeff"]),
        score=float(entry["score"]) if "score" in entry else None,
        model_name=str(entry["model_name"]) if "model_name" in entry else None,
    )
