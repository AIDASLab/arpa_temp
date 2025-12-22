"""Age-specific evaluation entry point for ADNI VQA LoRA models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train.eval_adni_vqa_lora import (  # type: ignore  # pylint: disable=wrong-import-position
    build_parser as build_base_parser,
)
from src.train.eval_adni_vqa_lora import evaluate as evaluate_lora  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = "Evaluate Med3DVLM age-regression LoRA checkpoints on the ADNI age split."
    parser.set_defaults(
        dataset_json="data/ADNI_VQA_age_test.json",
        dataset_label="adni_vqa_age",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = evaluate_lora(args)
    if result and result.get("task") == "regression":
        mae = result.get("mae")
        if isinstance(mae, (int, float)):
            print(f"[AgeEval] MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
