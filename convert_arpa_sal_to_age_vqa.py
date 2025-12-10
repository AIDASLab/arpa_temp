#!/usr/bin/env python3
"""
Convert ARPA_Sal.csv into an age-regression VQA JSON, mirroring ADNI_VQA_CVRF_GDS_AGE_test.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DIAGNOSIS_LABELS = [
    "No Cognitive Impairment",
    "Mild Cognitive Impairment",
    "Alzheimer's Dementia",
    "Not available or Other Dementia (not AD)",
]
DEPRESSION_LABELS = ["no", "yes"]
DIAGNOSIS_CODE_TO_LABEL = {"0": DIAGNOSIS_LABELS[0], "1": DIAGNOSIS_LABELS[1], "2": DIAGNOSIS_LABELS[2]}
SEX_CODE_TO_LABEL = {"0": "female", "1": "male"}
DEPRESSION_FLAG_TO_LABEL = {"0": "no", "1": "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ARPA_Sal.csv to age-regression VQA JSON.")
    parser.add_argument("--csv", default="ARPA_Sal.csv", help="Input CSV path (default: ARPA_Sal.csv)")
    parser.add_argument("--out", default="ARPA_Sal_age.json", help="Output JSON path (default: ARPA_Sal_age.json)")
    parser.add_argument("--split", default="test", help="Dataset split label (default: test)")
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_samples(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows:
        age = row.get("AGE", "").strip()
        bmi = row.get("BMI", "").strip()
        sbp = row.get("SYS_BP", "").strip()
        diagnosis_code = row.get("AD_FLAG", "").strip()
        depression_flag = row.get("DEP_FLAG", "").strip()
        sex_code = row.get("SEX", "").strip()

        diagnosis_label = DIAGNOSIS_CODE_TO_LABEL.get(diagnosis_code, DIAGNOSIS_LABELS[-1])
        depression_label = DEPRESSION_FLAG_TO_LABEL.get(depression_flag, "no")
        sex_label = SEX_CODE_TO_LABEL.get(sex_code, "unknown")

        question = (
            f"MRI scan for a {sex_label} participant with BMI {bmi} and systolic blood pressure {sbp} mmHg. "
            f"Depression status: {'depression' if depression_label == 'yes' else 'no depression'}. "
            f"Cognitive diagnosis: {diagnosis_label}. "
            "Predict the subject's age in years (one decimal)."
        )

        samples.append(
            {
                "image_id": row.get("ARPA_CD", "").strip(),
                "image_path": "path/to/image/" + row.get("ARPA_CD", "").strip(),
                "ptid": row.get("ARPA_CD", "").strip(),
                "visit": None,
                "acq_date": None,
                "source": "arpa_sal",
                "demographics": {"age": age, "sex": sex_label},
                "clinical": {
                    "bmi": _to_float(bmi),
                    "sbp": _to_float(sbp),
                    "gds_total": _to_float(row.get("DEP_RSLT", "").strip()),
                    "depression": depression_label == "yes",
                    "cog_brain_age": _to_float(row.get("COG_BRN_AGE", "").strip()),
                    "attention_bpi": _to_float(row.get("ATT_BPI", "").strip()),
                    "memory_bpi": _to_float(row.get("MEM_BPI", "").strip()),
                    "communication_bpi": _to_float(row.get("COM_BPI", "").strip()),
                },
                "question": question,
                "answer": _to_float(age),
                "answer_code": _to_float(age),
                "answer_units": "years",
            }
        )
    return samples


def summarize_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    return {"total": len(samples)}


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    rows = load_rows(csv_path)
    samples = build_samples(rows)

    payload = {
        "dataset": "ARPA_Sal_VQA_AGE",
        "task": "age_regression",
        "source_csvs": {"arpa_sal": str(csv_path.resolve())},
        "mri_root": None,
        "label_space": ["continuous_age_years"],
        "num_samples": len(samples),
        "source_counts": summarize_counts(samples),
        "notes": {
            "question_template": "MRI scan for a <sex> participant with BMI <bmi> and SBP <sbp> mmHg. "
            "Depression status: <depression>. Cognitive diagnosis: <diagnosis>. "
            "Predict age in years (one decimal).",
            "diagnosis_code_mapping": DIAGNOSIS_CODE_TO_LABEL,
            "sex_code_mapping": SEX_CODE_TO_LABEL,
            "depression_flag_mapping": DEPRESSION_FLAG_TO_LABEL,
        },
        "split": args.split,
        "samples": samples,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
