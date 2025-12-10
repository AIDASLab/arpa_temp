#!/usr/bin/env python3
"""
Convert ARPA_Sal.csv into a VQA-style JSON aligned with ADNI_VQA_CVRF_GDS_test.json.

This relies only on the standard library so it can run anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


# Static label space to mirror the ADNI VQA test set.
DIAGNOSIS_LABELS = [
    "No Cognitive Impairment",
    "Mild Cognitive Impairment",
    "Alzheimer's Dementia",
    "Not available or Other Dementia (not AD)",
]
DEPRESSION_LABELS = ["no", "yes"]

# Mapping of ARPA flags to labels/codes.
DIAGNOSIS_CODE_TO_LABEL = {"0": DIAGNOSIS_LABELS[0], "1": DIAGNOSIS_LABELS[1], "2": DIAGNOSIS_LABELS[2]}
SEX_CODE_TO_LABEL = {"0": "female", "1": "male"}
DEPRESSION_FLAG_TO_LABEL = {"0": "no", "1": "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ARPA_Sal.csv into VQA JSON format.")
    parser.add_argument("--csv", default="ARPA_Sal.csv", help="Input ARPA CSV path (default: ARPA_Sal.csv)")
    parser.add_argument("--out", default="ARPA_Sal.json", help="Output JSON path (default: ARPA_Sal.json)")
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_samples(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for row in rows:
        diagnosis_code = row.get("AD_FLAG", "").strip()
        depression_flag = row.get("DEP_FLAG", "").strip()

        age = row.get("AGE", "").strip()
        bmi = row.get("BMI", "").strip()
        sbp = row.get("SYS_BP", "").strip()
        sex_code = row.get("SEX", "").strip()

        diagnosis_label = DIAGNOSIS_CODE_TO_LABEL.get(diagnosis_code, DIAGNOSIS_LABELS[-1])
        depression_label = DEPRESSION_FLAG_TO_LABEL.get(depression_flag, "no")
        sex_label = SEX_CODE_TO_LABEL.get(sex_code, "unknown")

        question = (
            f"MRI scan for a {age}-year-old {sex_label} with BMI {bmi} and systolic blood pressure {sbp} mmHg. "
            f"Depression: {'depression' if depression_label == 'yes' else 'no depression'}. "
            "Predict the cognitive diagnosis. Respond with one of: "
            + ", ".join(DIAGNOSIS_LABELS)
            + "."
        )

        sample = {
            "image_id": row.get("ARPA_CD", "").strip(),
            "image_path": "path/to/image/" + row.get("ARPA_CD", "").strip(),
            "ptid": row.get("ARPA_CD", "").strip(),
            "visit": None,
            "acq_date": None,
            "source": "arpa_sal",
            "question": question,
            "answer": diagnosis_label,
            "answer_code": {
                "diagnosis_code": diagnosis_code if diagnosis_code else None,
                "depression": depression_label,
            },
            "choices": {"diagnosis": DIAGNOSIS_LABELS, "depression": DEPRESSION_LABELS},
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
            "metadata": {
                "institution_code": row.get("INSTCD", "").strip(),
                "raw_diagnosis_code": diagnosis_code,
                "raw_depression_flag": depression_flag,
                "judgment_code": row.get("JUDGRSLTCD", "").strip(),
                "judgment_comment": row.get("JUDDJUDGHANGCNTS", "").strip(),
            },
        }

        samples.append(sample)

    return samples


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


def summarize_counts(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    diagnosis_counts: Dict[str, int] = {}
    depression_counts: Dict[str, int] = {}

    for sample in samples:
        diag_code = sample["answer_code"]["diagnosis_code"] or "unknown"
        diagnosis_counts[diag_code] = diagnosis_counts.get(diag_code, 0) + 1

        dep_label = sample["answer_code"]["depression"]
        depression_counts[dep_label] = depression_counts.get(dep_label, 0) + 1

    return {"diagnosis": diagnosis_counts, "depression": depression_counts}


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    rows = load_rows(csv_path)
    samples = build_samples(rows)

    payload = {
        "source_csvs": {"arpa_sal": str(csv_path.resolve())},
        "mri_root": None,
        "label_space": {"diagnosis": DIAGNOSIS_LABELS, "depression": DEPRESSION_LABELS},
        "notes": {
            "question_template": "MRI scan for a <age>-year-old <sex> with BMI <bmi> and SBP <sbp> mmHg. "
            "Depression: <yes/no>. Predict the cognitive diagnosis.",
            "diagnosis_code_mapping": DIAGNOSIS_CODE_TO_LABEL,
            "sex_code_mapping": SEX_CODE_TO_LABEL,
            "depression_flag_mapping": DEPRESSION_FLAG_TO_LABEL,
        },
        "dataset": "ARPA_Sal_VQA",
        "num_samples": len(samples),
        "source_counts": summarize_counts(samples),
        "samples": samples,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
