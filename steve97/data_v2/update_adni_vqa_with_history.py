import json
import pandas as pd
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------
# 날짜 변환 함수
# ---------------------------------------------------------
def normalize_mmddyyyy(date_str):
    """Convert MM/DD/YYYY → YYYY-MM-DD"""
    return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")


def extract_year(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").year


# ---------------------------------------------------------
# question 재빌드
# ---------------------------------------------------------
def rebuild_question(sample):
    image_id = sample["image_id"]
    acq_date = sample["acq_date"]
    age = sample["demographics"]["age"]
    sex = sample["demographics"]["sex"]
    education = sample["demographics"]["education"]

    history_str = "; ".join(sample["medical_history"])

    q = (
        f"MRI scan (Image Data ID {image_id}) acquired on {acq_date} "
        f"for a {age}-year-old {sex} with {education}. "
        "Predict the cognitive diagnosis. Respond with one of: "
        "No Cognitive Impairment, Mild Cognitive Impairment, Alzheimer's Dementia, "
        "Not available or Other Dementia (not AD). "
        f"Relevant medical history: {history_str}."
    )
    return q


# ---------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------
def main():

    # 경로
    json_path = Path("/home/arpa/steve97/data/ADNI_VQA_with_history.json")
    csv_phc_path = Path("/home/arpa/ADSP_PHC_CVRF_04Dec2025.csv")
    csv_gds_path = Path("/home/arpa/GDSCALE_04Dec2025.csv")

    print("Loading ADNI VQA JSON...")
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = data["samples"]

    print("Loading CSV files...")
    phc_df = pd.read_csv(csv_phc_path, dtype=str)
    gds_df = pd.read_csv(csv_gds_path, dtype=str)

    # 날짜 clean
    phc_df["EXAMDATE"] = phc_df["EXAMDATE"].astype(str).str.strip()
    gds_df["VISDATE"] = gds_df["VISDATE"].astype(str).str.strip()

    total_samples = len(samples)

    # notes 계산용
    matched_bmi = 0
    matched_sbp = 0
    matched_depression = 0
    unmatched_samples = []

    print("Processing samples...")

    for sample in samples:

        ptid = sample["ptid"]
        visit = sample["visit"]
        acq_date_raw = sample["acq_date"]
        acq_date_norm = normalize_mmddyyyy(acq_date_raw)
        acq_date_dt = datetime.strptime(acq_date_norm, "%Y-%m-%d")

        matched_any = False

        # ----------------------------------------------
        # (1) PHC_CVRF — PTID 기반 필터 후 과거 기록 모두 반영
        # ----------------------------------------------
        phc_subset = phc_df[phc_df["PTID"] == ptid]

        for _, row in phc_subset.iterrows():
            try:
                row_date_dt = datetime.strptime(row["EXAMDATE"], "%Y-%m-%d")
            except:
                continue

            # sample보다 과거만 사용
            if row_date_dt < acq_date_dt:
                bmi = row.get("PHC_BMI", "")
                sbp = row.get("PHC_SBP", "")
                date_str = row["EXAMDATE"]

                if bmi not in ["", "nan", None]:
                    sample["medical_history"].append(f"BMI: {bmi} ({date_str})")
                    matched_bmi += 1
                    matched_any = True

                if sbp not in ["", "nan", None]:
                    sample["medical_history"].append(f"SBP: {sbp} mmHg ({date_str})")
                    matched_sbp += 1
                    matched_any = True

        # ----------------------------------------------
        # (2) GDSCALE — PTID 기반 필터 후 과거 기록 모두 반영
        # ----------------------------------------------
        gds_subset = gds_df[gds_df["PTID"] == ptid]

        for _, row in gds_subset.iterrows():
            try:
                row_date_dt = datetime.strptime(row["VISDATE"], "%Y-%m-%d")
            except:
                continue

            if row_date_dt < acq_date_dt:
                gdtotal_str = row.get("GDTOTAL", "")
                try:
                    gdtotal = int(gdtotal_str)
                except:
                    gdtotal = None

                visdate = row["VISDATE"]

                if gdtotal is not None:
                    if gdtotal >= 10:
                        sample["medical_history"].append(f"Depression ({visdate})")
                    else:
                        sample["medical_history"].append(f"No depression ({visdate})")

                    matched_depression += 1
                    matched_any = True

        # ----------------------------------------------
        # 매칭 아무것도 없으면 unmatched 기록
        # ----------------------------------------------
        if not matched_any:
            unmatched_samples.append({
                "ptid": ptid,
                "visit": visit,
                "acq_date": acq_date_raw,
                "image_id": sample["image_id"]
            })

        # ----------------------------------------------
        # question 재생성
        # ----------------------------------------------
        sample["question"] = rebuild_question(sample)

    # ============================================================
    # notes 업데이트
    # ============================================================
    data["notes"]["num_samples"] = total_samples
    data["notes"]["matched_bmi"] = matched_bmi
    data["notes"]["matched_sbp"] = matched_sbp
    data["notes"]["matched_depression"] = matched_depression
    data["notes"]["unmatched_samples"] = len(unmatched_samples)

    # ============================================================
    # unmatched 로그 저장
    # ============================================================
    unmatched_path = json_path.with_name("ADNI_VQA_unmatched.json")
    with open(unmatched_path, "w") as f:
        json.dump(unmatched_samples, f, indent=2)

    # ============================================================
    # 최종 JSON 저장
    # ============================================================
    backup_path = json_path.with_suffix(".backup.json")
    updated_path = json_path.with_suffix(".updated.json")

    with open(backup_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(updated_path, "w") as f:
        json.dump(data, f, indent=2)

    print("\n==============================================")
    print("UPDATE COMPLETE")
    print("Total samples:", total_samples)
    print("BMI matched:", matched_bmi)
    print("SBP matched:", matched_sbp)
    print("Depression matched:", matched_depression)
    print("Unmatched samples:", len(unmatched_samples))
    print("==============================================")
    print(f"Saved updated JSON → {updated_path}")
    print(f"Saved unmatched samples → {unmatched_path}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
