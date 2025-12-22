import json
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------
# question 재작성 함수
# ---------------------------------------------
def rebuild_question(sample):
    """Replace only the medical_history portion inside the existing question."""

    question = sample["question"]
    history_str = "; ".join(sample["medical_history"])

    marker = "Relevant medical history:"

    # case 1: 기존 question에 medical history가 이미 있음
    if marker in question:
        prefix = question.split(marker)[0]
        new_question = prefix + marker + " " + history_str + "."
        return new_question

    # case 2: 기존 question에 medical history가 없다면 끝에 추가
    else:
        if not question.strip().endswith("."):
            question += "."
        return question + f" {marker} {history_str}."


# ---------------------------------------------
# main
# ---------------------------------------------
def main():

    # history_path = Path("/home/arpa/steve97/data/ADNI_VQA_with_history.json")
    history_path = Path("/home/arpa/steve97/data_v2/ADNI_VQA_with_updated_history.json")
    age_path = Path("/home/arpa/steve97/data/ADNI_VQA_age.json")

    print("Loading JSON files...")
    with open(history_path, "r") as f:
        history_data = json.load(f)

    with open(age_path, "r") as f:
        age_data = json.load(f)

    history_samples = history_data["samples"]
    age_samples = age_data["samples"]

    # 인덱싱 편하게 딕셔너리로 만들기
    history_dict = {}

    for s in history_samples:
        key = (s["image_id"], s["ptid"], s["visit"], s["acq_date"])
        history_dict[key] = s

    updated = 0
    unmatched = []

    print("Processing age samples...")

    for sample in tqdm(age_samples):

        key = (sample["image_id"], sample["ptid"], sample["visit"], sample["acq_date"])

        if key in history_dict:
            history_sample = history_dict[key]

            # 기존 history 가져오기
            med_hist = history_sample.get("medical_history", [])

            # age 샘플에 추가
            sample["medical_history"] = list(med_hist)

            # question 재작성
            sample["question"] = rebuild_question(sample)

            updated += 1
        else:
            unmatched.append({
                "image_id": sample["image_id"],
                "ptid": sample["ptid"],
                "visit": sample["visit"],
                "acq_date": sample["acq_date"]
            })

    # 저장
    output_path = age_path.with_name("ADNI_VQA_age_updated.json")
    unmatched_path = age_path.with_name("ADNI_VQA_age_unmatched.json")

    with open(output_path, "w") as f:
        json.dump(age_data, f, indent=2)

    with open(unmatched_path, "w") as f:
        json.dump(unmatched, f, indent=2)

    print("==========================================")
    print("UPDATE COMPLETE")
    print("Matched + updated samples:", updated)
    print("Unmatched samples:", len(unmatched))
    print("Saved updated age file →", output_path)
    print("Saved unmatched list →", unmatched_path)
    print("==========================================")

if __name__ == "__main__":
    main()
