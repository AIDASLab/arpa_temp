import json
import random
from pathlib import Path

def main():

    # json_path = Path("/home/arpa/steve97/dataset_final/regression/ADNI_VQA_age_history.json")
    json_path = Path("/home/arpa/steve97/dataset_final/regression/ADNI_VQA_age_updated_history.json")

    print("Loading JSON...")
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    total = len(samples)

    print("Total samples found:", total)

    # -----------------------------------------------------
    # 1. 샘플 개수 검증 (7410개인지)
    # -----------------------------------------------------
    expected_total = 7410
    if total != expected_total:
        print(f"❌ ERROR: Expected {expected_total} samples, but found {total}. Split aborted.")
        return
    print("✔ Sample count correct. Proceeding with random split...")

    # -----------------------------------------------------
    # 2. split size 정의
    # -----------------------------------------------------
    test_size = 1482
    train_size = 5928
    assert test_size + train_size == expected_total

    # -----------------------------------------------------
    # 3. random shuffle (재현성 위해 seed 고정)
    # -----------------------------------------------------
    random_seed = 42
    random.Random(random_seed).shuffle(samples)

    # -----------------------------------------------------
    # 4. split 수행
    # -----------------------------------------------------
    test_samples = samples[:test_size]
    train_samples = samples[test_size:test_size + train_size]

    # -----------------------------------------------------
    # 5. 새로운 JSON 구조 생성
    # -----------------------------------------------------
    train_data = dict(data)
    train_data["samples"] = train_samples

    test_data = dict(data)
    test_data["samples"] = test_samples

    # -----------------------------------------------------
    # 6. 저장
    # -----------------------------------------------------
    train_output = json_path.with_name("ADNI_VQA_age_updated_history_train.json")
    test_output = json_path.with_name("ADNI_VQA_age_updated_history_test.json")

    with open(train_output, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(test_output, "w") as f:
        json.dump(test_data, f, indent=2)

    print("==============================================")
    print("RANDOM SPLIT COMPLETE (Regression Version)")
    print("Train samples:", len(train_samples))
    print("Test samples:", len(test_samples))
    print("Saved TRAIN →", train_output)
    print("Saved TEST →", test_output)
    print("==============================================")

if __name__ == "__main__":
    main()
