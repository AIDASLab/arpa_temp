import json
from collections import Counter

def check_ptid_unique(json_path):
    # JSON 파일 읽기
    with open(json_path, "r") as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    ptids = [s.get("ptid") for s in samples]

    # ptid 등장 횟수 카운트
    counter = Counter(ptids)

    # 중복 항목만 추출
    duplicates = {ptid: cnt for ptid, cnt in counter.items() if cnt > 1}

    # 출력
    print("Total samples:", len(samples))
    print("Unique ptid count:", len(set(ptids)))

    if len(duplicates) == 0:
        print("✔ 모든 ptid가 고유합니다 (중복 없음).")
    else:
        print("❌ 중복 ptid 발견됨:")
        for ptid, cnt in duplicates.items():
            print(f"  - {ptid}: {cnt}회 등장")

    return duplicates


# 실행
# json_path = "/home/arpa/steve97/data/ADNI_VQA_with_history.json"
# json_path = "/home/arpa/steve97/data/ADNI_VQA.json"
json_path = "/home/arpa/steve97/data/ADNI_VQA_age.json"
duplicates = check_ptid_unique(json_path)
