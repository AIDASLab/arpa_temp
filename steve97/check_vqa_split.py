#!/usr/bin/env python3
"""Check that test questions do not overlap with training questions."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Set, Iterable, Tuple


class AhoCorasick:
    """Multi-pattern search for detecting substrings efficiently."""

    def __init__(self) -> None:
        self.goto: List[Dict[str, int]] = [dict()]
        self.fail: List[int] = [0]
        self.output: List[Set[int]] = [set()]

    def add(self, pattern: str, value: int) -> None:
        node = 0
        for ch in pattern:
            if ch not in self.goto[node]:
                self.goto[node][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append(set())
            node = self.goto[node][ch]
        self.output[node].add(value)

    def build(self) -> None:
        from collections import deque

        queue = deque()
        for _, nxt in self.goto[0].items():
            queue.append(nxt)
            self.fail[nxt] = 0

        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                self.output[s].update(self.output[self.fail[s]])

    def search(self, text: str) -> Iterable[int]:
        node = 0
        for ch in text:
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for value in self.output[node]:
                yield value


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    return data.get("samples", [])


def main():
    parser = argparse.ArgumentParser(
        description="Verify that test questions are absent from the training split, including substring containment."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("/home/arpa/steve97/dataset_v2/ADNI_VQA_train_v2.json"),
        help="Path to training json file",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("/home/arpa/steve97/dataset_v2/ADNI_VQA_test_v2.json"),
        help="Path to test json file",
    )
    args = parser.parse_args()

    train_samples = load_samples(args.train)
    test_samples = load_samples(args.test)

    train_questions = {sample["question"] for sample in train_samples}
    test_questions = [sample["question"] for sample in test_samples]

    overlap_exact = train_questions.intersection(test_questions)

    automaton = AhoCorasick()
    for idx, question in enumerate(test_questions):
        automaton.add(question, idx)
    automaton.build()

    substring_hits: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for train_sample in train_samples:
        matches = set(automaton.search(train_sample["question"]))
        for match_idx in matches:
            test_sample = test_samples[match_idx]
            if train_sample["question"] == test_sample["question"]:
                continue
            substring_hits.append((test_sample, train_sample))

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Exact matching questions: {len(overlap_exact)}")
    print(f"Substring matches (test question contained in train question): {len(substring_hits)}")

    if overlap_exact:
        print("Sample exact overlapping questions:")
        for i, question in enumerate(sorted(overlap_exact)):
            print(f"{i+1}: {question}")
            if i >= 9:
                break

    if substring_hits:
        print("Sample substring overlaps:")
        for i, (test_sample, train_sample) in enumerate(substring_hits[:10]):
            print(f"{i+1}. test_id={test_sample.get('image_id')} train_id={train_sample.get('image_id')}")
            print("   test question:", test_sample["question"])
            print("   train question:", train_sample["question"])
        sys.exit(1)


if __name__ == "__main__":
    main()
