"""Evaluation script for ADNI VQA LoRA models."""

import argparse
import csv
import json
import os
import re
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
DIAG_PATTERN = re.compile(r"cognitive diagnosis:\s*([^.\n]+)", re.IGNORECASE)

# === 새 모델 구조 import (train 스크립트에서 정의한 래퍼) ===
try:
    from src.train.train_adni_vqa_lora_ys import CausalLMWithRegressionHead
except ImportError:
    CausalLMWithRegressionHead = None
    print(
        "[Eval] Warning: CausalLMWithRegressionHead could not be imported. "
        "Regression evaluation will fall back to text generation if used."
    )
# =============================================================


def load_json_dict(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_samples(path: Path) -> List[Dict]:
    return load_json_dict(path)["samples"]


def dataset_is_regression(meta: Dict) -> bool:
    task = str(meta.get("task", "")).lower()
    if task in {"age_regression", "regression"}:
        return True
    for label in meta.get("label_space", []) or []:
        if isinstance(label, str) and "continuous" in label.lower():
            return True
    return False


def denormalize_value(value: float, stats: Optional[Dict[str, float]]) -> float:
    if not stats:
        return value
    std = stats.get("std") or 1.0
    if std == 0:
        std = 1.0
    mean = stats.get("mean", 0.0)
    return value * std + mean


def _has_adapter_artifacts(path: Path) -> bool:
    """True if the directory looks like a PEFT adapter export."""
    if not path.is_dir():
        return False
    adapter_files = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
    )
    return any((path / name).exists() for name in adapter_files)


def _resolve_adapter_directory(lora_root: Path) -> Tuple[Path, bool]:
    """Return directory holding adapter artifacts. Second value shows fallback usage."""
    if _has_adapter_artifacts(lora_root):
        return lora_root, False
    best_root = lora_root / "best_checkpoints"
    if best_root.is_dir():
        candidates = [p for p in best_root.iterdir() if _has_adapter_artifacts(p)]
        if len(candidates) == 1:
            return candidates[0], True
    return lora_root, False


def extract_answer_text(full_text: str) -> str:
    if "Answer:" in full_text:
        return full_text.split("Answer:", 1)[1].strip()
    return full_text.strip()


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_first_float(text: str) -> Optional[float]:
    match = FLOAT_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def normalize_diagnosis_label(label: Optional[str]) -> Optional[str]:
    """Normalize diagnosis strings to consistent labels."""
    if not label:
        return None
    cleaned = str(label).strip()
    lowered = cleaned.lower().replace("’", "'")
    if "no cognitive impairment" in lowered:
        return "No Cognitive Impairment"
    if "mild cognitive impairment" in lowered or lowered == "mci":
        return "Mild Cognitive Impairment"
    if "alzheimer" in lowered or lowered == "ad":
        return "Alzheimer's Dementia"
    if "not available or other dementia" in lowered:
        return "Not available or Other Dementia (not AD)"
    return cleaned


def extract_diagnosis_label(sample: Dict) -> Optional[str]:
    """Pull cognitive diagnosis from structured fields or the question string."""
    # Structured fields first
    for key in ("diagnosis", "cognitive_diagnosis"):
        if key in sample:
            return normalize_diagnosis_label(sample.get(key))
    clinical = sample.get("clinical") or {}
    for key in ("diagnosis", "cognitive_diagnosis"):
        if key in clinical:
            return normalize_diagnosis_label(clinical.get(key))

    # Fallback: parse from question template
    question = sample.get("question", "")
    match = DIAG_PATTERN.search(question)
    if match:
        return normalize_diagnosis_label(match.group(1))
    return None


def fit_bias_correction(age_unimpaired: List[float], y_pred_unimpaired: List[float]) -> Optional[Tuple[float, float]]:
    """
    Use youngest 5% + oldest 5% of cognitively unimpaired group to fit y = a*x + b.
    Returns (a, b) or None if insufficient data.
    """
    if len(age_unimpaired) < 2:
        return None

    age_unimpaired = np.array(age_unimpaired, dtype=float)
    y_pred_unimpaired = np.array(y_pred_unimpaired, dtype=float)

    n = len(age_unimpaired)
    k = max(1, int(n * 0.05))

    idx_sorted = np.argsort(age_unimpaired)
    idx_low = idx_sorted[:k]
    idx_high = idx_sorted[-k:]
    idx_bias = np.concatenate([idx_low, idx_high])

    X = age_unimpaired[idx_bias].reshape(-1, 1)
    y = y_pred_unimpaired[idx_bias]

    reg = LinearRegression().fit(X, y)
    a = float(reg.coef_[0])
    b = float(reg.intercept_)
    return a, b


def apply_bias_correction(y_pred: np.ndarray, age: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply y' = y + [x - (a*x + b)]."""
    return y_pred + (age - (a * age + b))


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def maybe_init_distributed() -> None:
    if is_dist_initialized():
        return
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def sanitize_metric_key(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or "unknown"


def evaluate(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    maybe_init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
    wandb_enabled = args.use_wandb and get_rank() == 0
    if wandb_enabled:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[Eval] wandb is not installed; continuing without remote logging.")
            wandb_enabled = False
        else:
            wandb_kwargs = {
                "project": args.wandb_project,
                "job_type": "evaluation",
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            wandb_run = wandb.init(**wandb_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === LoRA base model 로드 ===
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )
    lora_root = Path(args.lora_path)
    adapter_dir, used_best = _resolve_adapter_directory(lora_root)
    if adapter_dir != lora_root:
        source_label = "best checkpoint" if used_best else "provided"
        print(
            f"[Eval] Adapter artifacts not found directly under {lora_root}. "
            f"Using {source_label} directory: {adapter_dir}"
        )
    if not _has_adapter_artifacts(adapter_dir):
        raise FileNotFoundError(
            f"No adapter_config/adapter_model files found under {args.lora_path}."
            " Please specify a directory that contains PEFT adapter weights."
        )

    norm_stats_path = adapter_dir / "regression_norm_stats.json"
    normalization_meta: Optional[Dict[str, float]] = None
    if norm_stats_path.exists():
        try:
            with norm_stats_path.open("r", encoding="utf-8") as f:
                normalization_meta = json.load(f)
        except Exception as err:
            print(f"[Eval] WARNING: Failed to read regression_norm_stats.json: {err}")
            normalization_meta = None
    normalized_run = bool(normalization_meta and normalization_meta.get("enabled"))
    if normalized_run:
        mean = normalization_meta.get("mean", 0.0)
        std = normalization_meta.get("std", 1.0)
        print(f"[Eval] Detected normalized checkpoint (mean={mean:.4f}, std={std:.4f}).")

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    dataset_path = PROJECT_ROOT / args.dataset_json
    dataset_data = load_json_dict(dataset_path)
    samples = dataset_data.get("samples", [])
    dataset_label = args.dataset_label or dataset_path.stem
    is_regression = dataset_is_regression(dataset_data)

    lora_path = adapter_dir

    if is_regression and CausalLMWithRegressionHead is not None:
        head_path = lora_path / "regression_head.bin"
        cfg_path = lora_path / "regression_head_config.json"

        loss_type = "gaussian"
        mle_variance = 1.0
        huber_beta = 1.0

        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    head_cfg = json.load(f)
                loss_type = head_cfg.get("loss_type", loss_type)
                mle_variance = float(head_cfg.get("mle_variance", mle_variance))
                huber_beta = float(head_cfg.get("huber_beta", huber_beta))
            except Exception as e:
                print(f"[Eval] WARNING: Failed to load regression_head_config.json: {e}")

        wrapper = CausalLMWithRegressionHead(
            model,
            loss_type=loss_type,
            mle_variance=mle_variance,
            huber_beta=huber_beta,
            normalization=normalization_meta,
        )

        if head_path.exists():
            print(f"[Eval] Loading regression head weights from {head_path}")
            head_state = torch.load(head_path, map_location="cpu")
            wrapper.regression_head.load_state_dict(head_state)
        else:
            print(
                f"[Eval] WARNING: regression_head.bin not found in {lora_path}; "
                "using randomly initialized regression head."
            )

        model = wrapper

    model.to(device)
    model.eval()

    if get_rank() == 0:
        print(
            f"[Eval] Dataset '{dataset_label}' -> {dataset_path} "
            f"({len(samples)} samples), task={'regression' if is_regression else 'classification'}"
        )

    world_size = get_world_size()
    rank = get_rank()

    processed = 0
    predictions_local: List[Dict[str, str]] = []

    if is_regression:
        abs_error = 0.0
        parsed = 0.0
        invalid_predictions = 0.0
        diagnosis_labels = sorted(
            filter(
                None,
                {extract_diagnosis_label(sample) for sample in samples},
            )
        )
        diag_abs_error = {label: 0.0 for label in diagnosis_labels}
        diag_parsed = {label: 0.0 for label in diagnosis_labels}
        diag_invalid = {label: 0.0 for label in diagnosis_labels}
        reg_records: List[Tuple[float, float, Optional[str]]] = []
    else:
        correct = 0.0
        total = 0.0
        class_names = sorted({sample["answer"] for sample in samples})
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        class_correct = [0.0] * len(class_names)
        class_total = [0.0] * len(class_names)

    for idx in range(rank, len(samples), world_size):
        sample = samples[idx]
        prompt = "You are a medical imaging assistant." + f" Question: {sample['question']}\nAnswer:"
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        batch = {k: v.to(device) for k, v in encoded.items()}
        prediction_correct: Optional[str] = None
        diag_label = extract_diagnosis_label(sample)

        if is_regression and CausalLMWithRegressionHead is not None:
            # === 회귀: regression head의 스칼라 출력 사용 ===
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                # CausalLMWithRegressionHead에서 logits의 마지막 차원 1짜리 스칼라
                if isinstance(outputs, dict):
                    pred_tensor = outputs["logits"].squeeze(-1)
                else:
                    pred_tensor = outputs.logits.squeeze(-1)
            pred_value_raw = float(pred_tensor.item())
            if normalized_run:
                pred_value = denormalize_value(pred_value_raw, normalization_meta)
            else:
                pred_value = pred_value_raw
            predicted = f"{pred_value:.4f}"
            print(predicted)

            gt_value = safe_float(sample.get("answer"))
            if gt_value is None:
                continue

            if not (math.isfinite(pred_value)):
                # NaN / inf 방어
                invalid_predictions += 1
                if diag_label in diag_invalid:
                    diag_invalid[diag_label] += 1
            else:
                diff = abs(pred_value - gt_value)
                abs_error += diff
                parsed += 1
                prediction_correct = f"{diff:.4f}"  # 회귀는 절대오차를 기록
                if diag_label in diag_abs_error:
                    diag_abs_error[diag_label] += diff
                    diag_parsed[diag_label] += 1
                reg_records.append((pred_value, gt_value, diag_label))

        else:
            # === classification or fallback: 텍스트 생성 기반 평가 ===
            input_ids = batch.pop("input_ids")
            with torch.no_grad():
                outputs = model.generate(
                    inputs=input_ids,
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = extract_answer_text(generated)
            predicted = completion.strip().split("\n")[0]
            print(predicted)

            if is_regression:
                # 회귀인데 wrapper를 못 불렀을 때는 이전 방식처럼 텍스트에서 숫자 파싱
                gt_value = safe_float(sample.get("answer"))
                if gt_value is None:
                    continue
                pred_value = extract_first_float(predicted)
                if pred_value is None:
                    invalid_predictions += 1
                    if diag_label in diag_invalid:
                        diag_invalid[diag_label] += 1
                else:
                    diff = abs(pred_value - gt_value)
                    abs_error += diff
                    parsed += 1
                    prediction_correct = f"{diff:.4f}"
                    if diag_label in diag_abs_error:
                        diag_abs_error[diag_label] += diff
                        diag_parsed[diag_label] += 1
                    reg_records.append((pred_value, gt_value, diag_label))
            else:
                # 분류
                is_correct = sample["answer"].lower() in predicted.lower()
                if is_correct:
                    correct += 1
                total += 1
                prediction_correct = str(bool(is_correct))

                cls_idx = class_to_idx.get(sample["answer"])
                if cls_idx is not None:
                    class_total[cls_idx] += 1
                    if is_correct:
                        class_correct[cls_idx] += 1

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} samples...")

        image_id = sample.get("image_id") or sample.get("image_path") or f"sample_{idx}"
        predictions_local.append(
            {
                "image_id": str(image_id),
                "prediction": str(predicted),
                "is_correct": "" if prediction_correct is None else prediction_correct,
            }
        )

    # Aggregate results across ranks
    result: Optional[Dict[str, float]] = None

    if is_regression:
        if is_dist_initialized():
            tensor = torch.tensor([abs_error, parsed, invalid_predictions], device=device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            abs_error = float(tensor[0].item())
            parsed = float(tensor[1].item())
            invalid_predictions = float(tensor[2].item())
            if diagnosis_labels:
                diag_abs_error_tensor = torch.tensor(
                    [diag_abs_error[label] for label in diagnosis_labels],
                    device=device,
                    dtype=torch.float32,
                )
                diag_parsed_tensor = torch.tensor(
                    [diag_parsed[label] for label in diagnosis_labels],
                    device=device,
                    dtype=torch.float32,
                )
                diag_invalid_tensor = torch.tensor(
                    [diag_invalid[label] for label in diagnosis_labels],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(diag_abs_error_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(diag_parsed_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(diag_invalid_tensor, op=dist.ReduceOp.SUM)
                diag_abs_error = {
                    label: float(value) for label, value in zip(diagnosis_labels, diag_abs_error_tensor.tolist())
                }
                diag_parsed = {
                    label: float(value) for label, value in zip(diagnosis_labels, diag_parsed_tensor.tolist())
                }
                diag_invalid = {
                    label: float(value) for label, value in zip(diagnosis_labels, diag_invalid_tensor.tolist())
                }
        mae = abs_error / parsed if parsed else float("nan")
        result = {
            "task": "regression",
            "mae": mae,
            "parsed": parsed,
            "invalid": invalid_predictions,
        }
        if diagnosis_labels:
            result["per_diagnosis_mae"] = {
                label: (diag_abs_error[label] / diag_parsed[label] if diag_parsed[label] else float("nan"))
                for label in diagnosis_labels
            }
    else:
        if is_dist_initialized():
            tensor = torch.tensor([correct, total], device=device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            correct = float(tensor[0].item())
            total = int(tensor[1].item())

            if class_names:
                class_correct_tensor = torch.tensor(class_correct, device=device, dtype=torch.float32)
                class_total_tensor = torch.tensor(class_total, device=device, dtype=torch.float32)
                dist.all_reduce(class_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(class_total_tensor, op=dist.ReduceOp.SUM)
                class_correct = class_correct_tensor.tolist()
                class_total = class_total_tensor.tolist()
        result = {
            "task": "classification",
            "accuracy": (correct / total) if total else 0.0,
            "correct": correct,
            "total": total,
        }

    # Gather predictions across ranks for CSV logging
    all_predictions: List[Dict[str, str]] = []
    all_reg_records: List[Tuple[float, float, Optional[str]]] = []
    if is_dist_initialized():
        gathered: List[List[Dict[str, str]]] = [None] * world_size  # type: ignore
        dist.all_gather_object(gathered, predictions_local)
        if rank == 0:
            for bucket in gathered:
                if bucket:
                    all_predictions.extend(bucket)
        if is_regression:
            gathered_reg: List[List[Tuple[float, float, Optional[str]]]] = [None] * world_size  # type: ignore
            dist.all_gather_object(gathered_reg, reg_records if is_regression else [])
            if rank == 0:
                for bucket in gathered_reg:
                    if bucket:
                        all_reg_records.extend(bucket)
    else:
        all_predictions = predictions_local
        if is_regression:
            all_reg_records = reg_records

    if get_rank() == 0:
        print(f"[Eval] Dataset '{dataset_label}' results")
        if is_regression:
            parsed_int = int(parsed)
            invalid_int = int(invalid_predictions)
            print(f"Parsed samples: {parsed_int} / {len(samples)} (invalid predictions: {invalid_int})")
            if parsed_int == 0:
                print("No valid predictions to compute MAE.")
            else:
                print(f"MAE: {mae:.4f}")
            diag_metrics = {}
            bias_metrics: Dict[str, float] = {}
            bias_diag_metrics: Dict[str, float] = {}
            if diagnosis_labels:
                print("Per-diagnosis MAE:")
                for label in diagnosis_labels:
                    diag_parsed_int = int(diag_parsed.get(label, 0))
                    diag_invalid_int = int(diag_invalid.get(label, 0))
                    diag_mae = (
                        diag_abs_error[label] / diag_parsed[label] if diag_parsed.get(label, 0) else float("nan")
                    )
                    print(
                        f"  Diagnosis '{label}': parsed={diag_parsed_int} "
                        f"invalid={diag_invalid_int} mae={diag_mae:.4f}"
                    )
                    safe_name = sanitize_metric_key(label)
                    diag_metrics[f"eval/{dataset_label}/diagnosis/{safe_name}/mae"] = diag_mae
                    diag_metrics[f"eval/{dataset_label}/diagnosis/{safe_name}/parsed"] = diag_parsed_int
                    diag_metrics[f"eval/{dataset_label}/diagnosis/{safe_name}/invalid"] = diag_invalid_int

            # === Bias correction using cognitively unimpaired extremes ===
            if all_reg_records:
                ages = np.array([gt for _, gt, _ in all_reg_records], dtype=float)
                preds = np.array([pred for pred, _, _ in all_reg_records], dtype=float)
                diag_list = [diag for _, _, diag in all_reg_records]
                cu_idx = [i for i, diag in enumerate(diag_list) if diag == "No Cognitive Impairment"]
                if cu_idx:
                    cu_ages = ages[cu_idx]
                    cu_preds = preds[cu_idx]
                    bias_params = fit_bias_correction(cu_ages.tolist(), cu_preds.tolist())
                    if bias_params is not None:
                        a, b = bias_params
                        corrected = apply_bias_correction(preds, ages, a, b)
                        mae_bias = float(np.mean(np.abs(corrected - ages))) if len(ages) else float("nan")
                        print(
                            f"Bias correction (CU youngest/oldest 5%): y = {a:.4f} * x + {b:.4f} | "
                            f"MAE after correction: {mae_bias:.4f}"
                        )
                        bias_diag_result = {}
                        bias_metrics = {
                            f"eval/{dataset_label}/bias_correction/a": a,
                            f"eval/{dataset_label}/bias_correction/b": b,
                            f"eval/{dataset_label}/mae_bias_corrected": mae_bias,
                        }
                        if diagnosis_labels:
                            print("Per-diagnosis MAE after bias correction:")
                            for label in diagnosis_labels:
                                mask = np.array([diag == label for diag in diag_list])
                                if mask.any():
                                    corr_mae = float(np.mean(np.abs(corrected[mask] - ages[mask])))
                                else:
                                    corr_mae = float("nan")
                                print(f"  Diagnosis '{label}': mae_after_correction={corr_mae:.4f}")
                                safe_name = sanitize_metric_key(label)
                                bias_diag_metrics[f"eval/{dataset_label}/diagnosis/{safe_name}/mae_bias_corrected"] = (
                                    corr_mae
                                )
                                bias_diag_result[label] = corr_mae
                        result["mae_bias_corrected"] = mae_bias
                        result["bias_correction_a"] = a
                        result["bias_correction_b"] = b
                        if bias_diag_result:
                            result["per_diagnosis_mae_bias_corrected"] = bias_diag_result

            if wandb_run is not None:
                log_payload = {
                    f"eval/{dataset_label}/mae": mae,
                    f"eval/{dataset_label}/parsed": parsed_int,
                    f"eval/{dataset_label}/invalid": invalid_int,
                }
                log_payload.update(diag_metrics)
                log_payload.update(bias_metrics)
                log_payload.update(bias_diag_metrics)
                wandb_run.log(log_payload)
        else:
            accuracy = correct / total if total else 0.0
            print(f"Total samples: {total}")
            print(f"Correct predictions: {int(correct)}")
            print(f"Accuracy: {accuracy:.4f}")
            class_metrics = {}
            for name, cls_correct, cls_total in zip(class_names, class_correct, class_total):
                cls_accuracy = cls_correct / cls_total if cls_total else 0.0
                print(
                    f"  Class '{name}': total={int(cls_total)} "
                    f"correct={int(cls_correct)} accuracy={cls_accuracy:.4f}"
                )
                safe_name = sanitize_metric_key(name)
                class_metrics[f"eval/{dataset_label}/class/{safe_name}/accuracy"] = cls_accuracy
                class_metrics[f"eval/{dataset_label}/class/{safe_name}/correct"] = cls_correct
                class_metrics[f"eval/{dataset_label}/class/{safe_name}/total"] = cls_total

            if wandb_run is not None:
                log_payload = {
                    f"eval/{dataset_label}/accuracy": accuracy,
                    f"eval/{dataset_label}/correct": int(correct),
                    f"eval/{dataset_label}/total": total,
                }
                log_payload.update(class_metrics)
                wandb_run.log(log_payload)

        output_csv_path = Path(args.output_csv) if args.output_csv else adapter_dir / f"{dataset_label or dataset_path.stem}_predictions.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "is_correct"])
            writer.writeheader()
            writer.writerows(all_predictions)
        print(f"[Eval] Saved predictions to {output_csv_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ADNI VQA LoRA model")
    parser.add_argument("--model_name", default="MagicXin/Med3DVLM-Qwen-2.5-7B")
    parser.add_argument("--lora_path", default="output/adni_vqa_lora")
    parser.add_argument("--dataset_json", default="data/ADNI_VQA_with_history_test.json")
    parser.add_argument("--dataset_label", default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        action="store_true",
        help="Log evaluation metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--no_wandb",
        dest="use_wandb",
        action="store_false",
        help="Skip logging evaluation metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb_project", default="Med3DVLM", help="W&B project name for eval logging.")
    parser.add_argument("--wandb_entity", default="", help="Optional W&B entity for eval logging.")
    parser.add_argument("--wandb_run_name", default="", help="Optional W&B run name for this evaluation job.")
    parser.add_argument(
        "--output_csv",
        default="",
        help="File path to save image_id and model predictions as CSV. Defaults to <lora_path>/<dataset>_predictions.csv.",
    )
    parser.set_defaults(use_wandb=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
