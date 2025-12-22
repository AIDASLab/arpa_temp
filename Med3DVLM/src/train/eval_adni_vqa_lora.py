"""Evaluation script for ADNI VQA LoRA models."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


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

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.to(device)
    model.eval()

    dataset_path = PROJECT_ROOT / args.dataset_json
    dataset_data = load_json_dict(dataset_path)
    samples = dataset_data.get("samples", [])
    dataset_label = args.dataset_label or dataset_path.stem
    is_regression = dataset_is_regression(dataset_data)

    if get_rank() == 0:
        print(f"[Eval] Dataset '{dataset_label}' -> {dataset_path} ({len(samples)} samples)")

    world_size = get_world_size()
    rank = get_rank()

    processed = 0

    if is_regression:
        abs_error = 0.0
        parsed = 0.0
        invalid_predictions = 0.0
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
            gt_value = safe_float(sample.get("answer"))
            if gt_value is None:
                continue
            pred_value = extract_first_float(predicted)
            if pred_value is None:
                invalid_predictions += 1
            else:
                diff = abs(pred_value - gt_value)
                abs_error += diff
                parsed += 1
        else:
            is_correct = sample["answer"].lower() in predicted.lower()
            if is_correct:
                correct += 1
            total += 1

            cls_idx = class_to_idx.get(sample["answer"])
            if cls_idx is not None:
                class_total[cls_idx] += 1
                if is_correct:
                    class_correct[cls_idx] += 1

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} samples...")

    # Aggregate results across ranks
    result: Optional[Dict[str, float]] = None

    if is_regression:
        if is_dist_initialized():
            tensor = torch.tensor([abs_error, parsed, invalid_predictions], device=device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            abs_error = float(tensor[0].item())
            parsed = float(tensor[1].item())
            invalid_predictions = float(tensor[2].item())
        mae = abs_error / parsed if parsed else float("nan")
        result = {
            "task": "regression",
            "mae": mae,
            "parsed": parsed,
            "invalid": invalid_predictions,
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
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/{dataset_label}/mae": mae,
                        f"eval/{dataset_label}/parsed": parsed_int,
                        f"eval/{dataset_label}/invalid": invalid_int,
                    }
                )
        else:
            accuracy = correct / total if total else 0.0
            print(f"Total samples: {total}")
            print(f"Correct predictions: {int(correct)}")
            print(f"Accuracy: {accuracy:.4f}")
            class_metrics = {}
            for name, cls_correct, cls_total in zip(class_names, class_correct, class_total):
                cls_accuracy = cls_correct / cls_total if cls_total else 0.0
                print(
                    f"  Class '{name}': total={int(cls_total)} correct={int(cls_correct)} accuracy={cls_accuracy:.4f}"
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
    parser.set_defaults(use_wandb=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
