"""LoRA fine-tuning entry point for ADNI VQA datasets."""

import argparse
import json
import os
import re
import wandb 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

# Dataset registry makes it easy to plug datasets in/out later.
DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "ADNI_VQA": {
        "train": "data/ADNI_VQA_train.json",
        "eval": "data/ADNI_VQA_test.json",
    },
    "ADNI_VQA_HISTORY": {
        "train": "data/ADNI_VQA_with_history_train.json",
        "eval": "data/ADNI_VQA_with_history_test.json",
    },
    "ADNI_VQA_AGE": {
        "train": "data/ADNI_VQA_age_train.json",
        "eval": "data/ADNI_VQA_age_test.json",
    },
}


def resolve_dataset_file(key: str, split: str) -> Path:
    key = key.strip()
    if not key:
        raise ValueError("Empty dataset keyword provided.")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset keyword '{key}'. Available: {list(DATASET_REGISTRY)}")
    entry = DATASET_REGISTRY[key]
    if split not in entry:
        raise ValueError(f"Dataset '{key}' does not define a '{split}' split.")
    path = PROJECT_ROOT / entry[split]
    if not path.exists():
        raise FileNotFoundError(f"Resolved dataset file not found: {path}")
    return path


def resolve_dataset_files(keys: List[str], split: str) -> List[Path]:
    files: List[Path] = []
    for key in keys:
        if not key.strip():
            continue
        files.append(resolve_dataset_file(key, split))
    if not files:
        raise ValueError(f"No dataset files resolved for split '{split}' and keys {keys}.")
    return files


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


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def resolve_precision(preferred: str) -> str:
    preferred = preferred.lower()
    if preferred not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision option '{preferred}'.")
    if preferred == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "fp32"
    if preferred in {"fp16", "bf16"} and not torch.cuda.is_available():
        print("[train] CUDA not available, falling back to fp32 training.")
        return "fp32"
    if preferred == "bf16" and not torch.cuda.is_bf16_supported():
        print("[train] Requested bf16 precision but hardware does not support it. Falling back to fp16.")
        return "fp16"
    return preferred


class ADNIVQADataset(Dataset):
    def __init__(self, json_files: List[Path], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.samples: List[Dict] = []
        for path in json_files:
            self.samples.extend(load_json_samples(path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or "</s>"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        prompt = "You are a medical imaging assistant." + f" Question: {sample['question']}\nAnswer:"
        answer = sample["answer"]
        full_text = prompt + " " + answer + self.eos

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]
        prompt_len = prompt_ids.size(1)
        labels[0, :prompt_len] = -100

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class GenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        datasets: List[Tuple[str, Dict[str, Any]]],
        max_new_tokens: int,
        max_samples: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.max_new_tokens = max_new_tokens
        self.max_samples = max_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        if hasattr(model, "module"):
            model = model.module
        current_epoch = state.epoch or 0
        is_main_process = getattr(state, "is_world_process_zero", True)
        for label, dataset in self.datasets:
            samples = dataset.get("samples", [])
            sample_count = len(samples)
            if self.max_samples > 0:
                sample_count = min(sample_count, self.max_samples)
            results = self._run_generation_eval(model, dataset, label, current_epoch=current_epoch)
            if results is None:
                continue
            task_type = results["task"]
            if not is_main_process:
                continue
            epoch_label = int(current_epoch) if current_epoch else 0
            safe_label = label.replace(" ", "_").lower()
            if task_type == "regression":
                mae = results["mae"]
                parsed = int(results["count"])
                invalid = int(results["invalid"])
                print(
                    f"[GenerationEval][{label}][epoch {epoch_label}] mae={mae:.4f} "
                    f"(parsed={parsed}, invalid={invalid}) on {sample_count} samples."
                )
                metrics = {
                    f"eval/{safe_label}/mae": mae,
                    f"eval/{safe_label}/parsed": parsed,
                    f"eval/{safe_label}/invalid": invalid,
                    "epoch": current_epoch,
                }
            else:
                accuracy = results["accuracy"]
                correct = results["correct"]
                total = results["total"]
                print(
                    f"[GenerationEval][{label}][epoch {epoch_label}] accuracy={accuracy:.4f} "
                    f"({int(correct)}/{int(total)}) on {sample_count} samples."
                )
                metrics = {
                    f"eval/{safe_label}/accuracy": accuracy,
                    f"eval/{safe_label}/correct": correct,
                    f"eval/{safe_label}/total": total,
                    "epoch": current_epoch,
                }
            self._record_metrics(metrics, state)

            if wandb.run is not None:
                wandb.log(metrics, step=getattr(state, "global_step", None))

    def _run_generation_eval(self, model, dataset: Dict[str, Any], label: str, current_epoch: float):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        samples = dataset.get("samples", [])
        target_samples = samples
        if self.max_samples > 0:
            target_samples = target_samples[: self.max_samples]

        if not target_samples:
            if was_training:
                model.train()
            return None

        rank = get_rank()
        world_size = get_world_size()

        is_regression = dataset_is_regression(dataset)
        correct = 0.0
        total = 0.0
        abs_error = 0.0
        parsed = 0.0
        invalid = 0.0
        for idx in range(rank, len(target_samples), world_size):
            sample = target_samples[idx]
            prompt = "You are a medical imaging assistant." + f" Question: {sample['question']}\nAnswer:"
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )
            batch = {k: v.to(device) for k, v in encoded.items()}
            input_ids = batch["input_ids"]
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs=input_ids,
                        attention_mask=batch.get("attention_mask"),
                        max_new_tokens=self.max_new_tokens,
                    )
                except RuntimeError as err:
                    sample_id = sample.get("image_id") or sample.get("ptid") or "unknown_sample"
                    if get_rank() == 0:
                        print(
                            f"[GenerationEval][{label}] Skipping sample '{sample_id}' due to generation error: {err}"
                        )
                    continue
            gen_tokens = outputs[0]
            generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip() if gen_tokens.numel() else ""
            completion = extract_answer_text(generated)
            predicted = completion.split("\n")[0] if completion else ""
            if is_regression:
                gt_value = safe_float(sample.get("answer"))
                if gt_value is None:
                    continue
                pred_value = extract_first_float(predicted)
                if pred_value is None:
                    invalid += 1
                else:
                    diff = abs(pred_value - gt_value)
                    abs_error += diff
                    parsed += 1
            else:
                if sample["answer"].lower() in predicted.lower():
                    correct += 1
                total += 1

        if is_regression:
            tensor = torch.tensor([abs_error, parsed, invalid], device=device, dtype=torch.float32)
            if is_dist_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            abs_error = float(tensor[0].item())
            parsed = float(tensor[1].item())
            invalid = float(tensor[2].item())
            mae = abs_error / parsed if parsed else float("nan")
            result = {"task": "regression", "mae": mae, "count": parsed, "invalid": invalid}
        else:
            tensor = torch.tensor([correct, total], device=device, dtype=torch.float32)
            if is_dist_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            correct = float(tensor[0].item())
            total = float(tensor[1].item())
            accuracy = correct / total if total else 0.0
            result = {"task": "classification", "accuracy": accuracy, "correct": correct, "total": total}

        if was_training:
            model.train()
        return result

    def _record_metrics(self, metrics: Dict[str, float], state) -> None:
        metrics_with_step = dict(metrics)
        metrics_with_step["step"] = getattr(state, "global_step", 0)
        if hasattr(state, "log_history"):
            state.log_history.append(metrics_with_step)


def parse_dataset_keys(value: str) -> List[str]:
    return [k.strip() for k in value.split(",") if k.strip()]


def train(args: argparse.Namespace) -> None:
    precision = resolve_precision(args.precision)
    use_fp16 = precision == "fp16"
    use_bf16 = precision == "bf16"
    torch_dtype = torch.float32
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_keys = parse_dataset_keys(args.train_dataset_keys)
    eval_keys = parse_dataset_keys(args.eval_dataset_keys) if args.eval_dataset_keys else train_keys

    train_files = resolve_dataset_files(train_keys, split="train")
    eval_files = resolve_dataset_files(eval_keys, split="eval")

    train_dataset = ADNIVQADataset(train_files, tokenizer, max_length=args.max_length)
    eval_dataset = ADNIVQADataset(eval_files, tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )

    if args.finetune_mode == "lora":
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    elif args.finetune_mode != "full":
        raise ValueError(f"Unknown finetune mode '{args.finetune_mode}'. Expected 'lora' or 'full'.")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    report_to = "wandb" if args.use_wandb else "none"
    run_name = args.wandb_run_name if args.use_wandb and args.wandb_run_name else None

    max_grad_norm = args.max_grad_norm
    if use_fp16 and max_grad_norm and max_grad_norm > 0:
        print(
            "[train] Disabling gradient clipping for fp16 precision because Torch's GradScaler "
            "cannot unscale fp16 gradients when Accelerate clips them.",
            flush=True,
        )
        max_grad_norm = 0.0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=5,
        evaluation_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
        report_to=report_to,
        run_name=run_name,
        max_grad_norm=max_grad_norm,
    )

    callbacks = []
    if args.generation_eval_per_epoch:
        generation_eval_datasets: List[Tuple[str, Dict[str, Any]]] = []
        for key in eval_keys:
            if not key.strip():
                continue
            path = resolve_dataset_file(key, split="eval")
            generation_eval_datasets.append((key, load_json_dict(path)))
        if generation_eval_datasets:
            callbacks.append(
                GenerationEvalCallback(
                    tokenizer=tokenizer,
                    datasets=generation_eval_datasets,
                    max_new_tokens=args.generation_eval_max_new_tokens,
                    max_samples=args.generation_eval_max_samples,
                )
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for ADNI VQA datasets")
    parser.add_argument("--model_name", default="MagicXin/Med3DVLM-Qwen-2.5-7B")
    parser.add_argument("--train_dataset_keys", default="ADNI_VQA,ADNI_VQA_HISTORY",
                        help="Comma-separated dataset keywords for training.")
    parser.add_argument("--eval_dataset_keys", default="",
                        help="Comma-separated dataset keywords for evaluation (defaults to train keys).")
    parser.add_argument("--finetune_mode", choices=["lora", "full"], default="lora",
                        help="Choose 'lora' for parameter-efficient tuning or 'full' for full fine-tuning.")
    parser.add_argument("--output_dir", default="output/adni_vqa_lora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", default="cosine",
                        help="Learning rate scheduler type (e.g., cosine, linear, polynomial).")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Computation precision for training. 'auto' picks bf16 when available, otherwise fp16 on CUDA or fp32 on CPU.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Automatically disabled when training in fp16 due to Torch GradScaler limitations.",
    )
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA application.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model/tokenizer to skip HF prompts.",
    )
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging via Hugging Face Trainer.",
    )
    parser.add_argument(
        "--no_wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        default="Med3DVLM",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        default="",
        help="Weights & Biases entity (team/user).",
    )
    parser.add_argument(
        "--wandb_run_name",
        default="",
        help="Optional W&B run name (defaults to HF auto-generated).",
    )
    parser.add_argument(
        "--generation_eval_per_epoch",
        dest="generation_eval_per_epoch",
        action="store_true",
        help="Run text-generation evaluation after each training epoch.",
    )
    parser.add_argument(
        "--no_generation_eval_per_epoch",
        dest="generation_eval_per_epoch",
        action="store_false",
        help="Skip text-generation evaluation after each epoch.",
    )
    parser.add_argument(
        "--generation_eval_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate during per-epoch evaluation.",
    )
    parser.add_argument(
        "--generation_eval_max_samples",
        type=int,
        default=0,
        help="Limit number of samples for generation eval (0 means all available eval samples).",
    )
    parser.set_defaults(use_wandb=True, generation_eval_per_epoch=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
