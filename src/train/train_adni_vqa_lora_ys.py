"""LoRA fine-tuning entry point for ADNI VQA datasets."""

import argparse
import json
import math
import os
import re
import statistics
import wandb 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

# Dataset registry makes it easy to plug datasets in/out later.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Classification
    "ADNI_VQA": {
        "train": "data/ADNI_VQA_train.json",
        "eval": "data/ADNI_VQA_test.json",
    },
    "ADNI_VQA_HISTORY": {
        "train": [
            "data/ADNI_VQA_train.json",
            "data/ADNI_VQA_with_history_train.json",
        ],
        "eval": "data/ADNI_VQA_test.json",
    },
    "ADNI_VQA_HISTORY_UPDATED": {
        "train": [
            "data/ADNI_VQA_train.json",
            "data/ADNI_VQA_with_history_train.json",
            "data/ADNI_VQA_with_updated_history_train.json",
        ],
        "eval": "data/ADNI_VQA_test.json",
    },
    "ADNI_VQA_v2": {
        "train": "data/ADNI_VQA_train_v2.json",
        "eval": "data/ADNI_VQA_test_v2.json",
    },
    "ADNI_VQA_CVRF_GDS": {
        "train": "data/ADNI_VQA_CVRF_GDS_train.json",
        "eval": "data/ADNI_VQA_CVRF_GDS_test.json",
    },
    "ADNI_VQA_CVRF_GDS_AGE": {
        "train": "data/ADNI_VQA_CVRF_GDS_AGE_train.json",
        "eval": "data/ADNI_VQA_CVRF_GDS_AGE_test.json",
    },
    # Regression
    "ADNI_VQA_AGE": {
        "train": "data/ADNI_VQA_age_train.json",
        "eval": "data/ADNI_VQA_age_test.json",
    },
    "ADNI_VQA_AGE_HISTORY": {
        "train": [
            "data/ADNI_VQA_age_train.json",
            "data/ADNI_VQA_age_history_train.json",
        ],
        "eval": "data/ADNI_VQA_age_test.json",
    },
    "ADNI_VQA_AGE_HISTORY_UPDATED": {
        "train": [
            "data/ADNI_VQA_age_train.json",
            "data/ADNI_VQA_age_history_train.json",
            "data/ADNI_VQA_age_updated_history_train.json",
        ],
        "eval": "data/ADNI_VQA_age_test.json",
    },
}


def _resolve_dataset_split_paths(key: str, split: str) -> List[Path]:
    key = key.strip()
    if not key:
        raise ValueError("Empty dataset keyword provided.")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset keyword '{key}'. Available: {list(DATASET_REGISTRY)}")
    entry = DATASET_REGISTRY[key]
    if split not in entry:
        raise ValueError(f"Dataset '{key}' does not define a '{split}' split.")
    spec = entry[split]
    if isinstance(spec, (list, tuple)):
        rel_paths = [p for p in spec if str(p).strip()]
    else:
        rel_paths = [spec]
    paths: List[Path] = []
    for rel in rel_paths:
        path = PROJECT_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Resolved dataset file not found: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"Dataset '{key}' split '{split}' did not resolve to any files.")
    return paths


def resolve_dataset_file(key: str, split: str) -> Path:
    paths = _resolve_dataset_split_paths(key, split)
    if len(paths) != 1:
        raise ValueError(
            f"Dataset '{key}' split '{split}' resolves to multiple files; use resolve_dataset_files instead."
        )
    return paths[0]


def resolve_dataset_files(keys: List[str], split: str) -> List[Path]:
    files: List[Path] = []
    for key in keys:
        if not key.strip():
            continue
        files.extend(_resolve_dataset_split_paths(key, split))
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


def unfreeze_modules_by_keyword(model: nn.Module, keywords: List[str], label: str) -> int:
    """Set requires_grad=True for parameters whose names contain any keyword."""
    total = 0
    for name, param in model.named_parameters():
        if not any(k in name for k in keywords):
            continue
        if not param.requires_grad:
            param.requires_grad = True
        total += param.numel()
    print(f"[train] Unfroze {total:,} parameters for {label} (keywords={keywords})")
    return total


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


def compute_regression_stats(dicts: List[Dict]) -> Optional[Dict[str, float]]:
    values: List[float] = []
    for data in dicts:
        for sample in data.get("samples", []) or []:
            val = safe_float(sample.get("answer"))
            if val is not None and math.isfinite(val):
                values.append(val)
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) > 1:
        std = statistics.pstdev(values)
    else:
        std = 1.0
    if std <= 0 or not math.isfinite(std):
        std = 1.0
    return {"mean": float(mean), "std": float(std), "count": float(len(values))}


def normalize_value(value: float, stats: Optional[Dict[str, float]]) -> float:
    if not stats:
        return value
    std = stats.get("std") or 1.0
    if std == 0:
        std = 1.0
    mean = stats.get("mean", 0.0)
    return (value - mean) / std


def denormalize_value(value: float, stats: Optional[Dict[str, float]]) -> float:
    if not stats:
        return value
    std = stats.get("std") or 1.0
    if std == 0:
        std = 1.0
    mean = stats.get("mean", 0.0)
    return value * std + mean


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


def make_regression_log_suffix(loss_type: str, mle_variance: float, huber_beta: float) -> str:
    """Create a wandb-friendly suffix containing loss type + hyperparameters."""
    if not loss_type:
        return ""
    lt = loss_type.strip().lower()
    parts = [f"loss={lt}"]
    if lt == "gaussian" and mle_variance is not None:
        parts.append(f"sigma2={mle_variance:g}")
    elif lt == "huber" and huber_beta is not None:
        parts.append(f"beta={huber_beta:g}")
    raw = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", raw)


class CausalLMWithRegressionHead(nn.Module):
    """
    Wraps a Causal LM (possibly PEFT/LoRA-wrapped) with a regression head.

    - loss_type: 'gaussian' | 'mae' | 'huber'
    - mle_variance: Gaussian NLL에서 사용하는 σ^2
    - huber_beta: Huber(SmoothL1) loss의 beta 파라미터
    """
    def __init__(
        self,
        backbone: nn.Module,
        loss_type: str = "gaussian",
        mle_variance: float = 1.0,
        huber_beta: float = 1.0,
        normalization: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.loss_type = loss_type
        self.mle_variance = mle_variance
        self.huber_beta = huber_beta
        self.normalization = normalization or {"enabled": False}

        hidden_size = infer_hidden_size(backbone)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def _compute_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        loss_type에 따라 적절한 regression loss를 계산.
        preds, labels는 같은 dtype이라고 가정 (forward 쪽에서 맞춰줌).
        """
        if self.loss_type == "gaussian":
            return gaussian_mle_loss(preds, labels, variance=self.mle_variance)
        elif self.loss_type == "mae":
            # MAE = L1
            return F.l1_loss(preds, labels)
        elif self.loss_type == "huber":
            # SmoothL1(Huber) loss
            return F.smooth_l1_loss(preds, labels, beta=self.huber_beta)
        else:
            raise ValueError(f"Unknown regression loss_type '{self.loss_type}'")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Backbone은 labels 없이 호출 (자기 loss는 안 쓰니까)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )

        # dict vs ModelOutput 모두 처리
        if isinstance(outputs, dict):
            hidden_states = outputs["hidden_states"]
        else:
            hidden_states = outputs.hidden_states

        pooled = select_last_hidden(hidden_states, attention_mask)
        head_dtype = next(self.regression_head.parameters()).dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(head_dtype)
        preds = self.regression_head(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            # device/dtype 정렬
            labels = labels.to(preds.device)
            if labels.dtype != torch.float32:
                labels = labels.float()

            preds_for_loss = preds.to(labels.dtype)
            loss = self._compute_loss(preds_for_loss, labels)

        # Trainer가 기대하는 형식으로 loss/logits 집어넣기
        if isinstance(outputs, dict):
            out = dict(outputs)
            out["logits"] = preds.unsqueeze(-1)
            out["loss"] = loss
            return out
        else:
            outputs.logits = preds.unsqueeze(-1)
            outputs.loss = loss
            return outputs

    # So classification path & generation eval can still call .generate()
    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """
        Save LoRA adapters (via backbone.save_pretrained) and the regression head
        into a single directory so that eval can reconstruct the same wrapper.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # 1) LoRA / backbone 쪽은 기존 HF/PEFT 포맷 그대로 저장
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(str(save_directory))
        else:
            torch.save(
                self.backbone.state_dict(),
                save_directory / "backbone_state_dict.bin",
            )

        # 2) Regression head 파라미터 저장
        head_path = save_directory / "regression_head.bin"
        torch.save(self.regression_head.state_dict(), head_path)

        # 3) 간단한 config (loss_type, mle_variance, huber_beta 저장)
        cfg = {
            "loss_type": self.loss_type,
            "mle_variance": float(self.mle_variance),
            "huber_beta": float(self.huber_beta),
        }
        cfg_path = save_directory / "regression_head_config.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        norm_cfg_path = save_directory / "regression_norm_stats.json"
        with norm_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(self.normalization or {"enabled": False}, f, ensure_ascii=False, indent=2)


def select_last_hidden(hidden_states, attention_mask: Optional[torch.Tensor]):
    """
    Pool the last valid hidden state according to attention_mask.

    - hidden_states: can be either a single 3D tensor [B, T, H]
                     or a tuple/list of such tensors (all layer outputs).
    - attention_mask: may be a Tensor or a (Tensor,) tuple/list due to wrappers.
    """

    if isinstance(hidden_states, (tuple, list)):
        if len(hidden_states) == 0:
            raise ValueError("hidden_states is an empty tuple/list.")
        hidden_states = hidden_states[-1]

    if attention_mask is None:
        return hidden_states[:, -1, :]
    
    if isinstance(attention_mask, (tuple, list)):
        if len(attention_mask) == 0:
            return hidden_states[:, -1, :]
        attention_mask = attention_mask[0]

    if not torch.is_tensor(attention_mask):
        raise TypeError(f"attention_mask must be a Tensor, got {type(attention_mask)}")

    if attention_mask.dim() != 2:
        raise ValueError(
            f"Attention mask must be 2D for regression pooling, got shape {attention_mask.shape}."
        )

    attention_mask = attention_mask.to(hidden_states.device)
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, lengths]


def gaussian_mle_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    variance: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood with fixed variance.
    """
    preds = predictions.to(torch.float32)
    targs = targets.to(torch.float32)
    var_tensor = torch.full_like(preds, fill_value=variance, dtype=preds.dtype)
    var_tensor = var_tensor.clamp_min(eps)
    return F.gaussian_nll_loss(preds, targs, var_tensor, full=True)


def infer_hidden_size(model: AutoModelForCausalLM) -> int:
    for attr in ("hidden_size", "n_embd", "dim"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int):
            return value
    raise ValueError("Unable to infer hidden size from model config.")


class ADNIVQADataset(Dataset):
    def __init__(
        self,
        json_files: List[Path],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        is_regression: bool = False,
        normalization: Optional[Dict[str, float]] = None,
    ):
        self.samples: List[Dict] = []
        for path in json_files:
            self.samples.extend(load_json_samples(path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or "</s>"
        self.is_regression = is_regression
        self.normalization = normalization if normalization and normalization.get("std") else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        prompt = "You are a medical imaging assistant." + f" Question: {sample['question']}\nAnswer:"
        if self.is_regression:
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            target_value = safe_float(sample.get("answer"))
            if target_value is None:
                raise ValueError("Regression sample missing numeric answer.")
            if self.normalization is not None:
                target_value = normalize_value(target_value, self.normalization)
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": torch.tensor(target_value, dtype=torch.float32),
            }

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
        regression_table_suffix: str = "",
        regression_normalized: bool = False,
        eval_interval_epochs: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.max_new_tokens = max_new_tokens
        self.max_samples = max_samples
        self.regression_table_suffix = regression_table_suffix
        self.regression_normalized = regression_normalized
        self.eval_interval_epochs = max(1, int(eval_interval_epochs))
        # ==== (YS) Recoding best metric ====
        self.best_metrics: Dict[str, float] = {}
        # ===================================

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return

        if hasattr(model, "module"):
            model = model.module
        current_epoch = state.epoch or 0
        epoch_label = int(current_epoch) if current_epoch else 0
        if epoch_label % self.eval_interval_epochs != 0:
            return
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
                # ==== (YS) Save best checkpoint based on MAE ====
                self._maybe_save_best_checkpoint(
                    model=model,
                    output_dir=args.output_dir,
                    safe_label=safe_label,
                    metric_name="mae",
                    metric_value=mae,
                    minimize=True,
                    epoch_label=epoch_label,
                )
                # ================================================
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
                # ==== (YS) Save best checkpoint based on Accuracy ====
                self._maybe_save_best_checkpoint(
                    model=model,
                    output_dir=args.output_dir,
                    safe_label=safe_label,
                    metric_name="accuracy",
                    metric_value=accuracy,
                    minimize=False,
                    epoch_label=epoch_label,
                )
                # =====================================================
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
        dataset_stats = dataset.get("regression_stats") if is_regression else None
        model_norm_stats = None
        if is_regression and self.regression_normalized and hasattr(model, "normalization"):
            model_norm_stats = getattr(model, "normalization")
        correct = 0.0
        total = 0.0
        abs_error = 0.0
        parsed = 0.0
        invalid = 0.0
        
        # ==== (YS) Broadcast logging intent so every rank can join the sample gather. ====
        has_local_wandb = wandb.run is not None
        if is_dist_initialized() and world_size > 1:
            flag_tensor = torch.tensor([1 if (rank == 0 and has_local_wandb) else 0], device=device, dtype=torch.int)
            dist.broadcast(flag_tensor, src=0)
            collect_examples = bool(flag_tensor.item())
        else:
            collect_examples = has_local_wandb
        max_table_rows = 128
        sample_rows: List[Tuple[str, str, str]] = []
        # =================================================================================
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

            if is_regression:
                real_model = model
                if hasattr(real_model, "module"):
                    real_model = real_model.module

                assert hasattr(real_model, "backbone") and hasattr(real_model, "regression_head"), \
                    "Expected CausalLMWithRegressionHead wrapper for regression task."

                with torch.no_grad():
                    outputs = real_model.backbone(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    if isinstance(outputs, dict):
                        all_hidden = outputs["hidden_states"]
                    else:
                        all_hidden = outputs.hidden_states
                    hidden_states = all_hidden[-1]
                    pooled = select_last_hidden(hidden_states, batch.get("attention_mask"))
                    head_dtype = next(real_model.regression_head.parameters()).dtype
                    if pooled.dtype != head_dtype:
                        pooled = pooled.to(head_dtype)
                    pred_tensor = real_model.regression_head(pooled).squeeze(-1)
                pred_value_raw = float(pred_tensor.item())
                if self.regression_normalized:
                    stats_for_denorm = dataset_stats or model_norm_stats
                    pred_value = denormalize_value(pred_value_raw, stats_for_denorm)
                else:
                    pred_value = pred_value_raw
                if not math.isfinite(pred_value):
                    invalid += 1
                    generated = ""
                    predicted = ""
                else:
                    generated = f"{pred_value:.4f}"
                    predicted = generated
            else:
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
                generated = (
                    self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip() if gen_tokens.numel() else ""
                )
                completion = extract_answer_text(generated)
                predicted = completion.split("\n")[0] if completion else ""

            # ==== (YS) Collect examples for logging ====
            if collect_examples and len(sample_rows) < max_table_rows:
                gt_text = sample.get("answer", "")
                sample_rows.append((prompt, str(gt_text), generated))
            # ===========================================
            if is_regression:
                gt_value = safe_float(sample.get("answer"))
                if gt_value is None or not math.isfinite(pred_value):
                    continue
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

        # ==== (YS) Log sample generations to wandb ====
        if collect_examples:
            rows_to_log: List[Tuple[str, str, str]]
            if is_dist_initialized() and world_size > 1:
                gathered_rows: List[List[Tuple[str, str, str]]] = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_rows, sample_rows)
                if rank == 0:
                    merged: List[Tuple[str, str, str]] = []
                    for rows in gathered_rows:
                        if rows:
                            merged.extend(rows)
                    rows_to_log = merged[:max_table_rows]
                else:
                    rows_to_log = []
            else:
                rows_to_log = sample_rows[:max_table_rows]

            if rows_to_log and rank == 0:
                safe_label = label.replace(" ", "_").lower()
                table = wandb.Table(columns=["prompt", "ground_truth", "generated"])
                for prompt_text, gt_text, gen_text in rows_to_log:
                    table.add_data(prompt_text, gt_text, gen_text)
                table_key = f"eval/{safe_label}/samples"
                if is_regression and self.regression_table_suffix:
                    table_key = f"{table_key}_{self.regression_table_suffix}"
                wandb.log({table_key: table, "epoch": current_epoch})
        # ===============================================

        if was_training:
            model.train()
        return result

    def _record_metrics(self, metrics: Dict[str, float], state) -> None:
        metrics_with_step = dict(metrics)
        metrics_with_step["step"] = getattr(state, "global_step", 0)
        if hasattr(state, "log_history"):
            state.log_history.append(metrics_with_step)

    # ==== (YS) Method to save checkpoint if best metric improved ====
    def _maybe_save_best_checkpoint(
        self,
        model,
        output_dir: str,
        safe_label: str,
        metric_name: str,
        metric_value: float,
        minimize: bool,
        epoch_label: int,
    ) -> None:
        if metric_value is None or (isinstance(metric_value, float) and math.isnan(metric_value)):
            return
        key = f"{safe_label}:{metric_name}"
        best_value = self.best_metrics.get(key)
        improved = best_value is None or ((metric_value < best_value) if minimize else (metric_value > best_value))
        if not improved:
            return
        self.best_metrics[key] = metric_value
        best_dir = Path(output_dir) / "best_checkpoints" / safe_label
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        info = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "epoch": epoch_label,
        }
        with (best_dir / "best_metric.json").open("w", encoding="utf-8") as f:
            json.dump(info, f)
        print(
            f"[GenerationEval][{safe_label}] New best {metric_name}={metric_value:.4f} at epoch {epoch_label}; "
            f"checkpoint saved to {best_dir}."
        )
    # ================================================================


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

    train_dicts = [load_json_dict(path) for path in train_files]
    eval_dicts = [load_json_dict(path) for path in eval_files]

    train_flags = [dataset_is_regression(meta) for meta in train_dicts]
    eval_flags = [dataset_is_regression(meta) for meta in eval_dicts]
    if any(train_flags) and not all(train_flags):
        raise ValueError("Mixed task types in training datasets are not supported.")
    if any(eval_flags) and not all(eval_flags):
        raise ValueError("Mixed task types in evaluation datasets are not supported.")
    train_is_regression = all(train_flags)
    eval_is_regression = all(eval_flags)
    if train_is_regression != eval_is_regression:
        raise ValueError("Training and evaluation datasets must agree on task type (regression vs classification).")
    is_regression_task = train_is_regression

    regression_normalized = bool(is_regression_task and args.normalized)
    train_norm_stats: Optional[Dict[str, float]] = None
    if regression_normalized:
        train_norm_stats = compute_regression_stats(train_dicts)
        if not train_norm_stats:
            raise ValueError("Unable to compute normalization stats for regression datasets.")
        print(
            f"[train] Normalizing regression targets: mean={train_norm_stats['mean']:.4f}, "
            f"std={train_norm_stats['std']:.4f} (count={int(train_norm_stats['count'])})"
        )

    train_dataset = ADNIVQADataset(
        train_files,
        tokenizer,
        max_length=args.max_length,
        is_regression=is_regression_task,
        normalization=train_norm_stats if regression_normalized else None,
    )
    eval_dataset = ADNIVQADataset(
        eval_files,
        tokenizer,
        max_length=args.max_length,
        is_regression=is_regression_task,
        normalization=train_norm_stats if regression_normalized else None,
    )

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
        vision_keywords = [m.strip() for m in args.vision_full_finetune_modules.split(",") if m.strip()]
        projector_keywords = [m.strip() for m in args.visual_projector_modules.split(",") if m.strip()]
        if args.vision_full_finetune and vision_keywords:
            unfreeze_modules_by_keyword(model, vision_keywords, label="vision encoder")
        if args.visual_projector_full_finetune and projector_keywords:
            unfreeze_modules_by_keyword(model, projector_keywords, label="visual projector")
    elif args.finetune_mode != "full":
        raise ValueError(f"Unknown finetune mode '{args.finetune_mode}'. Expected 'lora' or 'full'.")

    if is_regression_task:
        model = CausalLMWithRegressionHead(
            model,
            loss_type=args.regression_loss_type,
            mle_variance=args.regression_mle_variance,
            huber_beta=args.regression_huber_beta,
            normalization=(
                {
                    "enabled": True,
                    "mean": train_norm_stats["mean"],
                    "std": train_norm_stats["std"],
                    "count": train_norm_stats["count"],
                }
                if train_norm_stats and regression_normalized
                else {"enabled": False}
            ),
        )
        data_collator = default_data_collator
    else:
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

    regression_table_suffix = ""
    if is_regression_task:
        regression_table_suffix = make_regression_log_suffix(
            args.regression_loss_type,
            args.regression_mle_variance,
            args.regression_huber_beta,
        )

    callbacks = []
    if args.generation_eval_per_epoch:
        generation_eval_datasets: List[Tuple[str, Dict[str, Any]]] = []
        for key in eval_keys:
            if not key.strip():
                continue
            path = resolve_dataset_file(key, split="eval")
            data = load_json_dict(path)
            if is_regression_task and regression_normalized and train_norm_stats:
                data["regression_stats"] = train_norm_stats
            generation_eval_datasets.append((key, data))
        if generation_eval_datasets:
            callbacks.append(
                GenerationEvalCallback(
                    tokenizer=tokenizer,
                    datasets=generation_eval_datasets,
                    max_new_tokens=args.generation_eval_max_new_tokens,
                    max_samples=args.generation_eval_max_samples,
                    regression_table_suffix=regression_table_suffix,
                    regression_normalized=regression_normalized,
                    eval_interval_epochs=args.generation_eval_interval_epochs,
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
    parser.add_argument(
        "--regression_loss_type",
        choices=["gaussian", "mae", "huber"],
        default="gaussian",
        help="Which loss to use for regression tasks. "
             "'gaussian' = Gaussian NLL (scaled MSE), "
             "'mae' = L1 loss, "
             "'huber' = SmoothL1 loss.",
    )
    parser.add_argument(
        "--regression_mle_variance",
        type=float,
        default=1.0,
        help="Fixed variance σ^2 used in Gaussian NLL when regression_loss_type='gaussian'.",
    )
    parser.add_argument(
        "--regression_huber_beta",
        type=float,
        default=1.0,
        help="Beta parameter for Huber(SmoothL1) loss when regression_loss_type='huber'.",
    )
    parser.add_argument(
        "--normalized",
        dest="normalized",
        action="store_true",
        help="Normalize regression targets using train-set mean/std before training.",
    )
    parser.add_argument(
        "--no_normalized",
        dest="normalized",
        action="store_false",
        help="Disable regression target normalization.",
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
        "--vision_full_finetune",
        action="store_true",
        help="When using LoRA, unfreeze the vision encoder modules for full fine-tuning.",
    )
    parser.add_argument(
        "--vision_full_finetune_modules",
        default="vision_tower,vision_encoder,vision_model,visual_encoder",
        help="Comma-separated keywords to match vision encoder parameters to unfreeze.",
    )
    parser.add_argument(
        "--visual_projector_full_finetune",
        action="store_true",
        help="When using LoRA, unfreeze the visual/multimodal projector modules for full fine-tuning.",
    )
    parser.add_argument(
        "--visual_projector_modules",
        default="mm_projector,multimodal_projector,visual_projector,vision_projector,vision_proj,visual_proj",
        help="Comma-separated keywords to match projector parameters to unfreeze.",
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
    parser.add_argument(
        "--generation_eval_interval_epochs",
        type=int,
        default=1,
        help="Run generation evaluation every N epochs (defaults to every epoch).",
    )
    parser.set_defaults(use_wandb=True, generation_eval_per_epoch=True, normalized=False)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
