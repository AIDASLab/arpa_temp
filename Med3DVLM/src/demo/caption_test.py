"""Simple captioning demo on a single Med3DVLM sample with optional LoRA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore
except ImportError:  # pragma: no cover
    PeftModel = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption one of the demo volumes bundled with Med3DVLM."
    )
    parser.add_argument(
        "--model_path", default="./models/Med3DVLM-Qwen-2.5-7B", help="Base model to load."
    )
    parser.add_argument(
        "--lora_path",
        default="",
        help="Optional LoRA checkpoint directory. Leave empty to skip applying LoRA.",
    )
    parser.add_argument(
        "--data_root",
        default="./data/demo",
        help="Root directory that contains sample folders (e.g. data/demo/024421).",
    )
    parser.add_argument(
        "--sample_id",
        default="024421",
        help="Sample folder name inside data_root. Ignored when --image_path is set.",
    )
    parser.add_argument(
        "--image_path",
        default="",
        help="Direct path to a .nii.gz volume. Overrides --sample_id when provided.",
    )
    parser.add_argument(
        "--question",
        default="Describe the findings of the medical image you see.",
        help="Prompt or question to feed the model.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for the vision encoder inputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to move the model and tensors to.",
    )
    parser.add_argument(
        "--proj_out_num",
        type=int,
        default=256,
        help="Fallback projection token count when the config is missing proj_out_num.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_image_path(data_root: Path, sample_id: str, manual_path: str) -> Path:
    if manual_path:
        path = Path(manual_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provided image path '{path}' does not exist.")
        return path

    sample_dir = (data_root / sample_id).expanduser().resolve()
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory '{sample_dir}' was not found.")

    candidates = sorted(p for p in sample_dir.glob("*.nii.gz") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No .nii.gz volumes found under '{sample_dir}'. "
            "Please provide --image_path explicitly."
        )
    return candidates[0]


def dtype_from_string(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_volume(image_path: Path, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    volume = sitk.ReadImage(str(image_path))
    np_volume = sitk.GetArrayFromImage(volume).astype(np.float32, copy=False)
    np_volume = np.expand_dims(np_volume, axis=0)
    tensor = torch.from_numpy(np_volume).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def apply_lora_if_needed(model: AutoModelForCausalLM, lora_path: str):
    if not lora_path:
        return model
    if PeftModel is None:
        raise ImportError("peft package is required to load LoRA checkpoints.")
    resolved = Path(lora_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"LoRA path '{resolved}' does not exist.")
    print(f"[LoRA] Loading adapter from {resolved}")
    return PeftModel.from_pretrained(model, str(resolved))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    image_path = resolve_image_path(Path(args.data_root), args.sample_id, args.image_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device.type if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model = apply_lora_if_needed(model, args.lora_path)
    model = model.to(device)
    model.eval()

    proj_out_num = getattr(model.get_model().config, "proj_out_num", args.proj_out_num)
    image_tokens = "<im_patch>" * proj_out_num
    prompt = image_tokens + args.question

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=device)
    image_tensor = load_volume(image_path, dtype=dtype, device=device)

    with torch.inference_mode():
        outputs = model.generate(
            images=image_tensor,
            inputs=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )

    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Question: {args.question}")
    print(f"Image: {image_path}")
    print("Caption:\n", text)


if __name__ == "__main__":
    main()
