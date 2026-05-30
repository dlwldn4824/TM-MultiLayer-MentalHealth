"""Phase D/F/G training: LoRA SFT, DPO, distillation student (smoke/mini/full)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.training_io import read_jsonl
from src.training_paths import training_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_hf_model(ollama_model: str, cfg: dict[str, Any]) -> str:
    mapping = cfg.get("training", {}).get("hf_model_map", {})
    return mapping.get(ollama_model, ollama_model)


def _training_deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def _format_text_alpaca(example: dict[str, Any]) -> str:
    inst = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


def _load_sft_dataset(path: Path, max_samples: int | None) -> Any:
    from datasets import Dataset

    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]
    texts = [_format_text_alpaca(r) for r in rows]
    return Dataset.from_dict({"text": texts})


def _load_dpo_dataset(path: Path, max_samples: int | None) -> Any:
    from datasets import Dataset

    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]
    return Dataset.from_dict(
        {
            "prompt": [r["prompt"] for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
        }
    )


def run_lora_training(
    model_name: str,
    scale: str,
    cfg: dict[str, Any],
) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    paths = training_paths(cfg)
    tcfg = cfg["training"]
    lora_cfg = tcfg["lora"]
    hf_id = resolve_hf_model(model_name, cfg)

    train_path = paths["sft"] / "alpaca_flat_train.jsonl"
    if not train_path.exists():
        train_path = paths["sft"] / "alpaca_flat.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"No SFT train file at {train_path}. Run dataset_builder first.")

    max_samples = None
    max_steps = lora_cfg.get("max_steps_full", 500)
    if scale == "smoke":
        max_samples = tcfg.get("smoke_max_samples", 20)
        max_steps = lora_cfg.get("max_steps_smoke", 2)
    elif scale == "mini":
        max_samples = tcfg.get("mini_max_samples", 100)
        max_steps = lora_cfg.get("max_steps_mini", 20)

    dataset = _load_sft_dataset(train_path, max_samples)
    out_dir = paths["checkpoints"] / f"lora_{model_name.replace(':', '_')}_{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = bool(lora_cfg.get("use_bf16")) and torch.cuda.is_available()
    use_qlora = bool(lora_cfg.get("use_qlora")) and torch.cuda.is_available()

    model_kwargs: dict[str, Any] = {}
    if use_qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs,
    )
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=lora_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=lora_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=lora_cfg.get("learning_rate", 2e-4),
        bf16=use_bf16,
        logging_steps=1,
        save_steps=max_steps,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {"mode": "lora", "model": model_name, "hf_id": hf_id, "scale": scale, "max_steps": max_steps}
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("LoRA checkpoint saved to %s", out_dir)
    return out_dir


def run_dpo_training(model_name: str, scale: str, cfg: dict[str, Any]) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    paths = training_paths(cfg)
    tcfg = cfg["training"]
    dpo_cfg = tcfg["dpo"]
    hf_id = resolve_hf_model(model_name, cfg)

    train_path = paths["dpo"] / "dpo_pairs_train.jsonl"
    if not train_path.exists():
        train_path = paths["dpo"] / "dpo_pairs.jsonl"
    if not train_path.exists():
        raise FileNotFoundError("No DPO dataset. Run dataset_builder --stage dpo first.")

    max_samples = None
    max_steps = dpo_cfg.get("max_steps_full", 300)
    if scale == "smoke":
        max_samples = tcfg.get("smoke_max_samples", 20)
        max_steps = dpo_cfg.get("max_steps_smoke", 2)
    elif scale == "mini":
        max_samples = tcfg.get("mini_max_samples", 100)
        max_steps = dpo_cfg.get("max_steps_mini", 20)

    dataset = _load_dpo_dataset(train_path, max_samples)
    out_dir = paths["checkpoints"] / f"dpo_{model_name.replace(':', '_')}_{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"),
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_steps=max_steps,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        beta=dpo_cfg.get("beta", 0.1),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {"mode": "dpo", "model": model_name, "hf_id": hf_id, "scale": scale}
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def run_distill_training(model_name: str, scale: str, cfg: dict[str, Any]) -> Path:
    """Student SFT on teacher outputs (same as LoRA on student_sft split)."""
    paths = training_paths(cfg)
    student_train = paths["distillation"] / "student_sft_train.jsonl"
    if student_train.exists():
        import shutil

        backup = paths["sft"] / "alpaca_flat.jsonl"
        if not backup.exists():
            shutil.copy(student_train, paths["sft"] / "alpaca_flat.jsonl")
        shutil.copy(student_train, paths["sft"] / "alpaca_flat_train.jsonl")
    out = run_lora_training(model_name, scale, cfg)
    meta_path = out / "training_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mode"] = "distill"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def write_stub_checkpoint(mode: str, model_name: str, scale: str, reason: str) -> Path:
    paths = training_paths()
    out_dir = paths["checkpoints"] / f"{mode}_{model_name.replace(':', '_')}_{scale}_stub"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"mode": mode, "model": model_name, "scale": scale, "stub": True, "reason": reason}
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.warning("Wrote stub checkpoint (%s): %s", reason, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA / DPO / distillation training")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag (mapped to HF)")
    parser.add_argument("--mode", choices=("lora", "dpo", "distill"), default="lora")
    parser.add_argument("--scale", choices=("smoke", "mini", "full"), default="smoke")
    args = parser.parse_args()

    cfg = load_config()
    if not _training_deps_available():
        write_stub_checkpoint(args.mode, args.model, args.scale, "training deps not installed")
        return

    try:
        if args.mode == "lora":
            run_lora_training(args.model, args.scale, cfg)
        elif args.mode == "dpo":
            run_dpo_training(args.model, args.scale, cfg)
        else:
            run_distill_training(args.model, args.scale, cfg)
    except Exception as e:
        logger.exception("Training failed: %s", e)
        write_stub_checkpoint(args.mode, args.model, args.scale, str(e))


if __name__ == "__main__":
    main()
