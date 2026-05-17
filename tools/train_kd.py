import argparse
import datetime
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import DenoiseDataset, PairedDataset
from src.engine.kd_trainer import OnlineKDTrainer
from src.models import build_model


def get_args():
    parser = argparse.ArgumentParser(description="Online Multi-Teacher KD Training for Denoise")
    parser.add_argument("--config", type=str, required=True, help="Path to KD train config file")
    parser.add_argument("--data_config", type=str, help="Path to data config file")
    parser.add_argument("--work_dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, help="Path to KD checkpoint to resume")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    return parser.parse_args()


def infer_default_data_config_path(config: dict) -> str | None:
    if config.get("task") == "denoise":
        return "configs/data/denoise.yaml"
    return None


def load_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg_path = args.data_config or config.get("data_config_path") or infer_default_data_config_path(config)
    if data_cfg_path and os.path.exists(data_cfg_path):
        with open(data_cfg_path, "r") as f:
            config["data_config"] = yaml.safe_load(f)
        config["data_config_path"] = data_cfg_path
    else:
        config["data_config"] = {}
        config["data_config_path"] = None

    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
    return config


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_checkpoint_into_model(model, checkpoint_path, label, strict=True):
    if not checkpoint_path:
        raise ValueError(f"{label} checkpoint path is not set.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = normalize_state_dict(state_dict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if strict:
            raise
        model.load_state_dict(state_dict, strict=False)
    return model


def build_teachers(config):
    teachers = {}
    teacher_cfgs = config.get("kd", {}).get("teachers", {})
    for teacher_name, spec in teacher_cfgs.items():
        if not spec.get("enabled", True):
            continue

        config_path = spec.get("config")
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(f"Teacher config not found for {teacher_name}: {config_path}")

        with open(config_path, "r") as f:
            teacher_train_config = yaml.safe_load(f)

        teacher = build_model(teacher_train_config["model"])
        teacher = load_checkpoint_into_model(
            teacher,
            spec.get("checkpoint"),
            label=f"Teacher '{teacher_name}'",
            strict=spec.get("strict_load", True),
        )
        teacher.eval()
        teachers[teacher_name] = teacher

    if not teachers:
        raise ValueError("No enabled teachers configured for KD training.")
    return teachers


def maybe_load_student_pretrained(student, config, resume):
    if resume:
        return student
    pretrained_path = config["train"].get("pretrained_path")
    if pretrained_path:
        print(f"Loading student pretrained checkpoint: {pretrained_path}")
        student = load_checkpoint_into_model(student, pretrained_path, label="Student pretrained", strict=True)
    else:
        print("[Warning] Student pretrained_path is not set. KD will start from scratch.")
    return student


def build_dataloaders(config):
    if config.get("task") != "denoise":
        raise ValueError("tools/train_kd.py currently supports task: denoise only.")

    train_ds = DenoiseDataset(
        dataset_root=config["data"]["train_root"],
        scale_factor=1,
        patch_size=config["train"]["patch_size"],
        is_train=True,
        config=config.get("data_config", {}),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    if config["data"].get("val_hr_root") and config["data"].get("val_lr_root"):
        val_ds = PairedDataset(config["data"]["val_hr_root"], config["data"]["val_lr_root"])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    else:
        val_loader = None

    return train_loader, val_loader


def resolve_work_dir(config, args):
    if args.resume:
        return os.path.dirname(args.resume)
    if args.work_dir:
        return args.work_dir
    if config["train"].get("save_name"):
        return os.path.join("checkpoints", config["train"]["save_name"])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("checkpoints", f"{config['model']['name']}_{config['task']}_kd_{timestamp}")


def save_configs_for_run(config, work_dir, resume):
    if resume:
        return
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return

    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "train_config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    if config.get("data_config"):
        with open(os.path.join(work_dir, "data_config.yaml"), "w") as f:
            yaml.safe_dump(config["data_config"], f, sort_keys=False)


def main():
    args = get_args()
    config = load_config(args)

    print("Building KD dataloaders...")
    train_loader, val_loader = build_dataloaders(config)

    print(f"Building student model: {config['model']['name']}...")
    student = build_model(config["model"])
    student = maybe_load_student_pretrained(student, config, args.resume)

    print("Building frozen teacher models...")
    teachers = build_teachers(config)
    print(f"Teachers: {', '.join(teachers.keys())}")

    work_dir = resolve_work_dir(config, args)
    save_configs_for_run(config, work_dir, args.resume)

    trainer = OnlineKDTrainer(
        student_model=student,
        teacher_models=teachers,
        config=config,
        checkpoint_dir=work_dir,
    )
    trainer.fit(train_loader, val_loader, epochs=config["train"]["epochs"], resume_path=args.resume)


if __name__ == "__main__":
    main()
