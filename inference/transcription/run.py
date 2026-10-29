"""Load the local Parakeet checkpoint and transcribe one audio file."""

import argparse
import os
from pathlib import Path

# NeMo checkpoints use torch.load on trusted files. PyTorch 2.6+ defaults to
# weights_only=True, which rejects those archives.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
import nemo.collections.asr as nemo_asr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "transcription" / "models" / "parakeet-tdt-0.6b-v3.nemo"
PARTS = ("preprocessor", "encoder", "decoder", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Speech file to transcribe, wav or flac.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Local .nemo checkpoint.")
    return parser.parse_args()


def load_model(path: Path) -> torch.nn.Module:
    """Restore on CPU, cast to bf16, then move the smaller copy to the GPU."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path), map_location="cpu")
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
        model = model.to(device="cuda")
    model.eval()
    model.freeze()
    return model


def describe(model: torch.nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"class: {type(model).__name__}")
    print(f"device: {next(model.parameters()).device}  dtype: {next(model.parameters()).dtype}")
    print(f"parameters: {total / 1e6:.1f} M")
    for name in PARTS:
        module = getattr(model, name)
        count = sum(parameter.numel() for parameter in module.parameters())
        print(f"  {name:14} {type(module).__name__:32} {count / 1e6:7.2f} M")

    encoder = model.cfg.encoder
    decoder = model.cfg.decoder.prednet
    jointnet = model.cfg.joint.jointnet
    print(
        "encoder: "
        f"{encoder.n_layers} layers, d_model {encoder.d_model}, "
        f"{encoder.n_heads} heads, subsample x{encoder.subsampling_factor}"
    )
    print(f"decoder: {decoder.pred_rnn_layers}-layer LSTM, hidden {decoder.pred_hidden}")
    print(
        f"joint: hidden {jointnet.joint_hidden}, "
        f"extra outputs {model.cfg.joint.num_extra_outputs}"
    )


def transcribe(model: torch.nn.Module, audio: Path) -> str:
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

    hypotheses = model.transcribe([str(audio)], batch_size=1, num_workers=0, verbose=False)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    first = hypotheses[0]
    text = first.text if hasattr(first, "text") else str(first)

    if torch.cuda.is_available():
        after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        print(f"vram allocated: {before / 1024**2:.0f} MiB -> {after / 1024**2:.0f} MiB")
        print(f"vram peak:      {peak / 1024**2:.0f} MiB")
    return text


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    describe(model)
    print(transcribe(model, args.audio))


if __name__ == "__main__":
    main()
