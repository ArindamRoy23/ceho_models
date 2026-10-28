# ceho-models

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Inference servers for transcription, diarization, and speaker identification on audio files.

At the moment this is a toy project for learning deep-learning inference and audio processing. Much of the code is likely to be rewritten as we go.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Repository structure

Work is split into training and inference. Training code will live under `training/`; that directory is not in the tree yet. Inference code lives under `inference/`, with one nested [uv](https://docs.astral.sh/uv/) workspace per server. The root `pyproject.toml` lists those workspaces as members and has no dependencies of its own.

`data/` holds datasets and downloaded weights. It is listed in `.gitignore`, so nothing under it is committed. `inference/transcription/download.sh` writes the Parakeet checkpoint (`nvidia/parakeet-tdt-0.6b-v3`) to `data/transcription/models/`.

```
.
├── data/                         # gitignored local data
│   └── transcription/
│       └── models/               # Parakeet weights from download.sh
├── inference/
│   ├── transcription/            # speech-to-text workspace
│   │   ├── download.sh           # fetches the model into data/
│   │   └── pyproject.toml        # NeMo ASR, CUDA 12.6 PyTorch
│   └── diarization/              # diarization server, empty for now
├── main.py                       # placeholder entry point
├── pyproject.toml                # uv workspace root
├── uv.lock
└── .python-version               # 3.12
```

`inference/transcription` is the only workspace member so far. Its package depends on `nemo-toolkit[asr]` and a CUDA 12.6 build of PyTorch and torchaudio. `inference/diarization` is reserved for the diarization server and has no package files yet. `main.py` prints a hello message and does not load or serve a model.

## Run

```bash
uv run python main.py
```

## License

[MIT](LICENSE)
