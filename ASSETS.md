Download assets from [Hugging Face](https://huggingface.co/datasets/Floooding/artifact_submission) into this repository root:

```sh
python -m pip install -U huggingface_hub
hf download Floooding/artifact_submission --repo-type dataset --local-dir . --exclude README.md
python -c "from pathlib import Path; from zipfile import ZipFile; [ZipFile(p).extractall('.') for p in sorted(Path('packages').glob('*.zip'))]"
```

Keep `assets/*.zip` compressed; experiment loaders read them directly. Preserve this layout:

```text
assets/*.zip                   Dataset and Gemma reconstructor packages
assets/kv/                     Recorded K/V and evaluation inputs
models/litertlm/               Four deployed model bundles
models/derived/                Embeddings and model-derived inputs
models/learned_v/              Learned-V checkpoints
models/qwen_layer3_kv_image/    Qwen image reconstructor checkpoint
reconstruction/assets/        Layer-0 V tables and shallow-layer reference tables
```

The 5.1 extraction demo uses LiME inputs under `assets/lime/5.1/`.
