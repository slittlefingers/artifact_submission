# Asset placement

Download from [Hugging Face](https://huggingface.co/datasets/Floooding/artifact_submission). All destinations below are relative to the code directory containing `kvidence.py`.

| Resource | Hugging Face location | Local destination / action |
|---|---|---|
| Original datasets | `assets/data_*.zip` | Keep in `assets/`, without extracting. |
| Gemma image reconstructors (layers 0 and 4) | `assets/model_gemma_image_layer*.zip` | Keep in `assets/`, without extracting. |
| Gemma audio reconstructors (layers 0 and 4) | `assets/model_gemma_audio_layer*.zip` | Keep in `assets/`, without extracting. |
| 5.2 layer-0 text K/V | `packages/kv-5.2-*.zip` | Extract into the code root; creates `assets/kv/5.2/`. |
| 5.3 image K/V and inputs | `packages/kv-5.3-*.zip` | Extract into the code root; creates `assets/kv/5.3/`. |
| 5.4 audio K/V and inputs | `packages/kv-5.4-*.zip` | Extract into the code root; creates `assets/kv/5.4/`. |
| 5.5 text-search K/V | `packages/kv-5.5-*.zip` | Extract into the code root; creates `assets/kv/5.5/`. |
| 5.6 Gemma OmniDialog | `packages/kv-5.6_gemma-*.zip` | Extract into the code root; creates `assets/kv/5.6_gemma/`. |
| 5.6 Gemma PhotoChat | `packages/kv-5.6_gemma_photochat-*.zip` | Extract into the code root; creates `assets/kv/5.6_gemma_photochat/`. |
| 5.6 Qwen PhotoChat | `packages/kv-5.6_qwen-*.zip` | Extract into the code root; creates `assets/kv/5.6_qwen/`. |
| K/V manifests | `packages/kv-metadata-*.zip` | Extract into the code root; files go into `assets/kv/`. |
| Layer-0 V and shallow-layer reference tables | `packages/reference-tables-*.zip` | Extract into the code root; creates `reconstruction/assets/`. |
| Four deployed LiteRT-LM models | `models/litertlm/` | Preserve as `models/litertlm/`. |
| Embedding tables and derived inputs | `models/derived/` | Preserve as `models/derived/`. |
| Gemma and Qwen Learned-V models | `models/learned_v/` | Preserve as `models/learned_v/`, including configurations. |
| Qwen layer-3 image reconstructor | `models/qwen_layer3_kv_image/` | Preserve as `models/qwen_layer3_kv_image/`, including configurations. |
| Tokenizer files | `models/tokenizers/` | Preserve as `models/tokenizers/`. |
| Ghost and androidTest APKs | Included in the GitHub code | Keep in `target_app/`; install using the commands below. |
| 5.1 LiME inputs | Supply your own memory images | Place under `assets/lime/5.1/` as specified by the 5.1 demo. |

Download and extract from the code root:

```sh
python -m pip install -U huggingface_hub
hf download Floooding/artifact_submission --repo-type dataset --local-dir . --exclude README.md
python -c "from pathlib import Path; from zipfile import ZipFile; [ZipFile(p).extractall('.') for p in sorted(Path('packages').glob('*.zip'))]"
```

Each numbered ZIP is independently extractable. Download and extract all numbered ZIPs for each required resource. Do not extract `packages/*.zip` inside `assets/`: their entries already include the destination directories. Keep `assets/*.zip` compressed because the loaders read them directly.

Install both APKs:

```sh
adb install -r target_app/ghost-four-models.apk
adb install -r -t target_app/ghost-androidTest.apk
```

Copy the four `models/litertlm/*.litertlm` files to `/sdcard/Android/data/com.example.annoymous/files/` on the phone using [MODEL_PLACEMENT.md](target_app/MODEL_PLACEMENT.md).
