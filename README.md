Run from the repository root. Download datasets and models from [Hugging Face](https://huggingface.co/datasets/Floooding/artifact_submission). Keep the downloaded `assets/` and `models/` folders in this repository root; extract `packages/*.zip` into the repository root. See `ASSETS.md`.
```sh
python -m pip install -r requirements.txt
# Tests: C++17 clang++/g++ on PATH, or a Visual Studio Developer Command Prompt.
python -m pytest reconstruction/tests -q -rs
adb install -r target_app/ghost-four-models.apk
adb install -r -t target_app/ghost-androidTest.apk
# Copy the four models using target_app/MODEL_PLACEMENT.md.
# KVidence: acquisition -> extraction -> reconstruction.
python kvidence.py acquire --help
python kvidence.py extract --help
python kvidence.py reconstruct --help
# 5.1 demo: extract K/V from LiME memory images placed in assets/lime/5.1/.
python kvidence.py experiment 5.1
# 5.2: layer-0 V text-token recovery.
python kvidence.py experiment 5.2
# 5.3: image reconstruction from K/V.
python kvidence.py experiment 5.3
# 5.4: audio reconstruction from layer-0 V and layer-4 K/V.
python kvidence.py experiment 5.4
# 5.5: three text-search methods on Gemma and Qwen (phone NPU).
python kvidence.py experiment 5.5
# 5.6: multimodal conversation recovery (phone NPU).
python kvidence.py experiment 5.6-gemma
python kvidence.py experiment 5.6-qwen
# Quick demos for 5.2-5.6; --full runs the complete cohorts.
python kvidence.py experiment all
python kvidence.py experiment all --offline
python kvidence.py experiment all --full
```
