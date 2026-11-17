Download datasets and models from [Hugging Face](https://huggingface.co/datasets/Floooding/artifact_submission). Preserve the directory layout below:


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
