```sh
python kvidence.py reconstruct prepare-assets --model-name gemma4-e2b
python kvidence.py reconstruct prepare-assets --model-name qwen3.5-2b

python kvidence.py experiment 5.5          # 6 runs: 2 models x 3 methods, one prompt and one token each
python kvidence.py experiment 5.5 --full   # all 100 LMSYS prompts, complete
```
