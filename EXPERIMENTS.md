```sh
# 5.1 demo: extract K/V from LiME memory images placed in assets/lime/5.1/.
# Outputs: outputs/5.1/<run>/<model>/observations/layerN/{observed.npz,metadata.json}.
# Qwen layer 3; Gemma layers 0 and 4; Llama/Ministral layer 0.
python kvidence.py experiment 5.1
python kvidence.py experiment 5.2
python kvidence.py experiment 5.3
python kvidence.py experiment 5.4
python kvidence.py experiment 5.5
python kvidence.py experiment 5.5 --dataset base64
# Both datasets, all three methods; add --model qwen or --model gemma to select a model.
python kvidence.py experiment 5.5 --dataset all --full
python kvidence.py experiment 5.6-gemma
python kvidence.py experiment 5.6-gemma --positions 270
python kvidence.py experiment 5.6-gemma --start 36 --positions 136
python kvidence.py experiment 5.6-qwen
```
