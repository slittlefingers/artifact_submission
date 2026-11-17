```sh
cargo build --release --locked --manifest-path extraction/tools/btf2json/Cargo.toml

python kvidence.py symbols --btf2json extraction/tools/btf2json/target/release/btf2json --btf INPUT/vmlinux.btf --map INPUT/kallsyms --banner "EXACT TARGET LINUX BANNER" --vabits 39 --output INPUT/symbols/linux/target.json

python -m pip install -r extraction/tools/volatility3-arm64/requirements-minimal.txt
python extraction/run.py vol-help
python kvidence.py extract --dump INPUT/capture.lime --symbols INPUT/symbols --pid 1234 --recover-dir extraction/support --va-bits 39 --output OUTPUT/extracted.kvex --address-manifest OUTPUT/addresses.json --model models/litertlm/qwen3_5_2b_vl_spec_Google_Tensor_G5.litertlm --prefix-json INPUT/public_prefix.json --start 1 --count 100 --linear-cache
```
