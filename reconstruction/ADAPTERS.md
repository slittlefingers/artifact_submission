```sh
python reconstruction/kvrecon/adapters/qwen_verify/build.py --compiler NDK_CLANG++ --include LITERT_HEADERS --include NLOHMANN_JSON_INCLUDE --library-dir APP_JNI_LIBS --output build/qwen_verify_worker

python reconstruction/run.py check-adapter --model-config CONFIG/model.json --prefix-json INPUT/prefix.json --candidate-ids-json INPUT/candidates.json --output OUTPUT/adapter_check
```
