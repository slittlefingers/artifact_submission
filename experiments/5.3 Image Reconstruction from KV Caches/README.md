```sh
python kvidence.py experiment 5.3                       # every cohort, first image
python kvidence.py experiment 5.3 --full                # every cohort, all images
python kvidence.py experiment 5.3 vqav2_gemma_layer4 --full   # one cohort

python -m reconstruction.components.image_reconstruction.grid_search --model gemma --cohort assets/kv/5.3/vqav2/gemma_layer4   --checkpoint assets/model_gemma_image_layer4.zip --images assets/data_vqav2.zip   --inspection reconstruction/configs/gemma4_e2b_inspection.json --output OUT/grid_gemma_layer4 --limit 1
python -m reconstruction.components.image_reconstruction.grid_search --model qwen --cohort assets/kv/5.3/vqav2/qwen_layer3   --checkpoint models/qwen_layer3_kv_image --images assets/data_vqav2.zip   --inspection reconstruction/configs/qwen3_5_2b_dynvision_inspection.json --output OUT/grid_qwen_layer3 --limit 1
```
