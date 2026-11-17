```sh
adb install -r target_app/ghost-four-models.apk

adb shell mkdir -p /sdcard/Android/data/com.example.annoymous/files/
adb push models/litertlm/gemma-4-E2B-it_Google_Tensor_G5.litertlm /sdcard/Android/data/com.example.annoymous/files/
adb push models/litertlm/qwen3_5_2b_vl_spec_Google_Tensor_G5.litertlm /sdcard/Android/data/com.example.annoymous/files/
adb push models/litertlm/llama3_2_1b_Google_Tensor_G5.litertlm /sdcard/Android/data/com.example.annoymous/files/
adb push models/litertlm/ministral3_3b_vl_Google_Tensor_G5.litertlm /sdcard/Android/data/com.example.annoymous/files/

adb install -r -t target_app/ghost-androidTest.apk
```
