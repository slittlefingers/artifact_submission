#!/usr/bin/env bash
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$native_dir/../.." && pwd)"
headers="${LITERT_HEADERS:-/private/tmp/LiteRT-26895c9fbcc25c43faa8c1a98cd1fd28951602c3}"
json_headers="${JSON_HEADERS:-/PATH/TO/nlohmann_json/include}"
ndk="${ANDROID_NDK_HOME:-/path/to/user/Library/Android/sdk/ndk/30.0.14904198}"
lm_source="${LITERT_LM_SOURCE:?Set LITERT_LM_SOURCE to the matching deployed LiteRT-LM build source}"
generated="${LITERT_LM_GENERATED:-$lm_source/bazel-out/arm64-v8a-opt/bin}"
host="${NDK_HOST_TAG:-darwin-x86_64}"
output="$workspace/target_app/source/app/src/androidTest/jniLibs/arm64-v8a/libkvreconstruction.so"
mkdir -p "$(dirname -- "$output")"
"$ndk/toolchains/llvm/prebuilt/$host/bin/aarch64-linux-android31-clang++" \
 -shared -fPIC -std=c++17 -O2 -static-libstdc++ -I "$headers" -I "$json_headers" -I "${json_headers%/nlohmann_json/include}/stb" \
 -I "$lm_source" -I "$generated" -I "$generated/external/litert" \
 -I "$lm_source/external/com_google_absl" -I "$lm_source/external/com_google_protobuf/src" \
 -I "$lm_source/external/flatbuffers/include" -I "$lm_source/external/sentencepiece/src" \
 -I "$lm_source/external/com_google_protobuf/third_party/utf8_range" \
 "$native_dir/worker.cc" "$native_dir/audio_framework_bridge.cc" -L "$workspace/target_app/source/app/src/main/jniLibs/arm64-v8a" \
 -llitert-lm -llog -ldl -Wl,-z,max-page-size=16384 -Wl,--no-undefined -o "$output"
printf '%s\n' "Built $output"
