"""Cross-compile the Qwen verify worker (worker_main.cc) for Android ARM64.

Example (NDK clang, LiteRT and nlohmann/json headers, runtime libs from the app):
    python build.py --compiler NDK/aarch64-linux-android35-clang++ \
        --include LITERT_HEADERS --include NLOHMANN_JSON_INCLUDE \
        --library-dir APP_JNI_LIBS/arm64-v8a --output build/qwen_verify_worker
Writes the binary and <output>.build.json with source and binary SHA-256.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parents[2] / "native"
SOURCES = [HERE / "worker_main.cc", NATIVE / "qwen_worker.h", NATIVE / "litert_support.h",
           NATIVE / "qwen_prefill_capture.h", NATIVE / "qwen_image_prefix.inc", NATIVE / "socket_rpc.h"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compiler", type=Path, required=True, help="NDK aarch64 clang++ (or its .cmd wrapper)")
    p.add_argument("--include", type=Path, action="append", required=True, help="extra include directory (repeatable)")
    p.add_argument("--library-dir", type=Path, required=True, help="directory with liblitert-lm.so for arm64")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    missing = [str(s) for s in SOURCES if not s.is_file()]
    if missing:
        raise SystemExit(f"Missing sources: {missing}")
    cmd = [str(a.compiler), "-std=c++17", "-O2", "-static-libstdc++",
           *[x for d in a.include for x in ("-I", str(d))], "-I", str(NATIVE),
           str(HERE / "worker_main.cc"), "-L", str(a.library_dir),
           "-llitert-lm", "-llog", "-ldl", "-landroid",
           "-Wl,-z,max-page-size=16384", "-Wl,--no-undefined", "-o", str(a.output)]
    if os.name == "nt" and a.compiler.suffix.lower() == ".cmd":
        cmd = ["cmd.exe", "/c", *cmd]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    record = dict(binary=a.output.name, binary_sha256=sha256(a.output),
                  sources={s.name: sha256(s) for s in SOURCES})
    a.output.with_name(a.output.name + ".build.json").write_text(json.dumps(record, indent=2))
    print(a.output, record["binary_sha256"])


if __name__ == "__main__":
    main()
