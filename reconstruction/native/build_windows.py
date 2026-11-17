"""Build the Android acquisition worker against the installed LiteRT 2.3 headers."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    here=Path(__file__).resolve().parent
    code=here.parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generic-vocab',action='store_true')
    p.add_argument('--headers',type=Path,default=Path(os.environ.get('LITERT_HEADERS','/PATH/TO/LOCAL_PATH_DDAF71398D')))
    p.add_argument('--json-headers',type=Path,default=Path('/PATH/TO/LOCAL_PATH_9CACA62D74'))
    p.add_argument('--stb-headers',type=Path,default=Path('/PATH/TO/LOCAL_PATH_8D23AB6095'))
    p.add_argument('--generated-headers',type=Path,default=Path('/PATH/TO/LOCAL_PATH_4699CBEA26'))
    p.add_argument('--ndk',type=Path,default=Path(os.environ.get('ANDROID_NDK_HOME','/PATH/TO/LOCAL_PATH_90280034D3')))
    p.add_argument('--lm-source',type=Path,default=Path('/PATH/TO/LOCAL_PATH_B55564B3DB'))
    args=p.parse_args()
    output=reconstruction/components/'annoymous/app/src/androidTest/jniLibs/arm64-v8a/libkvreconstruction.so'
    if args.generic_vocab:output=output.with_name('libkvvocab.so')
    output.parent.mkdir(parents=True,exist_ok=True)
    clang=args.ndk/'toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android31-clang++.cmd'
    extra=[]
    include_extra=[]
    if not args.generic_vocab:
        extra.append(str(here/'audio_framework_bridge.cc'))
        external=args.lm_source/'external';generated=args.lm_source/'bazel-out/arm64-v8a-opt/bin'
        includes=[args.lm_source,generated,external/'com_google_absl',external/'com_google_protobuf/src',external/'flatbuffers/include',external/'sentencepiece/src',external/'com_google_protobuf/third_party/utf8_range']
        for path in includes:include_extra+=['-I',str(path)]
    command=['cmd.exe','/c',str(clang),'-shared','-fPIC','-std=c++17','-O2','-static-libstdc++',
        '-I',str(args.headers),'-I',str(args.generated_headers),'-I',str(args.json_headers),'-I',str(args.stb_headers),
        *include_extra,
        str(here/('vocab_generic.cc' if args.generic_vocab else 'worker.cc')),*extra,'-L',str(reconstruction/components/'annoymous/app/src/main/jniLibs/arm64-v8a'),
        '-llitert-lm','-llog','-ldl','-Wl,-z,max-page-size=16384','-Wl,--no-undefined','-o',str(output)]
    subprocess.run(command,check=True)
    print('Built',output)


if __name__=='__main__':main()
