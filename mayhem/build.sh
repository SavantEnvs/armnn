#!/usr/bin/env bash
# mayhem/build.sh — build Arm NN (core + CpuRef backend + TfLite parser + Arm NN serializer/
# deserializer + profiling pipe server/timeline decoder) with ASan+UBSan and SanCov, then the fuzz
# harnesses (libFuzzer + standalone) and the known-answer oracle mayhem/test.sh runs.
set -euo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
: "${SRC:=/mayhem}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

FB=/opt/toolchains/deps/flatbuffers-24.3.25
TFR=/opt/toolchains/deps/tensorflow-2.19.0/src
ONNXG=/opt/toolchains/deps/onnx-1.6.0/src
cd "$SRC"
B="$SRC/mayhem/build/armnn"          # gitignored by mayhem/.gitignore
OUT="$SRC/mayhem/build/bin"
mkdir -p "$B" "$OUT"

# SanCov for the library code too (coverage feedback from armnn itself, not only the harness).
COV="-fsanitize=fuzzer-no-link"
[ -n "$SANITIZER_FLAGS" ] || COV=""
# A custom build type: CMAKE_CXX_FLAGS_<CONFIG> lands AFTER upstream's own "-Werror ..." so our
# -Wno-error wins (clang-19 warns where upstream's gcc does not). Not NDEBUG: keep asserts.
PFLAGS="-O1 $SANITIZER_FLAGS $COV $DEBUG_FLAGS -Wno-error"
cmake -S "$SRC" -B "$B" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Mayhem \
    -DCMAKE_CXX_FLAGS_MAYHEM="$PFLAGS" -DCMAKE_C_FLAGS_MAYHEM="$PFLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$SANITIZER_FLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$SANITIZER_FLAGS" \
    -DUSE_CCACHE=OFF -DARMNNREF=1 -DARMCOMPUTENEON=0 -DARMCOMPUTECL=0 \
    -DBUILD_UNIT_TESTS=0 -DBUILD_TESTS=0 -DARMNN_SAMPLE_APPS_ENABLED=0 -DBUILD_GATORD_MOCK=0 \
    -DBUILD_ARMNN_SERIALIZER=1 -DBUILD_ARMNN_DESERIALIZER=1 -DFLATBUFFERS_ROOT="$FB" -DFLATC_DIR="$FB/bin" -DFLATC="$FB/bin/flatc" \
    -DBUILD_TF_LITE_PARSER=1 -DTF_LITE_GENERATED_PATH="$TFR/tensorflow/lite/schema" -DTENSORFLOW_ROOT="$TFR" \
    -DBUILD_ONNX_PARSER=1 -DONNX_GENERATED_SOURCES="$ONNXG" \
    -DBUILD_TIMELINE_DECODER=1 -DBUILD_BASE_PIPE_SERVER=1
cmake --build "$B" -j"$MAYHEM_JOBS"

INC=(-I"$SRC/include" -I"$SRC/profiling" -I"$SRC/profiling/common/include" -I"$SRC/profiling/server/include"
     -I"$SRC/profiling/client/include" -I"$SRC/third-party")
LIBDIRS=(-L"$B" -L"$B/profiling/server/src/basePipeServer" -L"$B/profiling/server/src/timelineDecoder"
         -L"$B/profiling/common/src" -L"$B/profiling/client/src")
RPATH="-Wl,-rpath,$B:$B/profiling/server/src/basePipeServer:$B/profiling/server/src/timelineDecoder:$B/profiling/common/src:$B/profiling/client/src"
CXXF="-std=c++17 -O1 $SANITIZER_FLAGS $DEBUG_FLAGS"

$CXX $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o "$OUT/lsan_off.o"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$OUT/standalone_main.o"

build_harness() {  # name libs...
    local name="$1"; shift
    $CXX $CXXF "${INC[@]}" -c "$SRC/mayhem/harnesses/$name.cpp" -o "$OUT/$name.o"
    $CXX $CXXF $LIB_FUZZING_ENGINE "$OUT/$name.o" "$OUT/lsan_off.o" "${LIBDIRS[@]}" "$RPATH" "$@" -lpthread -o "$SRC/$name"
    $CXX $CXXF "$OUT/standalone_main.o" "$OUT/$name.o" "$OUT/lsan_off.o" "${LIBDIRS[@]}" "$RPATH" "$@" -lpthread -o "$SRC/$name-standalone"
}
build_harness fuzz_tflite_parser   -larmnnTfLiteParser -larmnn
build_harness fuzz_tflite_run      -larmnnTfLiteParser -larmnn
build_harness fuzz_deserializer    -larmnnDeserializer -larmnnSerializer -larmnn
build_harness fuzz_deserializer_run -larmnnDeserializer -larmnn
build_harness fuzz_onnx_parser     -larmnnOnnxParser -larmnn -lprotobuf
build_harness fuzz_onnx_text       -larmnnOnnxParser -larmnn -lprotobuf
build_harness fuzz_pipe_server     -larmnnBasePipeServer -lpipeCommon
build_harness fuzz_timeline_decoder -ltimelineDecoder -lpipeCommon

# Oracle for mayhem/test.sh (known-answer checks through the public API, see the source).
$CXX -std=c++17 -O1 $SANITIZER_FLAGS $COVERAGE_FLAGS "${INC[@]}" "$SRC/mayhem/oracle/armnn_oracle.cpp" \
    "${LIBDIRS[@]}" "$RPATH" -larmnnTfLiteParser -larmnnDeserializer -larmnnSerializer -larmnn -lpthread \
    -o "$OUT/armnn_oracle"
