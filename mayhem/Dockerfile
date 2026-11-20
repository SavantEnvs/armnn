# syntax=docker/dockerfile:1.7
#
# Commit-image Dockerfile (SMALL form, PREFERRED) — builds ON TOP of the org C/C++ base
# image (ghcr.io/savantenvs/base). The base already provides debian + clang/llvm +
# libFuzzer/afl++ + the identities + the build-contract ENV, so this only adds the
# per-commit build (built on top of the prebuilt base, rather than building the toolchain in-image).
#
# Build (from the repo root):
#   DOCKER_BUILDKIT=1 docker build -f mayhem/Dockerfile \
#       --build-arg COMMIT=$(git rev-parse HEAD) \
#       [--build-arg SOURCE_DATE_EPOCH=$(date +%s)] \
#       -t mytag .

# The org base, pinned to a fixed DATE-tagged base (YYYYMMDD; shared by all integrations
# minted against it). Use the current dated tag; pin by digest in production: ...base@sha256:<digest>.
FROM ghcr.io/savantenvs/base:20261023

ARG COMMIT
ARG REPO_URL
# SOURCE_DATE_EPOCH stays an ARG (available to build.sh); do NOT bake it as an ENV — an
# empty value breaks clang ("must be a non-negative decimal integer") both here and for
# any in-image rebuild. build.sh unsets it when empty.
ARG SOURCE_DATE_EPOCH

# Sanitizer override: the base exports SANITIZER_FLAGS (ASan+UBSan, halting) as the default.
# To change it for a build (e.g. ASan-only, or add MSan): --build-arg SANITIZER_FLAGS="...".
# Declare it WITHOUT a default (no '='): an undefaulted ARG lets the base ENV flow through to
# build.sh when not passed, and an explicit --build-arg overrides it. (An empty default — `=` —
# would instead shadow the base ENV and silently wipe the sanitizers.)
ARG SANITIZER_FLAGS
# Coverage (optional, off by default): instrument the TEST build to measure how much of the project
# the suite exercises — --build-arg COVERAGE_FLAGS="-fprofile-instr-generate -fcoverage-mapping".
# Empty by default (no effect on the normal oracle build); build.sh appends it to the test build only.
ARG COVERAGE_FLAGS=
# Build parallelism (optional). build.sh falls back to nproc when unset: --build-arg MAYHEM_JOBS=8
ARG MAYHEM_JOBS=

# If mayhem/build.sh needs EXTRA apt packages, install them as root before the build
# (re-pin to SOURCE_DATE_EPOCH first for reproducibility — repro-sources-list.sh ships
# in the base):
#   USER root
#   RUN [ -n "$SOURCE_DATE_EPOCH" ] && repro-sources-list.sh || true ; \
#       apt-get update && apt-get install -yq --no-install-recommends <pkgs> \
#    && rm -rf /var/lib/apt/lists/*
#   USER mayhem

# Anything the upstream build would download, or a library apt lacks or ships at the wrong version, is
# built once from pinned source into a static prefix under /opt/toolchains/deps/<name>-<version>/
# (root-owned, world-readable, no .so); build.sh just points at it, e.g. via PKG_CONFIG_PATH
# (SPEC §6.2 item 19). The source stays in $d/src and the build runs out of tree with -gdwarf-3, so
# backtraces show /opt/toolchains/deps/... paths and the harness binary stays DWARF <= 3. Example:
#   USER root
#   ARG JANSSON_VER=2.14.1
#   ARG JANSSON_SHA256=<sha256 of the tarball>
#   RUN d=/opt/toolchains/deps/jansson-${JANSSON_VER} && mkdir -p "$d/src" /tmp/dep-build \
#    && curl -fsSL -o /tmp/dep.tgz \
#         https://github.com/akheron/jansson/releases/download/v${JANSSON_VER}/jansson-${JANSSON_VER}.tar.gz \
#    && echo "${JANSSON_SHA256}  /tmp/dep.tgz" | sha256sum -c - \
#    && tar -xzf /tmp/dep.tgz --strip-components=1 -C "$d/src" && cd /tmp/dep-build \
#    && "$d/src/configure" --prefix="$d" --disable-shared --enable-static CFLAGS="-O2 -g -gdwarf-3" \
#    && make -j"$(nproc)" install \
#    && cd / && rm -rf /tmp/dep-build /tmp/dep.tgz && chmod -R a+rX "$d"
#   USER mayhem

# The ONNX parser needs protobuf (Debian's libprotobuf-dev + the matching protoc) and the C++ code
# protoc generates from onnx.proto at the ONNX version upstream pins (common.sh: 1.6.0).
USER root
RUN [ -n "$SOURCE_DATE_EPOCH" ] && repro-sources-list.sh || true ; \
    apt-get update && apt-get install -yq --no-install-recommends libprotobuf-dev protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*
ARG ONNX_VER=1.6.0
ARG ONNX_PROTO_SHA256=4d4aad1c9b8428bd389cac622c0188c871f501bb9f9e5617b0c4bd09dc320973
RUN o=/opt/toolchains/deps/onnx-${ONNX_VER}/src && mkdir -p "$o/onnx" \
 && curl -fsSL -o "$o/onnx/onnx.proto" https://raw.githubusercontent.com/onnx/onnx/v${ONNX_VER}/onnx/onnx.proto \
 && echo "${ONNX_PROTO_SHA256}  $o/onnx/onnx.proto" | sha256sum -c - \
 && cd "$o" && protoc onnx/onnx.proto --proto_path=. --cpp_out=. \
 && chmod -R a+rX /opt/toolchains/deps
USER mayhem

# Arm NN's TfLite parser and Arm NN serializer/deserializer need flatbuffers (headers, static lib, flatc)
# at the version upstream pins (build-tool/scripts/common.sh: 24.3.25), and the TfLite parser needs the
# C++ header generated from TensorFlow's schema.fbs (v2.19.0, as setup-armnn.sh does) plus TF's two
# version headers; those three files are all of TENSORFLOW_ROOT the parser build reads. apt ships a
# different flatbuffers, so both are built once here from pinned sources (SPEC 6.2 item 19).
USER root
ARG FLATBUFFERS_VER=24.3.25
ARG FLATBUFFERS_SHA256=4157c5cacdb59737c5d627e47ac26b140e9ee28b1102f812b36068aab728c1ed
ARG TF_VER=2.19.0
ARG TFLITE_SCHEMA_SHA256=20685cadade475c2a5cccf29339ecd2bf27755da50dc3d2c092f1ee2e21ff7d3
RUN d=/opt/toolchains/deps/flatbuffers-${FLATBUFFERS_VER} && mkdir -p "$d/src" /tmp/dep-build \
 && curl -fsSL -o /tmp/dep.tgz https://github.com/google/flatbuffers/archive/refs/tags/v${FLATBUFFERS_VER}.tar.gz \
 && echo "${FLATBUFFERS_SHA256}  /tmp/dep.tgz" | sha256sum -c - \
 && tar -xzf /tmp/dep.tgz --strip-components=1 -C "$d/src" \
 && cmake -S "$d/src" -B /tmp/dep-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$d" \
      -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_CXX_FLAGS="-O2 -g -gdwarf-3" -DCMAKE_C_FLAGS="-O2 -g -gdwarf-3" \
 && cmake --build /tmp/dep-build -j"$(nproc)" && cmake --install /tmp/dep-build \
 && t=/opt/toolchains/deps/tensorflow-${TF_VER}/src && raw=https://raw.githubusercontent.com/tensorflow/tensorflow/v${TF_VER} \
 && mkdir -p "$t/tensorflow/lite/schema" "$t/tensorflow/core/public" \
 && curl -fsSL -o "$t/tensorflow/lite/schema/schema.fbs" $raw/tensorflow/compiler/mlir/lite/schema/schema.fbs \
 && curl -fsSL -o "$t/tensorflow/lite/version.h" $raw/tensorflow/lite/version.h \
 && curl -fsSL -o "$t/tensorflow/core/public/version.h" $raw/tensorflow/core/public/version.h \
 && printf '%s  %s\n' "${TFLITE_SCHEMA_SHA256}" "$t/tensorflow/lite/schema/schema.fbs" \
      72bae7962a144ac54561efb75494b26b6cb3443c21ed5c750c03cb1b7cb9426e "$t/tensorflow/lite/version.h" \
      b700dd405838049473b13dc157725db47d494a64a013cfcfec71f49c78fe4dbc "$t/tensorflow/core/public/version.h" \
    | sha256sum -c - \
 && cd "$t/tensorflow/lite/schema" && "$d/bin/flatc" -c --gen-object-api --reflect-types --reflect-names schema.fbs \
 && cd / && rm -rf /tmp/dep-build /tmp/dep.tgz && chmod -R a+rX /opt/toolchains/deps
USER mayhem

# Source enters from the build CONTEXT (the CI checkout): the FULL repo (incl. .git
# history) is copied to /mayhem as `mayhem`; build.sh builds in place.
USER mayhem
WORKDIR /mayhem
COPY --chown=2000:2000 . /mayhem

# Anything a Dockerfile step materializes into the tree (downloaded fixtures, generated headers,
# vendored deps) MUST be committed or (re)created by build.sh: rlenv's patch grader runs
# `git clean -ffdX` before every graded build, so a gitignored, untracked file is gone by then and
# every honest patch scores zero (SPEC §6.2 item 16; verify-repo.sh checks it).
#
# OPTIONAL compile-cache pre-fill (SPEC §6.2 item 18). Bases minted with /opt/mayhem/ccache/prefill
# run the build under rlenv's pinned ccache, so the patch grader's first (otherwise cold) rebuild is
# warm — this is what keeps large C/C++ targets inside the build time limit. Older bases fall back to
# the plain build; a missing or stale pre-fill only costs a cold cache. Never enable ccache in
# build.sh or via ENV/PATH, and keep the compiler resolving by name (CC=clang) or nothing is cached.
RUN if [ -x /opt/mayhem/ccache/prefill ]; then /opt/mayhem/ccache/prefill bash mayhem/build.sh; \
    else bash mayhem/build.sh; fi

# Run the project's functional tests at build time IF the repo ships them (mayhem/test.sh is
# optional — omit it only when the repo has no tests meaningful for RL). A failing suite fails the
# image build, so the commit image is guaranteed to build AND pass its own tests.
RUN if [ -f mayhem/test.sh ]; then bash mayhem/test.sh; else echo "no mayhem/test.sh — repo ships no functional tests"; fi

LABEL org.opencontainers.image.source="${REPO_URL}" \
      org.opencontainers.image.revision="${COMMIT}" \
      rlenv.tier="commit" \
      rlenv.source_date_epoch="${SOURCE_DATE_EPOCH}" \
      rlenv.user="mayhem"

USER mayhem
WORKDIR /mayhem
