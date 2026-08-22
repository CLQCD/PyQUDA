#!/bin/bash

set -euo pipefail

TOOLCHAIN=$1

ARCH="${TOOLCHAIN%%-*}"
case ${TOOLCHAIN} in
*-linux-gnu)
    ZIG_TARGET="${ARCH}-linux-gnu.2.17"
    export CFLAGS="-std=c99 -Wno-error=implicit-function-declaration -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export CXXFLAGS="-std=c++11 -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export LDFLAGS="-Wl,--gc-sections -Wl,-s"
    ;;
*-apple-darwin)
    ZIG_TARGET="${ARCH}-macos.11.0"
    export CFLAGS="-std=c99 -Wno-error=implicit-function-declaration -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export CXXFLAGS="-std=c++11 -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export LDFLAGS="-Wl,-dead_strip -Wl,-S -Wl,-x"
    ;;
*)
    echo "Unsupported toolchain: ${TOOLCHAIN}"
    exit 1
    ;;
esac

LIBXML2_VERSION=2.9.14
QDPXX_COMMIT=c1e9b5209f89d232af064a60004f8ac7c9a5c734
CHROMA_COMMIT=71ed3c2debc963641999abe6d40bb2532ba4e249

if [ "$2" == "download" ]; then
    curl -LO https://gitlab.gnome.org/GNOME/libxml2/-/archive/v${LIBXML2_VERSION}/libxml2-v${LIBXML2_VERSION}.tar.gz
    tar -xzf libxml2-v${LIBXML2_VERSION}.tar.gz

    git init qdpxx
    cd qdpxx
    git remote add origin https://github.com/usqcd-software/qdpxx.git
    git fetch --depth 1 origin ${QDPXX_COMMIT}
    git checkout FETCH_HEAD
    git submodule update --init --recursive --depth 1
    cd ..

    git init chroma
    cd chroma
    git remote add origin https://github.com/JeffersonLab/chroma.git
    git fetch --depth 1 origin ${CHROMA_COMMIT}
    git checkout FETCH_HEAD
    git submodule update --init --recursive --depth 1
    cd ..
fi

SOURCE_DIR=$(pwd)
BUILD_DIR=$(pwd)/build/${TOOLCHAIN}
INSTALL_DIR=$(pwd)/install/${TOOLCHAIN}

mkdir -p "${BUILD_DIR}"
export CC="zig cc -target ${ZIG_TARGET}"
export CXX="zig c++ -target ${ZIG_TARGET}"
export AR="zig ar"
export RANLIB="zig ranlib"

cd "${SOURCE_DIR}/libxml2-v${LIBXML2_VERSION}"
NOCONFIGURE=1 ./autogen.sh
mkdir -p "${BUILD_DIR}/libxml2"
cd "${BUILD_DIR}/libxml2"
"${SOURCE_DIR}/libxml2-v${LIBXML2_VERSION}/configure" \
    --host="${TOOLCHAIN}" \
    --prefix="${INSTALL_DIR}" \
    --disable-shared \
    --with-minimum \
    --without-lzma \
    --with-sax1 \
    --with-output \
    --with-xpath
make -j"$(nproc)"
make install -j"$(nproc)"

cd "${SOURCE_DIR}/qdpxx"
./autogen.sh
mkdir -p "${BUILD_DIR}/qdpxx"
cd "${BUILD_DIR}/qdpxx"
"${SOURCE_DIR}/qdpxx/configure" \
    --host="${TOOLCHAIN}" \
    --prefix="${INSTALL_DIR}" \
    --with-libxml2="${INSTALL_DIR}" \
    --enable-parallel-arch=scalar \
    --enable-precision=double
make -j"$(nproc)"
make install -j"$(nproc)"

cd "${SOURCE_DIR}/chroma"
./autogen.sh
mkdir -p "${BUILD_DIR}/chroma"
cd "${BUILD_DIR}/chroma"
"${SOURCE_DIR}/chroma/configure" \
    --host="${TOOLCHAIN}" \
    --prefix="${INSTALL_DIR}" \
    --with-qdp="${INSTALL_DIR}"
make -j"$(nproc)"
make install -j"$(nproc)"
