#!/bin/bash

set -euo pipefail

TOOLCHAIN=$1

ARCH="${TOOLCHAIN%%-*}"
case ${TOOLCHAIN} in
*-linux-gnu)
    ZIG_TARGET="${ARCH}-linux-gnu.2.17"
    OS=Linux
    export CFLAGS="-std=c99 -Wno-error=implicit-function-declaration -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export CXXFLAGS="-std=c++11 -Ofast -DNDEBUG -ffunction-sections -fdata-sections"
    export LDFLAGS="-Wl,--gc-sections -Wl,-s"
    ;;
*-apple-darwin)
    ZIG_TARGET="${ARCH}-macos.11.0"
    OS=Darwin
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
CMAKE_TOOLCHAIN_FILE="${BUILD_DIR}.cmake"
cat >"${CMAKE_TOOLCHAIN_FILE}" <<EOF
set(CMAKE_SYSTEM_PROCESSOR ${ARCH})
set(CMAKE_SYSTEM_NAME ${OS})
set(CMAKE_C_COMPILER zig cc)
set(CMAKE_C_COMPILER_TARGET ${ZIG_TARGET})
set(CMAKE_CXX_COMPILER zig c++)
set(CMAKE_CXX_COMPILER_TARGET ${ZIG_TARGET})
foreach(lang C CXX)
    set(CMAKE_\${lang}_COMPILER_AR zig ar)
    set(CMAKE_\${lang}_COMPILER_RANLIB zig ranlib)
endforeach()
EOF

rm -f "${BUILD_DIR}/libxml2/CMakeCache.txt"
cmake -B "${BUILD_DIR}/libxml2" -S "${SOURCE_DIR}/libxml2-v${LIBXML2_VERSION}" \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLIBXML2_WITH_C14N=OFF \
    -DLIBXML2_WITH_CATALOG=OFF \
    -DLIBXML2_WITH_DEBUG=OFF \
    -DLIBXML2_WITH_DOCB=OFF \
    -DLIBXML2_WITH_FTP=OFF \
    -DLIBXML2_WITH_HTML=OFF \
    -DLIBXML2_WITH_HTTP=OFF \
    -DLIBXML2_WITH_ICONV=OFF \
    -DLIBXML2_WITH_ISO8859X=OFF \
    -DLIBXML2_WITH_LEGACY=OFF \
    -DLIBXML2_WITH_MEM_DEBUG=OFF \
    -DLIBXML2_WITH_PATTERN=OFF \
    -DLIBXML2_WITH_PUSH=OFF \
    -DLIBXML2_WITH_PYTHON=OFF \
    -DLIBXML2_WITH_READER=OFF \
    -DLIBXML2_WITH_REGEXPS=OFF \
    -DLIBXML2_WITH_RUN_DEBUG=OFF \
    -DLIBXML2_WITH_SCHEMAS=OFF \
    -DLIBXML2_WITH_SCHEMATRON=OFF \
    -DLIBXML2_WITH_THREADS=OFF \
    -DLIBXML2_WITH_THREAD_ALLOC=OFF \
    -DLIBXML2_WITH_TREE=OFF \
    -DLIBXML2_WITH_VALID=OFF \
    -DLIBXML2_WITH_WRITER=OFF \
    -DLIBXML2_WITH_XINCLUDE=OFF \
    -DLIBXML2_WITH_XPTR=OFF \
    -DLIBXML2_WITH_ZLIB=OFF \
    -DLIBXML2_WITH_MODULES=OFF \
    -DLIBXML2_WITH_LZMA=OFF \
    -DLIBXML2_WITH_PROGRAMS=OFF \
    -DLIBXML2_WITH_TESTS=OFF \
    -DHAVE_PTHREAD_H=OFF
cmake --build "${BUILD_DIR}/libxml2" -j"$(nproc)"
cmake --install "${BUILD_DIR}/libxml2"

rm -f "${BUILD_DIR}/qdpxx/CMakeCache.txt"
cmake -B "${BUILD_DIR}/qdpxx" -S "${SOURCE_DIR}/qdpxx" \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DQDP_PARALLEL_ARCH=scalar \
    -DQDP_PRECISION=double
cmake --build "${BUILD_DIR}/qdpxx" -j"$(nproc)"
cmake --install "${BUILD_DIR}/qdpxx"

rm -f "${BUILD_DIR}/chroma/CMakeCache.txt"
cmake -B "${BUILD_DIR}/chroma" -S "${SOURCE_DIR}/chroma" \
    -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_DISABLE_FIND_PACKAGE_GMP=ON
cmake --build "${BUILD_DIR}/chroma" -j"$(nproc)"
cmake --install "${BUILD_DIR}/chroma"
