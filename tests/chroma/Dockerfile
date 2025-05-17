FROM debian:12

RUN apt-get update && apt-get dist-upgrade -y && \
    apt-get install -y \
    gnulib \
    pkg-config \
    git

RUN curl https://gitlab.gnome.org/GNOME/libxml2/-/archive/v2.9.14/libxml2-v2.9.14.tar.gz -o libxml2-v2.9.14.tar.gz && tar -xzf libxml2-v2.9.14.tar.gz && \
    git clone --recursive --branch devel --depth 1 https://github.com/usqcd-software/qdpxx.git && \
    git clone --recursive --branch devel --depth 1 https://github.com/JeffersonLab/chroma.git

ENV LDFLAGS="-static -static-libgcc -static-libstdc++" \
    CFLAGS="-Ofast -std=c99" \
    CXXFLAGS="-Ofast -std=c++11"

RUN cd libxml2-v2.9.14 && \
    ./autogen.sh \
    --prefix=$(pwd)/install \
    --disable-shared \
    --with-minimum \
    --with-sax1 \
    --with-writer \
    --with-xpath && \
    make -j32 && make install -j32

RUN cd qdpxx && \
    ./autogen.sh && ./configure \
    --prefix=$(pwd)/install \
    --with-libxml2=$(pwd)/../libxml2-v2.9.14/install \
    --enable-parallel-arch=scalar \
    --enable-precision=double && \
    make -j32 && make install -j32

RUN cd chroma && \
    ./autogen.sh && ./configure \
    --prefix=$(pwd)/install \
    --with-qdp=$(pwd)/../qdpxx/install && \
    make -j32 && make install -j32

RUN cd chroma/install/bin && \
    ls -la chroma && strip chroma && ls -la chroma && \
    ls -la hmc && strip hmc && ls -la hmc

FROM scratch

COPY --from=0 /chroma/install/bin/chroma /chroma/install/bin/hmc /
