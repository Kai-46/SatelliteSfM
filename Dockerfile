# syntax=docker/dockerfile:1

ARG BASE_IMAGE=nvidia/cuda:12.5.1-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.5.1-runtime-ubuntu24.04
ARG UV_VERSION=0.11.26

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# =============================== builder ===============================
FROM ${BASE_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=86
ARG PATCHER_REF=4e87e172ed1d8b57f17d9c4d19d0d5c2fb4ef1a2
ARG PATCHES_REF=b6b1d36881e061a8d1ea9dfa375ed787b794ba19
ARG COLMAP_COMMIT=09c22c3c3603dc981d80b6d816fcbdbc4fa9a81a

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates wget ccache \
      python3 python3-dev python3-venv \
      gdal-bin libgdal-dev libtiff-dev \
      libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
      libboost-system-dev libboost-test-dev \
      libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
      libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev \
      qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/ccache set -eux; \
    git clone https://github.com/SBCV/RepositoryPatcher /opt/RepositoryPatcher; \
    git -C /opt/RepositoryPatcher checkout "${PATCHER_REF}"; \
    chmod +x /opt/RepositoryPatcher/*.sh; \
    git clone https://github.com/SBCV/ColmapForVisSatPatched /opt/ColmapForVisSatPatched; \
    git -C /opt/ColmapForVisSatPatched checkout "${PATCHES_REF}"; \
    git clone https://github.com/colmap/colmap /opt/colmap; \
    git -C /opt/colmap checkout "${COLMAP_COMMIT}"; \
    /opt/RepositoryPatcher/apply_patches.sh \
      --tool git_apply reject \
      --patch_dp /opt/ColmapForVisSatPatched/patches \
      --target_repository_dp /opt/colmap \
      --commit "${COLMAP_COMMIT}"; \
    cmake -S /opt/colmap -B /opt/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/colmap-install \
      -DCMAKE_PREFIX_PATH=/usr/local \
      -DCUDA_ENABLED=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache; \
    ninja -C /opt/colmap/build; \
    ninja -C /opt/colmap/build install; \
    rm -rf /opt/colmap/build

COPY --from=uv /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python python3 "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt

RUN uv pip install --no-cache -r /tmp/requirements.txt setuptools wheel && \
    uv pip install --no-cache --no-build-isolation "gdal==$(gdal-config --version)"

# =============================== runtime ===============================
FROM ${RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

COPY docker/runtime-packages.txt /tmp/runtime-packages.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 ca-certificates $(sed 's/#.*//' /tmp/runtime-packages.txt) && \
    rm -rf /var/lib/apt/lists/* /tmp/runtime-packages.txt

COPY --from=builder /opt/colmap-install/ /usr/local/
COPY --from=builder /opt/venv /opt/venv
RUN ldconfig

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV COLMAP_BIN=/usr/local/bin/colmap

WORKDIR /workdir
COPY . /workdir

CMD ["bash"]
