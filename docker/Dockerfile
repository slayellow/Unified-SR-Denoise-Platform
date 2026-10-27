# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Job image for unified-sr-denoise-platform (docs/modules/20261019-MOD-DockerEnv.md). Build it with docker/build.sh.
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04@sha256:60cfa1fb926e4830ec1dbee1b7511eeba0fee2333b16cbec79c356e2afc2f359
# The base entrypoint prints a CUDA banner on every run; job output must start with the command's own output.
ENTRYPOINT []
ARG DEBIAN_FRONTEND=noninteractive

# gcc, libc6-dev and python3-dev are needed at run time: Triton compiles AIMET's GPU kernels on first use.
# libc++/libunwind 14 are the QAIRT tools' runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3.10-venv gcc libc6-dev git \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libc++1-14 libc++abi1-14 libunwind-14 tzdata curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Same versions as the old sr container. --no-build-isolation builds the two sdists (filterpy, openai-clip)
# with the pinned setuptools and packaging; opencv-python-headless goes last because its cv2 files are the ones
# the old environment ran (opencv-python installs the same cv2 package).
ENV PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m pip install --no-deps pip==26.0.1 setuptools==82.0.0 wheel==0.46.3 packaging==25.0
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-deps --no-build-isolation -r /tmp/requirements.txt \
    && python3 -m pip install --no-deps --force-reinstall opencv-python-headless==4.12.0.88 \
    && python3 -m pip check \
    && rm /tmp/requirements.txt

# The old container's static ffmpeg 7.0.2 (libx264), taken byte-identical from the imageio-ffmpeg wheel on PyPI.
# Only the binary is installed; zipfile drops the exec bit, so install -m 0755 sets it.
ARG FFMPEG_WHEEL_URL=https://files.pythonhosted.org/packages/a0/2d/43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl
ARG FFMPEG_WHEEL_SHA256=c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282
RUN curl -fsSL "${FFMPEG_WHEEL_URL}" -o /tmp/ff.whl \
    && echo "${FFMPEG_WHEEL_SHA256}  /tmp/ff.whl" | sha256sum -c - \
    && python3 -c "import zipfile; open('/tmp/ffmpeg', 'wb').write(zipfile.ZipFile('/tmp/ff.whl').read('imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'))" \
    && install -m 0755 /tmp/ffmpeg /usr/local/bin/ffmpeg \
    && rm /tmp/ff.whl /tmp/ffmpeg

# QAIRT 2.41 from the pristine tarball (docker/build.sh passes it as the named context "qairt"), without examples/.
# Root-owned and read-only for the job user. The converter's NumPy-1 venv is entered only through qairt-env;
# qairt-requirements.txt pins all of it, pip and setuptools included (`pip freeze --all`).
COPY docker/qairt-requirements.txt /tmp/qairt-requirements.txt
RUN --mount=type=bind,from=qairt,target=/ctx \
    mkdir -p /opt/qairt \
    && tar --no-same-owner --strip-components=1 --exclude='QAIRT_2.41/examples' \
        -xzf /ctx/QAIRT_2.41.tar.gz -C /opt/qairt \
    && python3 -m venv /opt/qairt-venv \
    && /opt/qairt-venv/bin/pip install --no-cache-dir -r /tmp/qairt-requirements.txt \
    && /opt/qairt-venv/bin/pip check \
    && rm /tmp/qairt-requirements.txt
COPY --chmod=0755 docker/qairt-env /usr/local/bin/qairt-env
# Fail the build when a conversion-path binary or Python extension misses a shared library
# (qti/tvm sits outside qti/aisw and needs libLLVM-14, which the ONNX path never loads).
RUN BAD=0; \
    for F in $(find /opt/qairt/bin/x86_64-linux-clang /opt/qairt/lib/x86_64-linux-clang /opt/qairt/lib/python/qti/aisw -type f); do \
        if head -c 4 "$F" | grep -q ELF \
            && LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang ldd "$F" | grep -q "not found"; then \
            echo "missing shared library: $F"; BAD=1; \
        fi; \
    done; \
    [ "$BAD" = 0 ]

ARG UID=1017
ARG GID=1017
ARG USERNAME=jshong
RUN groupadd -g "${GID}" "${USERNAME}" \
    && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}" \
    && install -d -o "${UID}" -g "${GID}" "/home/${USERNAME}/.cache" "/home/${USERNAME}/.cache/torch" \
        "/home/${USERNAME}/.config"
LABEL usdp.uid="${UID}" usdp.gid="${GID}" usdp.home="/home/${USERNAME}"
ENV TZ=Asia/Seoul LANG=C.UTF-8 NUMPY_MADVISE_HUGEPAGE=0 PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=void
USER ${UID}:${GID}
