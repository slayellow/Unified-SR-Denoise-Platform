#!/bin/bash
# Build the usdp job image as usdp:<YYYYMMDD> and usdp:latest (docs/modules/20261019-MOD-DockerEnv.md §3.2).
#   QAIRT_TARBALL=<path> USDP_TAG=<tag> docker/build.sh
# The QAIRT tarball reaches the build through a temp folder holding a hard link, because the folder next to it
# has a .dockerignore that hides it. The tarball never enters the repo or git.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
QAIRT_TARBALL=${QAIRT_TARBALL:-$HOME/Code/qualcomm-vlm/env/QAIRT_2.41.tar.gz}
QAIRT_SHA256=4e73e38f5e60e083b7c1e535ceaabb8a18e7523d912ad3b7dbe99d9217e3520d
USDP_TAG=${USDP_TAG:-$(date +%Y%m%d)}
CTX=""

cleanup()
{
    if [ -n "$CTX" ]; then
        rm -f "$CTX/QAIRT_2.41.tar.gz"
        rmdir "$CTX"
    fi
}

echo "------ Build Start! ------"
for TOOL in docker sha256sum id date mktemp realpath; do
    command -v "$TOOL" > /dev/null || { echo "build.sh: $TOOL missing" >&2; exit 1; }
done
docker buildx version > /dev/null || { echo "build.sh: docker buildx missing" >&2; exit 1; }
[ -f "$QAIRT_TARBALL" ] || { echo "build.sh: QAIRT tarball not found: $QAIRT_TARBALL (set QAIRT_TARBALL)" >&2; exit 1; }
# A symlink would reach the build context as a link to a path the build cannot see.
QAIRT_TARBALL=$(realpath -e -- "$QAIRT_TARBALL")
echo "$QAIRT_SHA256  $QAIRT_TARBALL" | sha256sum -c --status - \
    || { echo "build.sh: $QAIRT_TARBALL does not match sha256 $QAIRT_SHA256" >&2; exit 1; }

CTX=$(mktemp -d)
trap cleanup EXIT
ln "$QAIRT_TARBALL" "$CTX/QAIRT_2.41.tar.gz" 2> /dev/null || cp "$QAIRT_TARBALL" "$CTX/QAIRT_2.41.tar.gz"
docker buildx build --builder default --progress plain --build-context qairt="$CTX" \
    --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg USERNAME="$(id -un)" \
    -f "$SCRIPT_DIR/Dockerfile" -t "usdp:$USDP_TAG" -t usdp:latest "$REPO"
docker image inspect -f '{{.Id}}' "usdp:$USDP_TAG"
echo "Build Finish!!"
