#!/bin/bash
# Start one job container from the usdp image as the calling user (docs/modules/20261019-MOD-DockerRun.md).
#   docker/run.sh [--gpu N] [--name NAME [--log FILE]] [--mount SRC[:DST[:ro]]]... [--offline] [--force] -- CMD...
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd -P)
USDP_IMAGE=${USDP_IMAGE:-usdp:latest}
USDP_DATA_MOUNT=${USDP_DATA_MOUNT:-/mnt/data_server}
USDP_PROBE_TIMEOUT=${USDP_PROBE_TIMEOUT:-10}

GPU=""
GPU_SET=0
NAME=""
LOG=""
LOG_ABS=""
LOG_IN=""
OFFLINE=0
FORCE=0
IMG_HOME=""
MOUNT_SPECS=()
MOUNT_SRCS=()
MOUNT_DSTS=()
MOUNT_ROS=()
MOUNT_ARGS=()
CMD=()
ARGS=()

usage()
{
    echo "run.sh: usage: docker/run.sh [--gpu N] [--name NAME [--log FILE]] [--mount SRC[:DST[:ro]]]... [--offline] [--force] -- CMD..." >&2
    echo "run.sh: $1" >&2
    exit 2
}

refuse()
{
    echo "run.sh: refused: $1" >&2
    exit 1
}

# An option value must be present and must not look like another option (--name --force).
need_value()
{
    [ "$2" -ge 2 ] && [[ "$3" != --* ]] || usage "$1 needs a value"
}

# docker --mount takes a comma-separated list, so a comma inside a path would split the field.
no_comma()
{
    [[ "$2" != *,* ]] || refuse "$1 $2 contains a comma, which docker --mount cannot take"
}

add_mount()
{
    local SRC DST MODE
    IFS=: read -r SRC DST MODE <<< "$1"
    [ -n "$SRC" ] || usage "--mount needs SRC[:DST[:ro]]"
    case "$MODE" in
        "" | ro) ;;
        *) usage "--mount mode must be ro, got '$MODE'" ;;
    esac
    no_comma "--mount source" "$SRC"
    no_comma "--mount target" "$DST"
    MOUNT_SPECS+=("$1")
}

# Resolves the --mount specs only after check_data_mount, because touching a path on a hung sshfs blocks.
# The source is resolved for docker; without DST the job sees the path as typed (symlinks kept).
resolve_mounts()
{
    local SPEC SRC DST MODE
    for SPEC in "${MOUNT_SPECS[@]}"; do
        IFS=: read -r SRC DST MODE <<< "$SPEC"
        [ -n "$DST" ] || DST=$SRC
        SRC=$(realpath -m -- "$SRC")
        DST=$(realpath -ms -- "$DST")
        no_comma "--mount source" "$SRC"
        MOUNT_SRCS+=("$SRC")
        MOUNT_DSTS+=("$DST")
        MOUNT_ROS+=("$MODE")
        MOUNT_ARGS+=(--mount "type=bind,source=$SRC,target=$DST${MODE:+,readonly}")
    done
}

parse_args()
{
    local SEP=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --gpu)
                [ "$GPU_SET" = 0 ] || usage "--gpu given twice (one GPU per job)"
                need_value --gpu "$#" "${2:-}"
                GPU=$2
                GPU_SET=1
                shift 2
                ;;
            --name)
                need_value --name "$#" "${2:-}"
                [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || usage "--name must match [A-Za-z0-9][A-Za-z0-9_.-]*"
                NAME=$2
                shift 2
                ;;
            --log)
                need_value --log "$#" "${2:-}"
                [ -n "$2" ] || usage "--log needs a value"
                LOG=$2
                shift 2
                ;;
            --mount)
                need_value --mount "$#" "${2:-}"
                add_mount "$2"
                shift 2
                ;;
            --offline)
                OFFLINE=1
                shift
                ;;
            --force)
                FORCE=1
                shift
                ;;
            --)
                SEP=1
                shift
                break
                ;;
            *) usage "unknown option '$1' (the command goes after --)" ;;
        esac
    done
    [ "$SEP" = 1 ] || usage "missing -- before the command"
    [ "$#" -gt 0 ] || usage "missing command after --"
    [ -z "$LOG" ] || [ -n "$NAME" ] || usage "--log needs --name"
    CMD=("$@")
}

check_image()
{
    local LABELS ERR IMG_UID IMG_GID
    # Labels come from stdout only: docker may print warnings (an unreadable ~/.docker/config.json) on stderr.
    if ! LABELS=$(docker image inspect -f '{{index .Config.Labels "usdp.uid"}} {{index .Config.Labels "usdp.gid"}} {{index .Config.Labels "usdp.home"}}' "$USDP_IMAGE" 2>/dev/null); then
        ERR=$(docker image inspect "$USDP_IMAGE" 2>&1 > /dev/null) || true
        case "$ERR" in
            *"No such image"*) refuse "image $USDP_IMAGE not found; build it first with docker/build.sh" ;;
            *) refuse "docker image inspect $USDP_IMAGE failed: $(head -n 1 <<< "$ERR")" ;;
        esac
    fi
    read -r IMG_UID IMG_GID IMG_HOME <<< "$LABELS"
    [[ "$IMG_UID" =~ ^[0-9]+$ && "$IMG_GID" =~ ^[0-9]+$ && "$IMG_HOME" == /* ]] \
        || refuse "image $USDP_IMAGE has no usdp.uid/usdp.gid/usdp.home labels; build it with docker/build.sh"
    [ "$(id -u)" != 0 ] || refuse "run as your own user, not root (the image runs as uid:gid $IMG_UID:$IMG_GID)"
    [ "$IMG_UID:$IMG_GID" = "$(id -u):$(id -g)" ] \
        || refuse "image $USDP_IMAGE was built for uid:gid $IMG_UID:$IMG_GID but the caller is $(id -u):$(id -g); rebuild with docker/build.sh"
}

check_paths()
{
    local SRC I BEST CAND FIRST=-1 SRCS DSTS ROS
    no_comma "the repository path" "$REPO"
    no_comma "HOME" "$HOME"
    resolve_mounts
    for SRC in "${MOUNT_SRCS[@]}"; do
        [ -e "$SRC" ] || refuse "--mount source $SRC does not exist"
    done
    [ -n "$NAME" ] || return 0
    LOG_ABS=$(realpath -m -- "${LOG:-$REPO/logs/$NAME.log}")
    # The container writes the log at a path it sees: the deepest mount over that path (an exact-path mount
    # included) must be read-write and lead back to the host file. The path as typed is tried first, then the
    # resolved one, so symlinked mounts and a symlinked repo both work.
    SRCS=("$REPO" /mnt "$HOME/.cache/torch" "${MOUNT_SRCS[@]}")
    DSTS=("$REPO" /mnt "$IMG_HOME/.cache/torch" "${MOUNT_DSTS[@]}")
    ROS=("" "" "" "${MOUNT_ROS[@]}")
    for CAND in "$(realpath -ms -- "${LOG:-$REPO/logs/$NAME.log}")" "$LOG_ABS"; do
        BEST=-1
        for I in "${!DSTS[@]}"; do
            if [[ "$CAND" == "${DSTS[$I]}" || "$CAND" == "${DSTS[$I]}"/* ]] \
                && { [ "$BEST" = -1 ] || [ "${#DSTS[$I]}" -gt "${#DSTS[$BEST]}" ]; }; then
                BEST=$I
            fi
        done
        [ "$BEST" != -1 ] || continue
        [ "$FIRST" != -1 ] || FIRST=$BEST
        if [ -z "${ROS[$BEST]}" ] && [ "$(realpath -m -- "${SRCS[$BEST]}${CAND#"${DSTS[$BEST]}"}")" = "$LOG_ABS" ]; then
            LOG_IN=$CAND
            break
        fi
    done
    if [ -z "$LOG_IN" ]; then
        [ "$FIRST" != -1 ] || refuse "--log $LOG_ABS is outside the mounted paths, so the container cannot write it"
        refuse "--log $LOG_ABS is under --mount ${SRCS[$FIRST]}:${DSTS[$FIRST]}${ROS[$FIRST]:+:ro}, so the container cannot write it at that path"
    fi
    [ -z "$(docker ps -a -q --filter "name=^usdp-$NAME\$")" ] || refuse "container name usdp-$NAME is in use"
}

# A hung sshfs leaves its reader in an uninterruptible wait that timeout cannot kill, so the read probe runs
# in the background and run.sh stops waiting after USDP_PROBE_TIMEOUT. findmnt -C reads only the mount table.
check_data_mount()
{
    local FSTYPE PID TENTHS I OK=0
    FSTYPE=$(findmnt -C -rn -M "$USDP_DATA_MOUNT" -o FSTYPE 2>/dev/null | tail -n 1) || FSTYPE=""
    if [ "$FSTYPE" = fuse.sshfs ]; then
        ls -A "$USDP_DATA_MOUNT" > /dev/null 2>&1 &
        PID=$!
        TENTHS=$(awk -v T="$USDP_PROBE_TIMEOUT" 'BEGIN { t = int(T * 10); print (t < 1 ? 1 : t) }')
        for ((I = 0; I < TENTHS; I++)); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.1
        done
        if ! kill -0 "$PID" 2>/dev/null && wait "$PID"; then
            OK=1
        fi
    fi
    [ "$OK" = 1 ] \
        || refuse "$USDP_DATA_MOUNT is not a readable sshfs mount (fstype '${FSTYPE:-none}'); root must remount with sshfs -o allow_other"
}

check_gpu()
{
    local LIST UUID APPS IDS BUSY="" ID
    [[ "$GPU" =~ ^(0|[1-9][0-9]*)$ ]] || refuse "bad GPU index '$GPU'; give one number, e.g. --gpu 2"
    LIST=$(timeout "$USDP_PROBE_TIMEOUT" nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null) \
        || refuse "nvidia-smi --query-gpu failed"
    UUID=$(awk -F', ' -v G="$GPU" '$1 == G { print $2 }' <<< "$LIST")
    [[ "$UUID" == GPU-* ]] || refuse "GPU $GPU not found by nvidia-smi"
    APPS=$(timeout "$USDP_PROBE_TIMEOUT" nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) \
        || refuse "nvidia-smi --query-compute-apps failed"
    if grep -qxF "$UUID" <<< "$APPS"; then
        BUSY="a running process"
    fi
    IDS=$(docker ps -q --filter "label=usdp.gpu=$GPU") \
        || refuse "docker ps failed, so run.sh cannot tell whether GPU $GPU is in use"
    for ID in $IDS; do
        BUSY="usdp container $ID"
    done
    [ -n "$BUSY" ] || return 0
    [ "$FORCE" = 1 ] || refuse "GPU $GPU is in use by $BUSY; pick an idle GPU or pass --force before --"
    echo "run.sh: --force: starting on GPU $GPU although it is in use by $BUSY" >&2
}

docker_args()
{
    ARGS=(run --rm --init --user "$(id -u):$(id -g)" --ipc host
        --mount "type=bind,source=$REPO,target=$REPO"
        --mount "type=bind,source=/mnt,target=/mnt,bind-propagation=rslave"
        --mount "type=bind,source=$HOME/.cache/torch,target=$IMG_HOME/.cache/torch"
        -w "$REPO" -e TINI_KILL_PROCESS_GROUP=1 "${MOUNT_ARGS[@]}")
    if [ "$OFFLINE" = 1 ]; then
        ARGS+=(--network none)
    fi
    if [ "$GPU_SET" = 1 ]; then
        ARGS+=(--gpus "\"device=$GPU\"" --label "usdp.gpu=$GPU" -e USDP_GPU_CHECK=1)
    else
        echo "run.sh: GPU: none (CPU run)" >&2
    fi
    if [ -n "$NAME" ]; then
        ARGS+=(-d --name "usdp-$NAME" -e "USDP_LOG=$LOG_IN")
    elif [ -t 0 ] && [ -t 1 ]; then
        ARGS+=(-it)
    elif [ ! -t 0 ]; then
        ARGS+=(-i)
    fi
    ARGS+=("$USDP_IMAGE" "$REPO/docker/entry.sh" "${CMD[@]}")
}

main()
{
    local TOOL
    parse_args "$@"
    [[ "$USDP_PROBE_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v T="$USDP_PROBE_TIMEOUT" 'BEGIN { exit !(T > 0) }' \
        || usage "USDP_PROBE_TIMEOUT must be a positive number of seconds, got '$USDP_PROBE_TIMEOUT'"
    for TOOL in docker findmnt timeout realpath awk; do
        command -v "$TOOL" > /dev/null || refuse "$TOOL missing"
    done
    if [ "$GPU_SET" = 1 ]; then
        command -v nvidia-smi > /dev/null || refuse "nvidia-smi missing"
    fi
    check_image
    check_data_mount
    check_paths
    if [ "$GPU_SET" = 1 ]; then
        check_gpu
    fi
    mkdir -p "$HOME/.cache/torch"
    if [ -n "$NAME" ]; then
        mkdir -p "$(dirname "$LOG_ABS")"
        : >> "$LOG_ABS"
    fi
    docker_args
    if [ -z "$NAME" ]; then
        exec docker "${ARGS[@]}"
    fi
    docker "${ARGS[@]}" > /dev/null
    echo "run.sh: started usdp-$NAME, log $LOG_ABS (stop: docker stop usdp-$NAME)"
}

main "$@"
