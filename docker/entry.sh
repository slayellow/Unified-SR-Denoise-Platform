#!/bin/bash
# Job container entry point (docs/modules/20261019-MOD-DockerRun.md): optional one-GPU check, then the command.
# USDP_GPU_CHECK=1 requires exactly one visible CUDA device; USDP_LOG makes it a detached run that appends the
# command output and a final "[run.sh] exit <code>" line to that file.
# Everything sits in main, so bash has read the whole file before a running job could see an edited copy.
set -u

log_exit()
{
    if [ -n "${USDP_LOG:-}" ]; then
        echo "[run.sh] exit $1" >> "$USDP_LOG"
    fi
    exit "$1"
}

refuse()
{
    echo "entry.sh: refused: $1" >&2
    if [ -n "${USDP_LOG:-}" ]; then
        echo "entry.sh: refused: $1" >> "$USDP_LOG"
    fi
    log_exit 3
}

main()
{
    local RC=0 SIG=0
    # Handlers (not ignores) keep this shell alive through docker stop or Ctrl-C so it can write the exit line,
    # and remember the signal even when the probe itself survives it. TINI_KILL_PROCESS_GROUP=1 delivers the
    # signal to the whole group, the CUDA probe and the command included.
    trap 'SIG=143' TERM
    trap 'SIG=130' INT
    if [ "${USDP_GPU_CHECK:-0}" = 1 ]; then
        timeout --foreground "${USDP_CUDA_TIMEOUT:-60}" python3 -c \
            'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)' \
            || RC=$?
        [ "$SIG" = 0 ] || log_exit "$SIG"
        case "$RC" in
            0) ;;
            130 | 143) log_exit "$RC" ;;
            *) refuse "CUDA does not see exactly one GPU; check --gpu and the NVIDIA runtime" ;;
        esac
    fi
    if [ "$#" -eq 0 ]; then
        echo "entry.sh: usage: entry.sh CMD..." >&2
        exit 2
    fi
    if [ -z "${USDP_LOG:-}" ]; then
        exec "$@"
    fi
    RC=0
    "$@" >> "$USDP_LOG" 2>&1 || RC=$?
    log_exit "$RC"
}

main "$@"
exit
