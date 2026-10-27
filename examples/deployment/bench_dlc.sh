#!/bin/bash
# Time every DLC on the device with snpe-net-run.
#
#   usage: ./bench_dlc.sh [필터]        # 필터는 파일명 부분 일치: IR, x3, v4 ...
#
#   BicubicPP_IR_640x360_x2_v4                    2.048 ms      488.3 fps   (2.041~2.056)
#
# Inputs come from /data/bench, built by examples/deployment/make_bench_raw.py
# and copied over: one directory per domain and input size, holding float32
# NHWC raw frames and the input_list.txt snpe-net-run reads. A model whose
# input size has no directory is reported and skipped, so this stays usable
# while DLCs are still being converted.
#
# Each figure is the mean of REPEATS separate snpe-net-run processes, and each
# of those already averages every inference it fits into DURATION seconds. The
# two levels measure different things and both matter: the inner average smooths
# per-frame jitter, while repeating the whole process catches the run-to-run
# spread from DSP init and thermal state, which a single run hides. The range
# after each mean is the min and max across the repeats -- if it is wide, the
# mean is not worth quoting to three digits.
#
#   PERF      one profile (default burst), or "all" for the 11-profile sweep
#   REPEATS   separate runs averaged per figure (default 5)
#   DURATION  seconds per run (default 5); the input list loops for that long
#   RUNTIME   dsp (default) | cpu | gpu
#
# burst is the default because a 30 fps camera path runs each frame as a short
# isolated burst rather than a sustained load. PERF=all shows what the other
# power policies cost; run it in the background, it takes a while.

set -u

BENCH=${BENCH:-/data/bench}
MODEL_DIR=${MODEL_DIR:-/data/dlc}
DURATION=${DURATION:-5}
REPEATS=${REPEATS:-5}
PERF=${PERF:-burst}
RUNTIME=${RUNTIME:-dsp}
FILTER=${1:-}

ALL_PROFILES="burst high_performance sustained_high_performance balanced low_balanced \
default system_settings power_saver low_power_saver high_power_saver extreme_power_saver"

if [ "$PERF" = "all" ]; then PROFILES=$ALL_PROFILES; else PROFILES=$PERF; fi

case "$RUNTIME" in
    dsp) RT_FLAG=--use_dsp ;;
    gpu) RT_FLAG=--use_gpu ;;
    cpu) RT_FLAG= ;;
    *) echo "RUNTIME 은 dsp|gpu|cpu" >&2; exit 1 ;;
esac

[ -d "$BENCH" ] || { echo "입력 폴더 없음: $BENCH (make_bench_raw.py 로 만들어 scp 하세요)" >&2; exit 1; }

# 출력 텐서 폴더는 실행마다 수 GB 다. 루프가 다음 모델 전에 지우지만
# 마지막 하나가 남으므로, 어떻게 끝나든 정리되도록 걸어둔다. 지우는 폴더라
# 환경 변수로 받지 않고 BENCH 아래로 고정한다(OUT=/data 같은 값이면 그 폴더를 지웠다).
# Ctrl-C 는 루프 전체를 멈춘다: INT 에 rm 만 걸면 진행 중인 측정 하나만 끊기고
# 다음 모델로 넘어갔다.
OUT=$(cd "$BENCH" && pwd)/out
trap 'rm -rf "$OUT"' EXIT
trap 'exit 130' INT TERM

# One measurement: REPEATS runs -> "mean min max" in microseconds, or nothing.
measure() {
    _dlc=$1; _list=$2; _prof=$3
    _vals=""
    _i=0
    while [ $_i -lt "$REPEATS" ]; do
        rm -rf "$OUT"
        if (cd "$BENCH" && snpe-net-run --container "$_dlc" --input_list "$_list" \
                $RT_FLAG --perf_profile "$_prof" --duration "$DURATION" \
                --profiling_level basic --output_dir "$OUT") > /tmp/bench_run.log 2>&1; then
            _us=$(snpe-diagview --input_log "$OUT/SNPEDiag_0.log" 2>/dev/null \
                  | grep -a "Total Inference Time" | grep -oE "[0-9]+" | head -1)
            [ -n "$_us" ] && _vals="$_vals $_us"
        fi
        _i=$((_i + 1))
    done
    [ -n "$_vals" ] || return 1
    echo "$_vals" | awk '{ s=0; mn=$1; mx=$1
        for (i = 1; i <= NF; i++) { s += $i; if ($i < mn) mn = $i; if ($i > mx) mx = $i }
        printf "%.0f %d %d", s / NF, mn, mx }'
}

echo "runtime=$RUNTIME  repeats=$REPEATS x ${DURATION}s  inputs=$BENCH"
[ "$PERF" = "all" ] || printf '%-42s %10s %12s %18s\n' "model" "avg" "fps" "(min~max)"

for dlc in "$MODEL_DIR"/*.dlc; do
    [ -f "$dlc" ] || continue
    base=$(basename "$dlc" .dlc)
    case "$base" in *"$FILTER"*) ;; *) continue ;; esac

    # Names carry domain and input size: <net>_<EO|IR>_<W>x<H>_x<scale>[_ver]
    dom=$(echo "$base" | grep -oE "_(EO|IR)_" | head -1 | tr -d _)
    wh=$(echo "$base" | grep -oE "_[0-9]+x[0-9]+_" | head -1 | tr -d _)
    list=${dom}_${wh}/input_list.txt
    if [ -z "$dom" ] || [ -z "$wh" ] || [ ! -f "$BENCH/$list" ]; then
        printf '%-42s %10s\n' "$base" "입력없음"
        continue
    fi

    [ "$PERF" = "all" ] && echo "$base"
    for prof in $PROFILES; do
        if ! res=$(measure "$dlc" "$list" "$prof"); then
            if [ "$PERF" = "all" ]; then printf '  %-30s %10s\n' "$prof" "실패"
            else printf '%-42s %10s\n' "$base" "실패"; fi
            continue
        fi
        set -- $res
        ms=$(awk -v u="$1" 'BEGIN{printf "%.3f", u/1000}')
        fps=$(awk -v u="$1" 'BEGIN{printf "%.1f", 1000000/u}')
        rng=$(awk -v a="$2" -v b="$3" 'BEGIN{printf "%.3f~%.3f", a/1000, b/1000}')
        if [ "$PERF" = "all" ]; then
            printf '  %-30s %8s ms %10s fps   (%s)\n' "$prof" "$ms" "$fps" "$rng"
        else
            printf '%-42s %8s ms %10s fps   (%s)\n' "$base" "$ms" "$fps" "$rng"
        fi
    done
done
