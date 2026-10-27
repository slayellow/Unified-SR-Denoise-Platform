#!/bin/bash
# Acceptance gates for the usdp image on this host (docs/modules/20261019-MOD-DockerEnv.md §3.4).
#   docker/accept.sh [--gpu N] [--only TC_DOCK_09_1,TC_DRUN_01_3,...] [--out DIR]
# One line per test case: "PASS TC-..", "FAIL TC-..: <why>" or "NOT RUN TC-..: <why>".
# Exits 1 unless every selected test case passes. Outputs stay in --out (default results/docker_accept_<date>).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd -P)
RUN="$SCRIPT_DIR/run.sh"
BASE_TESTS=280
DN_ONNX_MD5=81ac40c78450489a6fac04c05569a2ec
IR_X4_MD5=5de6458e5cfce8c4d0666b54727b4294
IR_X4_1024_MD5=37d8f6ff051a97d1df9dbc2f8fa5434d
DN_DLC=BicubicPPDenoise_EO_1280x720_v1.dlc
DN_DLC_MD5=62b56d8f6afad86df29ee27ce4572b31
# The shipped DLC records the work folder convert_all.sh got from mktemp -d (R1).
DN_DLC_TMP=/tmp/tmp.nIupqkU2Au
X2_EO_GPU_MD5=f4a757f3551f194f1e0dd35944fcefa5
X2_EO_CPU_MD5=7e2fe59ffbeb1660b7f83091e63a87e5

GPU=""
ONLY=""
OUT=""
PASSED=0
NOT_PASSED=0
SUMMARY_DONE=0
PYTEST_DONE=0
PYTEST_RC=0
DN_DONE=0
IR_DONE=0
DLC_DONE=0
DLC_RC=()
GPU_EXPORT_DONE=0
GPU_EXPORT_RC=0
TRAIN_DONE=0
TRAIN_RC=0

pass_tc()
{
    echo "PASS $1"
    PASSED=$((PASSED + 1))
}

fail_tc()
{
    echo "FAIL $1: $2"
    NOT_PASSED=$((NOT_PASSED + 1))
}

not_run_tc()
{
    echo "NOT RUN $1: $2"
    NOT_PASSED=$((NOT_PASSED + 1))
}

md5_is()
{
    [ -f "$1" ] && [ "$(md5sum "$1" | cut -d' ' -f1)" = "$2" ]
}

# GPU test cases print NOT RUN without --gpu, so a CPU-only run never reads as a full pass.
need_gpu()
{
    [ -n "$GPU" ] || { not_run_tc "$1" "needs --gpu N"; return 1; }
}

same_param_encodings()
{
    python3 -c 'import json, sys
a, b = (json.load(open(p))["param_encodings"] for p in sys.argv[1:3])
sys.exit(0 if a == b else 1)' "$1" "$2" 2> /dev/null
}

# A DLC is a stored zip. QAIRT 2.41 writes uninitialized memory into the startOpIndex and endOpIndex
# fields of dlc.metadata, so those two values are masked; every other byte of every entry must match.
same_dlc()
{
    python3 -c 'import re, struct, sys, zipfile
def entries(path):
    raw = open(path, "rb").read()
    out = []
    for info in zipfile.ZipFile(path).infolist():
        name_len, extra_len = struct.unpack("<HH", raw[info.header_offset + 26:info.header_offset + 30])
        start = info.header_offset + 30 + name_len + extra_len
        data = raw[start:start + info.compress_size]
        if info.filename.startswith("dlc.metadata"):
            data = re.sub(rb"(\"(?:start|end)OpIndex\": )[0-9]+", rb"\1N", data)
        out.append((info.filename, info.compress_type, data))
    return out
sys.exit(0 if entries(sys.argv[1]) == entries(sys.argv[2]) else 1)' "$1" "$2" 2> /dev/null
}

# Prints pytest's final summary line ("3 failed, 10 passed in 1.00s"), or nothing when pytest did not run.
pytest_summary()
{
    grep -E '^[0-9]+ (passed|failed|skipped)|^[0-9]+ .* in [0-9.]+s' "$1" 2> /dev/null | tail -n 1 || true
}

# The summary line must read "<expected> passed" with no skipped, failed or error count.
pytest_summary_ok()
{
    local LINE
    LINE=$(pytest_summary "$1")
    [[ "$LINE" =~ ^$2\ passed ]] && [[ ! "$LINE" =~ (skipped|failed|error) ]] && ! grep -q '^SKIPPED' "$1"
}

pytest_once()
{
    local COLLECTED
    [ "$PYTEST_DONE" = 0 ] || return 0
    PYTEST_DONE=1
    COLLECTED=$("$RUN" --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --collect-only -q -p no:cacheprovider \
        tests/test_docker_run.py tests/test_docker_image.py 2> /dev/null | grep -oE '^[0-9]+ tests? collected' \
        | grep -oE '^[0-9]+') || COLLECTED=""
    if [[ "$COLLECTED" =~ ^[0-9]+$ ]]; then
        echo "$((BASE_TESTS + COLLECTED))" > "$OUT/pytest.expected"
    else
        echo "unknown" > "$OUT/pytest.expected"
    fi
    git -C "$REPO" status --porcelain --ignored > "$OUT/git_before.txt" 2>&1 || true
    touch "$OUT/pytest.stamp"
    "$RUN" --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
        > "$OUT/pytest.log" 2>&1 || PYTEST_RC=$?
    git -C "$REPO" status --porcelain --ignored > "$OUT/git_after.txt" 2>&1 || true
    find "$REPO" -path "$REPO/.git" -prune -o -path "$OUT" -prune -o -newer "$OUT/pytest.stamp" -print \
        > "$OUT/pytest_new_files.txt" 2>&1 || true
}

TC_DOCK_09_1_PytestAllPassNoSkip()
{
    local EXPECTED
    pytest_once
    EXPECTED=$(cat "$OUT/pytest.expected")
    if [ "$PYTEST_RC" = 0 ] && pytest_summary_ok "$OUT/pytest.log" "$EXPECTED"; then
        pass_tc TC-DOCK-09-1
    else
        fail_tc TC-DOCK-09-1 "rc $PYTEST_RC, want '$EXPECTED passed' and no skip, got '$(pytest_summary "$OUT/pytest.log")' (see $OUT/pytest.log)"
    fi
}

TC_DOCK_09_2_PytestLeavesNoFiles()
{
    pytest_once
    if [ -z "$(pytest_summary "$OUT/pytest.log")" ]; then
        fail_tc TC-DOCK-09-2 "pytest did not run (see $OUT/pytest.log)"
    elif cmp -s "$OUT/git_before.txt" "$OUT/git_after.txt" && [ ! -s "$OUT/pytest_new_files.txt" ]; then
        pass_tc TC-DOCK-09-2
    else
        fail_tc TC-DOCK-09-2 "repo changed: see $OUT/git_before.txt, git_after.txt, pytest_new_files.txt"
    fi
}

export_dn_once()
{
    [ "$DN_DONE" = 0 ] || return 0
    DN_DONE=1
    "$RUN" --offline -- bash results/denoise_release_v1/export.sh "$OUT/dn" > "$OUT/dn_export.out" 2>&1 || true
}

export_ir_once()
{
    [ "$IR_DONE" = 0 ] || return 0
    IR_DONE=1
    "$RUN" --offline -- env SCALE=4 bash results/release_v3/eval/export.sh results/release_v3/checkpoints/IR_x4.pth \
        "$OUT/ir" > "$OUT/ir_export.out" 2>&1 || true
}

TC_DOCK_10_1_DenoiseExportMatchesShipped()
{
    local SHIPPED="$REPO/results/denoise_release_v1/dn_bpp_half4_w16a16/qat/qat_model.encodings"
    export_dn_once
    if ! md5_is "$OUT/dn/qat/qat_model.onnx" "$DN_ONNX_MD5"; then
        fail_tc TC-DOCK-10-1 "DN ONNX md5 is not $DN_ONNX_MD5 (see $OUT/dn/export.log)"
    elif ! same_param_encodings "$OUT/dn/qat/qat_model.encodings" "$SHIPPED"; then
        fail_tc TC-DOCK-10-1 "DN param_encodings differ from the shipped file"
    else
        pass_tc TC-DOCK-10-1
    fi
}

TC_DOCK_10_2_IrX4ExportMatchesShipped()
{
    local PROF MD5 WHY=""
    export_ir_once
    for PROF in x4_ir x4_ir_1024; do
        MD5=$IR_X4_MD5
        [ "$PROF" = x4_ir ] || MD5=$IR_X4_1024_MD5
        if ! grep -qx "$PROF exit 0" "$OUT/ir_export.out"; then
            WHY="$WHY $PROF did not exit 0;"
        elif ! md5_is "$OUT/ir/$PROF/qat/qat_model.onnx" "$MD5"; then
            WHY="$WHY $PROF ONNX md5 is not $MD5;"
        elif ! same_param_encodings "$OUT/ir/$PROF/qat/qat_model.encodings" \
            "$REPO/results/release_v3/$PROF/qat/qat_model.encodings"; then
            WHY="$WHY $PROF param_encodings differ;"
        fi
    done
    if [ -z "$WHY" ]; then
        pass_tc TC-DOCK-10-2
    else
        fail_tc TC-DOCK-10-2 "${WHY# }"
    fi
}

TC_DOCK_10_3_ExportOutputsOwnedNoTraceback()
{
    local FOREIGN
    export_dn_once
    export_ir_once
    FOREIGN=$(find "$OUT/dn" "$OUT/ir" ! -user "$(id -un)" 2> /dev/null | head -n 3) || true
    if [ ! -d "$OUT/dn" ] || [ ! -d "$OUT/ir" ]; then
        fail_tc TC-DOCK-10-3 "export output folders missing"
    elif [ -n "$FOREIGN" ]; then
        fail_tc TC-DOCK-10-3 "files not owned by $(id -un): $FOREIGN"
    elif grep -l Traceback "$OUT/dn/export.log" "$OUT"/ir/*.log "$OUT/dn_export.out" "$OUT/ir_export.out" > /dev/null 2>&1; then
        fail_tc TC-DOCK-10-3 "a Traceback is in the export logs"
    else
        pass_tc TC-DOCK-10-3
    fi
}

# Converts copies of the shipped Denoise inputs twice; the shipped folder is never written.
convert_dn_twice()
{
    local K DIR
    [ "$DLC_DONE" = 0 ] || return 0
    DLC_DONE=1
    mkdir -p "$OUT/shim"
    cat > "$OUT/shim/mktemp" << EOF
#!/bin/bash
if [ "\$*" = -d ]; then mkdir $DN_DLC_TMP && echo $DN_DLC_TMP; else exec /usr/bin/mktemp "\$@"; fi
EOF
    chmod +x "$OUT/shim/mktemp"
    for K in 1 2; do
        DIR="$OUT/dn_convert$K"
        mkdir -p "$DIR/dn_bpp_half4_w16a16"
        DLC_RC[K]=0
        if ! { cp "$REPO/results/denoise_release_v1/convert_all.sh" "$DIR/" \
            && cp -r "$REPO/results/denoise_release_v1/dn_bpp_half4_w16a16/qat" "$DIR/dn_bpp_half4_w16a16/"; } \
            > "$OUT/dn_convert$K.log" 2>&1; then
            DLC_RC[K]=1
            continue
        fi
        "$RUN" --offline -- env PATH="$OUT/shim:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
            qairt-env bash "$DIR/convert_all.sh" > "$OUT/dn_convert$K.log" 2>&1 || DLC_RC[K]=$?
    done
}

# Prints why conversion $1 is not clean, or nothing.
# qairt-quantizer 2.41 prints "Following OPs fallback to float:" and the op list on the next
# line, or "Following OPs fallback to float." when the list is empty.
dlc_run_problem()
{
    local LOG="$OUT/dn_convert$1.log"
    if [ "${DLC_RC[$1]}" != 0 ]; then
        echo "conversion $1 exited ${DLC_RC[$1]} (see $LOG)"
    elif ! grep -q '생성 1개 / 건너뜀 0개' "$LOG"; then
        echo "conversion $1: '생성 1개 / 건너뜀 0개' is not in $LOG"
    elif grep -q 'Following OPs fallback to float:' "$LOG"; then
        echo "conversion $1: ops fall back to float: $(grep -A1 'Following OPs fallback to float:' "$LOG" | sed -n 2p)"
    elif ! grep -q 'Following OPs fallback to float\.$' "$LOG"; then
        echo "conversion $1: the empty float fallback list is not in $LOG"
    fi
}

TC_DOCK_11_1_DenoiseDlcMatchesShipped()
{
    local WHY SHIPPED="$REPO/results/denoise_release_v1/$DN_DLC"
    convert_dn_twice
    WHY=$(dlc_run_problem 1)
    if [ -n "$WHY" ]; then
        fail_tc TC-DOCK-11-1 "$WHY"
    elif ! md5_is "$SHIPPED" "$DN_DLC_MD5"; then
        fail_tc TC-DOCK-11-1 "the shipped DLC is no longer the released file (md5 $DN_DLC_MD5)"
    elif ! same_dlc "$OUT/dn_convert1/$DN_DLC" "$SHIPPED"; then
        fail_tc TC-DOCK-11-1 "the DLC differs from the shipped one outside the two op index fields"
    else
        pass_tc TC-DOCK-11-1
    fi
}

TC_DOCK_11_2_DlcConversionDeterministic()
{
    local WHY
    convert_dn_twice
    WHY=$(dlc_run_problem 2)
    if [ -n "$WHY" ]; then
        fail_tc TC-DOCK-11-2 "$WHY"
    elif ! same_dlc "$OUT/dn_convert2/$DN_DLC" "$OUT/dn_convert1/$DN_DLC"; then
        fail_tc TC-DOCK-11-2 "the two conversions differ outside the two op index fields"
    else
        pass_tc TC-DOCK-11-2
    fi
}

export_gpu_once()
{
    [ "$GPU_EXPORT_DONE" = 0 ] || return 0
    GPU_EXPORT_DONE=1
    "$RUN" --gpu "$GPU" --offline -- bash -c 'python3 examples/aimet_quantization/6_apply_qat.py \
        --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml \
        --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
        --checkpoint results/release_v3/checkpoints/EO_x2.pth --output_dir "$1" \
        --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 \
        --width 640 --height 360 --device 0 \
        && echo "triton so files: $(find ~/.triton/cache -name "*.so" | wc -l)"' _ "$OUT/x2_eo_gpu" \
        > "$OUT/x2_eo_gpu.log" 2>&1 || GPU_EXPORT_RC=$?
}

TC_DOCK_12_1_GpuExportMatchesShipped()
{
    need_gpu TC-DOCK-12-1 || return 0
    export_gpu_once
    if [ "$GPU_EXPORT_RC" != 0 ]; then
        fail_tc TC-DOCK-12-1 "export exited $GPU_EXPORT_RC (see $OUT/x2_eo_gpu.log)"
    elif ! md5_is "$OUT/x2_eo_gpu/qat_model.onnx" "$X2_EO_GPU_MD5"; then
        fail_tc TC-DOCK-12-1 "ONNX md5 is not $X2_EO_GPU_MD5 (R2: compare with the old container, §2)"
    else
        pass_tc TC-DOCK-12-1
    fi
}

TC_DOCK_12_2_GpuExportReallyOnGpu()
{
    local SO
    need_gpu TC-DOCK-12-2 || return 0
    export_gpu_once
    SO=$(grep -oE '^triton so files: [0-9]+$' "$OUT/x2_eo_gpu.log" | grep -oE '[0-9]+$') || SO=0
    if [ "$GPU_EXPORT_RC" != 0 ] || [ ! -f "$OUT/x2_eo_gpu/qat_model.onnx" ]; then
        fail_tc TC-DOCK-12-2 "export exited $GPU_EXPORT_RC (see $OUT/x2_eo_gpu.log)"
    elif md5_is "$OUT/x2_eo_gpu/qat_model.onnx" "$X2_EO_CPU_MD5"; then
        fail_tc TC-DOCK-12-2 "the ONNX equals the CPU export $X2_EO_CPU_MD5"
    elif [ "$SO" -lt 2 ]; then
        fail_tc TC-DOCK-12-2 "$SO Triton .so files were built, want at least 2"
    else
        pass_tc TC-DOCK-12-2
    fi
}

train_once()
{
    local NAME
    [ "$TRAIN_DONE" = 0 ] || return 0
    TRAIN_DONE=1
    NAME="smoke_train_$(date +%Y%m%d)"
    "$RUN" --gpu "$GPU" --offline --name "$NAME" --log "$OUT/train_dn_1ep.log" -- \
        python3 tools/train.py --config configs/train/Denoise/bicubicppdenoise.yaml --epochs 1 \
        --work_dir "$OUT/train_dn_1ep" > "$OUT/train_start.out" 2>&1 || TRAIN_RC=$?
    if [ "$TRAIN_RC" = 0 ]; then
        docker wait "usdp-$NAME" > /dev/null 2>&1 || true
    fi
}

TC_DOCK_13_1_OneEpochTrainingCompletes()
{
    local F LAST MISSING=""
    need_gpu TC-DOCK-13-1 || return 0
    train_once
    LAST=$(tail -n 1 "$OUT/train_dn_1ep.log" 2> /dev/null) || LAST=""
    for F in best.pth last.pth training_log.csv; do
        [ -f "$OUT/train_dn_1ep/$F" ] || MISSING="$MISSING $F"
    done
    if [ "$TRAIN_RC" != 0 ]; then
        fail_tc TC-DOCK-13-1 "run.sh did not start the job (rc $TRAIN_RC, see $OUT/train_start.out)"
    elif [ "$LAST" != "[run.sh] exit 0" ]; then
        fail_tc TC-DOCK-13-1 "log ends with '$LAST' (see $OUT/train_dn_1ep.log)"
    elif [ -n "$MISSING" ]; then
        fail_tc TC-DOCK-13-1 "missing in work_dir:$MISSING"
    else
        pass_tc TC-DOCK-13-1
    fi
}

TC_DOCK_13_2_TrainingOutputsOwnedOffline()
{
    local FOREIGN
    need_gpu TC-DOCK-13-2 || return 0
    train_once
    FOREIGN=$(find "$OUT/train_dn_1ep" "$OUT/train_dn_1ep.log" ! -user "$(id -un)" 2> /dev/null | head -n 3) || true
    if [ ! -d "$OUT/train_dn_1ep" ]; then
        fail_tc TC-DOCK-13-2 "work_dir $OUT/train_dn_1ep was not made"
    elif [ -n "$FOREIGN" ]; then
        fail_tc TC-DOCK-13-2 "files not owned by $(id -un): $FOREIGN"
    else
        pass_tc TC-DOCK-13-2
    fi
}

TC_DRUN_01_3_RealOutputsOwnedByCaller()
{
    local CACHE_OWNER
    CACHE_OWNER=$("$RUN" --offline -- bash -c 'touch "$1/own_probe" && stat -c %U "$HOME/.cache"' _ "$OUT" 2> /dev/null) || true
    if [ "$(stat -c %U "$OUT/own_probe" 2> /dev/null)" != "$(id -un)" ]; then
        fail_tc TC-DRUN-01-3 "a file made in the container is not owned by $(id -un)"
    elif [ "$CACHE_OWNER" != "$(id -un)" ]; then
        fail_tc TC-DRUN-01-3 "~/.cache in the container is owned by '$CACHE_OWNER'"
    else
        pass_tc TC-DRUN-01-3
    fi
}

TC_DRUN_03_3_RealContainerSeesOneGpu()
{
    local UUID LIST COUNT
    need_gpu TC-DRUN-03-3 || return 0
    UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2> /dev/null \
        | awk -F', ' -v N="$GPU" '$1 == N { print $2 }') || UUID=""
    LIST=$("$RUN" --gpu "$GPU" --offline -- nvidia-smi -L 2> "$OUT/drun_03_3_gpu.err") || LIST=""
    COUNT=$("$RUN" --offline -- python3 -c 'import torch; print(torch.cuda.device_count())' \
        2> "$OUT/drun_03_3_nogpu.err") || COUNT=""
    if [ -z "$UUID" ] || [ "$(printf '%s\n' "$LIST" | grep -c .)" != 1 ] || [[ "$LIST" != *"$UUID"* ]]; then
        fail_tc TC-DRUN-03-3 "nvidia-smi -L with --gpu $GPU printed '$LIST', want one line with '$UUID'"
    elif [ "$COUNT" != 0 ] || ! grep -q 'GPU: none' "$OUT/drun_03_3_nogpu.err"; then
        fail_tc TC-DRUN-03-3 "without --gpu torch saw '$COUNT' GPUs, want 0 and 'GPU: none' on stderr"
    else
        pass_tc TC-DRUN-03-3
    fi
}

TC_DRUN_04_4_RealDockerStopLogs143()
{
    local LAST
    if "$RUN" --offline --name stop_probe --log "$OUT/stop_probe.log" -- sleep 300 > "$OUT/stop_probe.out" 2>&1; then
        sleep 3
        docker stop usdp-stop_probe > /dev/null 2>&1 || true
    fi
    LAST=$(tail -n 1 "$OUT/stop_probe.log" 2> /dev/null) || LAST=""
    if [ "$LAST" = "[run.sh] exit 143" ]; then
        pass_tc TC-DRUN-04-4
    else
        fail_tc TC-DRUN-04-4 "log ends with '$LAST', want '[run.sh] exit 143' (see $OUT/stop_probe.out)"
    fi
}

# Bypasses run.sh: a GPU job whose container got no GPU must stop before its command runs.
TC_DRUN_05_5_RealNoGpuInjectedRefused()
{
    local RC=0
    docker run --rm --user "$(id -u):$(id -g)" -e USDP_GPU_CHECK=1 \
        --mount "type=bind,source=$REPO,target=$REPO" --mount "type=bind,source=$OUT,target=$OUT" \
        "${USDP_IMAGE:-usdp:latest}" "$REPO/docker/entry.sh" touch "$OUT/drun_05_5_ran" \
        > "$OUT/drun_05_5.out" 2> "$OUT/drun_05_5.err" || RC=$?
    if [ "$RC" = 3 ] && grep -q 'entry.sh: refused' "$OUT/drun_05_5.err" && [ ! -e "$OUT/drun_05_5_ran" ]; then
        pass_tc TC-DRUN-05-5
    else
        fail_tc TC-DRUN-05-5 "rc $RC, want 3 with 'entry.sh: refused' and the command not run (see $OUT/drun_05_5.err)"
    fi
}

ALL_TCS=(
    TC_DOCK_09_1_PytestAllPassNoSkip
    TC_DOCK_09_2_PytestLeavesNoFiles
    TC_DOCK_10_1_DenoiseExportMatchesShipped
    TC_DOCK_10_2_IrX4ExportMatchesShipped
    TC_DOCK_10_3_ExportOutputsOwnedNoTraceback
    TC_DOCK_11_1_DenoiseDlcMatchesShipped
    TC_DOCK_11_2_DlcConversionDeterministic
    TC_DOCK_12_1_GpuExportMatchesShipped
    TC_DOCK_12_2_GpuExportReallyOnGpu
    TC_DOCK_13_1_OneEpochTrainingCompletes
    TC_DOCK_13_2_TrainingOutputsOwnedOffline
    TC_DRUN_01_3_RealOutputsOwnedByCaller
    TC_DRUN_03_3_RealContainerSeesOneGpu
    TC_DRUN_04_4_RealDockerStopLogs143
    TC_DRUN_05_5_RealNoGpuInjectedRefused
)

main()
{
    local TC ID SHORT_IDS="" IDS=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --gpu) GPU=$2; shift 2 ;;
            --only) ONLY=$2; shift 2 ;;
            --out) OUT=$2; shift 2 ;;
            *) echo "accept.sh: usage: docker/accept.sh [--gpu N] [--only TC_A,TC_B] [--out DIR]" >&2; exit 2 ;;
        esac
    done
    for TC in "${ALL_TCS[@]}"; do
        SHORT_IDS="$SHORT_IDS,$(echo "$TC" | cut -d_ -f1-4)"
    done
    IFS=, read -r -a IDS <<< "$ONLY"
    for ID in "${IDS[@]}"; do
        [[ "$SHORT_IDS," == *",$ID,"* ]] || { echo "accept.sh: unknown test case $ID" >&2; exit 2; }
    done
    OUT=$(realpath -m -- "${OUT:-$REPO/results/docker_accept_$(date +%Y%m%d)}")
    [[ "$OUT" == "$REPO"/* || "$OUT" == /mnt/* ]] \
        || { echo "accept.sh: --out $OUT must be under $REPO or /mnt, which the job containers mount" >&2; exit 2; }
    [ ! -e "$OUT" ] || { echo "accept.sh: $OUT exists; pass --out <new folder>" >&2; exit 1; }
    mkdir -p "$OUT"
    trap 'RC=$?; [ "$SUMMARY_DONE" = 1 ] || echo "accept.sh: aborted before the summary (exit $RC); only the cases above ran" >&2' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    for TC in "${ALL_TCS[@]}"; do
        if [ -z "$ONLY" ] || [[ ",$ONLY," == *",$(echo "$TC" | cut -d_ -f1-4),"* ]]; then
            "$TC"
        fi
    done
    echo "accept.sh: $PASSED passed, $NOT_PASSED not passed (outputs in $OUT)"
    SUMMARY_DONE=1
    [ "$NOT_PASSED" = 0 ]
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
