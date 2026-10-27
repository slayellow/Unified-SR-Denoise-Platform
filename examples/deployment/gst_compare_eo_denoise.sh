#!/bin/bash
# Denoise the same EO camera window with up to three denoise DLCs -- [v2], the deployed
# one, [v3], the SVFocusDenoiseNet W8A8 v3 C, and BicubicPPDenoise[v1], the W16A16
# candidate -- and upscale each denoised frame with Bicubic++ at every scale asked for,
# as one mosaic over RTSP: one row per denoise, one column per scale (same camera frame
# in every tile while the DSP keeps up, see DSP load).
#
#   usage: [ROWS="v2 v1"] [BICUBICPP_VER=v3] ./gst_compare_eo_denoise.sh [2] [3] [4]
#          no scale given: 2 3 4.  ROWS lists the rows top to bottom from v2, v3, v1;
#          default "v2 v1" (owner, 10-13), e.g. ROWS="v2 v3 v1" for all three
#
#   port 20000 : mosaic of 1280x720 tiles, each labelled at its top-left corner,
#                (1280 x number of scales) x (720 x number of rows) -- 3840x1440 for 2 3 4
#
#   +---------------------------+---------------------------+---------------------------+
#   | DN_v2+Bicubic++_v3_x2     | DN_v2+Bicubic++_v3_x3     | DN_v2+Bicubic++_v3_x4     |
#   +---------------------------+---------------------------+---------------------------+
#   | DN_BicubicPPDenoise[v1]+  | ... _x3                   | ... _x4                   |
#   | Bicubic++_v3_x2           |                           |                           |
#   +---------------------------+---------------------------+---------------------------+
#
# Models (the DLCs of the rows in ROWS must exist, or the script stops before anything starts):
#   v2  DENOISE_V2  $MODEL_DIR/SVFocusDenoiseNet_1280x720_v2.dlc, layer n4_5
#   v3  DENOISE_V3  $MODEL_DIR/SVFocusDenoiseNet_EO_1280x720_x1_v3c_w8a8.dlc, layer final_act
#   v1  DENOISE_V1  $MODEL_DIR/BicubicPPDenoise_EO_1280x720_v1.dlc, layer final_act
#               (HTP offline prepare with --set_output_layers final_act: the layer
#               must be the prepared one, or the graph is silently re-prepared)
#   Bicubic++   $MODEL_DIR/BicubicPP_EO_<W>x<H>_x<s>_$BICUBICPP_VER.dlc per scale
# DENOISE_V2_NAME / DENOISE_V3_NAME / DENOISE_V1_NAME (v2 / v3 / BicubicPPDenoise[v1])
# name the rows in the labels; to show another candidate as v1, point DENOISE_V1 at its
# DLC (e.g. ..._x1_fullrelu_w16a16.dlc) and rename the row. Names and versions are
# letters, digits, _ and [] only: labels carry no spaces, since gst-launch ends a {...}
# property value at an unescaped space.
#
# Each denoise net runs once at 1280x720 and its output is teed to its Bicubic++
# tiles, so a row's tiles all see the same denoised pixels -- the denoise branch
# and the Bicubic++ branch of gst_compare_eo.sh, one copy per row. The source tee
# stays NV12 with one consumer per denoise: see gst_compare_eo.sh for why RGBA starts
# only at the denoise input (GBM pool).
#
# Geometry (as gst_compare_eo.sh)
#   1920x1080 -> centred 1280x720 crop (native pixels) -> each denoise
#   denoised 1280x720 -> centred SR-input crop -> Bicubic++ -> trimmed to 1280x720
#
#   scale   SR input   crop            model out   output trim
#   -------------------------------------------------------------
#   2       640x360    320,180         1280x720    0
#   3       428x240    426,240         1284x720    2   (1280 not divisible by 3)
#   4       320x180    480,270         1280x720    0
#
# DSP load: every camera frame runs one denoise net per row plus one Bicubic++ net per
# row and scale on the one HTP. Burst latencies: denoise v2 7.1, v3 7.7 ms and the
# BicubicPPDenoise half4 W16 graph 7.9 ms (board 98, 2026-10-12, NSP about 90 C), and
# Bicubic++ v3 x2 3.9, x3 5.5, x4 3.4 ms (results/release_v3/MANIFEST.md, 2026-09-14
# record; on a hot board the 10-06 MANIFEST note saw ~46 % more). Three rows add up to
# about 61 ms for 2 3 4 and 34 ms for 2 alone, both more than one 30 fps frame; the
# leaky queues drop the frames the DSP cannot take, each branch on its own.
# Measured tile rates per row set (PROBE=1, board 98, 2026-10-12 and 10-13): the table in
# examples/deployment/README.md, section "디노이즈 v2 / v3 / BicubicPPDenoise[v1] 비교".
# With one scale the label of each tile loses its last character ("..._x" for x2):
# qtivoverlay cuts the text box short on a 1280-wide mosaic (seen on 10-12 too).
# qtivcomposer waits MOSAIC_LATENCY ms (default 33) for a late tile.
#
# Composer, labels, encoder, RTSP relay and recording are the mosaic output path of
# gst_compare_eo.sh (3840x1440 first run on board 98 on 2026-10-12): one qtivcomposer, one qtivoverlay writing the labels (LABEL=0
# leaves it out), one qtic2venc at MOSAIC_BITRATE (default 6 Mbit/s per tile, the
# per-tile rate of the 2x2 scripts), the encoder output teed to the RTP sender and
# the file writer.
#
# TRANSPORT=rtsp (default): gst-rtsp-server on the mosaic port, fed over loopback
# UDP on the same port number, so the viewer pulls it over TCP. The banner prints
# the Mac command:
#   open -n -a VLC --args --rtsp-tcp rtsp://<board>:20000
# TRANSPORT=udp: plain RTP to HOST on the same port. PORT_OFFSET shifts the port.
#
# RECORD=1 also writes $OUT_ROOT/<STAMP>_gst_compare_eo_denoise_x<scales>/mosaic.mp4
# (.h264 without mp4mux) and models.txt (row and column, model, DLC md5 and path).
# STAMP defaults to the board's date-time; an existing folder stops the script. Stop
# with Ctrl-C: gst-launch -e turns that into EOS, which lets mp4mux write the index.
# The boards' clocks come up in 2000; sync first or pass STAMP=20261012_180000:
#   ssh root@<board> "date -s '$(date '+%Y-%m-%d %H:%M:%S')'"
#
# Other knobs: MODEL_DIR (/data/dlc), BPP_LAYER (final_act), OUT_ROOT (/data),
# NET_QUEUE (leaky queue in front of the RTP sender), ENC_OPTS (extra qtic2venc
# properties), BOARD_IP (first global IPv4, for the printed URL), HOST (UDP), PROBE=1
# (an identity probe named probe_<pad> in front of every composer pad; with gst-launch
# -v each buffer it passes is printed, which is how the tile frame rates were counted).
#
# Run it alone: it refuses to start while any gst-launch-1.0, another qtiqmmfsrc
# pipeline or an snpe-net-run (board evaluation, bench_dlc.sh) runs, since that would
# hold the camera or share the DSP.

set -e
# The pipeline text is passed to gst-launch unquoted, and a row name such as
# BicubicPPDenoise[v1] would otherwise be a filename pattern.
set -f

SCALES=${*:-2 3 4}
HOST=${HOST:-192.168.100.112}
TRANSPORT=${TRANSPORT:-rtsp}
MODEL_DIR=${MODEL_DIR:-/data/dlc}
ROWS=${ROWS:-v2 v1}
DENOISE_V2=${DENOISE_V2:-$MODEL_DIR/SVFocusDenoiseNet_1280x720_v2.dlc}
DENOISE_V2_LAYER=${DENOISE_V2_LAYER:-n4_5}
DENOISE_V2_NAME=${DENOISE_V2_NAME:-v2}
DENOISE_V3=${DENOISE_V3:-$MODEL_DIR/SVFocusDenoiseNet_EO_1280x720_x1_v3c_w8a8.dlc}
DENOISE_V3_LAYER=${DENOISE_V3_LAYER:-final_act}
DENOISE_V3_NAME=${DENOISE_V3_NAME:-v3}
DENOISE_V1=${DENOISE_V1:-$MODEL_DIR/BicubicPPDenoise_EO_1280x720_v1.dlc}
DENOISE_V1_LAYER=${DENOISE_V1_LAYER:-final_act}
DENOISE_V1_NAME=${DENOISE_V1_NAME:-BicubicPPDenoise[v1]}
BICUBICPP_VER=${BICUBICPP_VER:-v3}
BPP_LAYER=${BPP_LAYER:-final_act}
RECORD=${RECORD:-0}
OUT_ROOT=${OUT_ROOT:-/data}
PORT_OFFSET=${PORT_OFFSET:-0}
NET_QUEUE=${NET_QUEUE-leaky=2 max-size-buffers=30 max-size-time=0 max-size-bytes=0}
ENC_OPTS=${ENC_OPTS:-}
MOSAIC_LATENCY=${MOSAIC_LATENCY:-33}
LABEL=${LABEL:-1}
PROBE=${PROBE:-0}

# geom <scale>: SR input size, its crop in the denoised 1280x720 and the output trim
geom() {
    case "$1" in
        2) IN_W=640; IN_H=360 ;;
        3) IN_W=428; IN_H=240 ;;
        4) IN_W=320; IN_H=180 ;;
    esac
    CROP_X=$(( (1280 - IN_W) / 2 ))
    CROP_Y=$(( (720 - IN_H) / 2 ))
    OUT_W=$(( IN_W * $1 ))
    OUT_H=$(( IN_H * $1 ))
    TRIM_X=$(( (OUT_W - 1280) / 2 ))
    BICUBICPP=$MODEL_DIR/BicubicPP_EO_${IN_W}x${IN_H}_x$1_${BICUBICPP_VER}.dlc
}

case "$BICUBICPP_VER" in ''|*[!A-Za-z0-9_]*) echo "bad version: [$BICUBICPP_VER] (letters, digits, _)" >&2; exit 1 ;; esac
# row_cfg <key>: the DLC, layer and label name of row v2, v3 or v1 in R_DLC, R_LAYER, R_NAME
row_cfg() {
    case "$1" in
        v2) R_DLC=$DENOISE_V2 R_LAYER=$DENOISE_V2_LAYER R_NAME=$DENOISE_V2_NAME ;;
        v3) R_DLC=$DENOISE_V3 R_LAYER=$DENOISE_V3_LAYER R_NAME=$DENOISE_V3_NAME ;;
        v1) R_DLC=$DENOISE_V1 R_LAYER=$DENOISE_V1_LAYER R_NAME=$DENOISE_V1_NAME ;;
        *) echo "ROWS: unknown row [$1] (v2, v3, v1)" >&2; exit 1 ;;
    esac
}
NROWS=0
SEEN=" "
SEEN_DLC=" "
for key in $ROWS; do
    row_cfg "$key"
    # the names become part of a label and of models.txt
    case "$R_NAME" in ''|*[!][A-Za-z0-9_]*) echo "bad row name: [$R_NAME] (letters, digits, _, [])" >&2; exit 1 ;; esac
    case "$SEEN" in *" $R_NAME "*) echo "row name $R_NAME used twice (ROWS=\"$ROWS\")" >&2; exit 1 ;; esac
    case "$SEEN_DLC" in *" $R_DLC "*) echo "rows share one DLC: $R_DLC (ROWS=\"$ROWS\")" >&2; exit 1 ;; esac
    SEEN="$SEEN$R_NAME "
    SEEN_DLC="$SEEN_DLC$R_DLC "
    NROWS=$(( NROWS + 1 ))
done
[ "$NROWS" -ge 1 ] || { echo "ROWS is empty; give one or more of v2 v3 v1" >&2; exit 1; }
NSCALES=0; SCALE_TAG=""
for s in $SCALES; do
    case "$s" in 2|3|4) ;; *) echo "usage: $0 [2] [3] [4]  (got scale [$s])" >&2; exit 1 ;; esac
    case " $SCALE_TAG " in *" $s "*) echo "scale $s given twice" >&2; exit 1 ;; esac
    SCALE_TAG="$SCALE_TAG $s"
    NSCALES=$(( NSCALES + 1 ))
done
SCALE_TAG=$(echo $SCALE_TAG | tr -d ' ')
MOSAIC_BITRATE=${MOSAIC_BITRATE:-$(( 6000000 * NROWS * NSCALES ))}

# 1920x1080 -> centred 1280x720 fed to every denoise net
DN_X=$(( (1920 - 1280) / 2 ))
DN_Y=$(( (1080 - 720) / 2 ))

BOARD_IP=${BOARD_IP:-$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)}
BOARD_IP=${BOARD_IP:-<board-ip>}
PORT_MOSAIC=$(( 20000 + PORT_OFFSET ))

# --- models: every DLC in use is required
for key in $ROWS; do
    row_cfg "$key"
    [ -f "$R_DLC" ] || { echo "model not found: $R_DLC (row $key, $R_NAME; DENOISE_${key^^}= picks another)" >&2; exit 1; }
done
for s in $SCALES; do
    geom "$s"
    [ -f "$BICUBICPP" ] || { echo "model not found: $BICUBICPP (Bicubic++ $BICUBICPP_VER x$s; BICUBICPP_VER picks the version)" >&2; exit 1; }
done

# Another pipeline holding the camera makes gst fail cryptically (or stall), and any
# other gst or snpe-net-run job shares the DSP budget; name the culprit before the recording folder
# exists. The whole ps line is shown: its PID column differs between BusyBox and procps.
BUSY=$(ps -ef | grep -a -E -e "[q]tiqmmfsrc" -e "(^|[ /])[g]st-launch-1\.0( |\$)" -e "[s]npe-net-run" | head -1)
if [ -n "$BUSY" ]; then
    echo "ERROR: another gst pipeline holds the camera or shares the DSP; stop it (kill <PID>) first:" >&2
    echo "       $BUSY" >&2
    exit 1
fi

# --- recording target
if [ "$RECORD" = 1 ]; then
    STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
    case "$STAMP" in 20[2-9][0-9]*) ;; *)
        echo "WARNING: board clock says $(date) -- folder will start with $STAMP. Sync with date -s or pass STAMP=" >&2 ;;
    esac
    OUT_DIR=$OUT_ROOT/${STAMP}_$(basename "$0" .sh)_x$SCALE_TAG
    if [ -e "$OUT_DIR" ]; then echo "ERROR: $OUT_DIR already exists; pass another STAMP=" >&2; exit 1; fi
    mkdir -p "$OUT_DIR"
    if gst-inspect-1.0 mp4mux >/dev/null 2>&1; then
        EXT=mp4; MUX="h264parse ! mp4mux"
    else
        EXT=h264; MUX="h264parse"
    fi
fi

case "$TRANSPORT" in
    rtsp) SEND_HOST=127.0.0.1; echo "EO camera=0 1920x1080   scales$(printf ' x%s' $SCALES)   board $BOARD_IP -> RTSP (TCP)" ;;
    udp)  SEND_HOST=$HOST;     echo "EO camera=0 1920x1080   scales$(printf ' x%s' $SCALES)   board $BOARD_IP -> $HOST (UDP)" ;;
    *)    echo "TRANSPORT must be rtsp or udp, got $TRANSPORT" >&2; exit 1 ;;
esac
echo "  NET_QUEUE=[$NET_QUEUE]  ENC_OPTS=[$ENC_OPTS]"
echo "  denoise crop <$DN_X,$DN_Y,1280,720>"

export WAYLAND_DISPLAY=wayland-1
export XDG_RUNTIME_DIR=/run/user/root

# encoder_out <name> <port>: what follows the qtic2venc -- the RTP sender, plus the file
# writer when recording. Unquoted expansion is word split into gst-launch arguments;
# caps parentheses, crop angle brackets and the label strings survive it.
encoder_out() {
    local name=$1 port=$2 rec=""
    if [ "$RECORD" = 1 ]; then
        rec="e_$name. ! queue ! $MUX ! filesink location=$OUT_DIR/$name.$EXT"
    fi
    printf '%s' "tee name=e_$name e_$name. ! queue $NET_QUEUE ! h264parse config-interval=-1 ! rtph264pay name=pay_$port mtu=1350 ! \
      udpsink name=sink_$port host=$SEND_HOST port=$port sync=false async=false $rec"
}

# denoise_branch <dlc> <layer> <tee>: the denoise consumer of the source tee, ending in
# a tee of the denoised 1280x720 RGBA frame.
denoise_branch() {
    printf '%s' "src. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop=<$DN_X,$DN_Y,1280,720> ! \
      video/x-raw(memory:GBM),format=RGBA,width=1280,height=720 ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$1 layers=<$2> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=1280,height=720 ! \
      tee name=$3 "
}

# sr_branch <tee>: one Bicubic++ consumer of a denoise tee at the scale geom set last,
# up to the composer.
sr_branch() {
    printf '%s' "$1. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop=<$CROP_X,$CROP_Y,$IN_W,$IN_H> ! \
      video/x-raw(memory:GBM),format=RGBA,width=$IN_W,height=$IN_H ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$BICUBICPP layers=<$BPP_LAYER> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform crop=<$TRIM_X,0,1280,720> ! \
      video/x-raw(memory:GBM),format=NV12,width=1280,height=720 ! "
}

# tile <x> <y> <label> <banner text> <branch>: label the tile at <x,y>, print it, and
# make the branch (ending in "! ") the composer's next sink pad there. Pads are
# numbered without gaps in link order.
TILE_PADS=""; LABELS=""; TILES=""; PAD=0
tile() {
    LABELS="$LABELS${LABELS:+,}(structure)\"Text,contents=\\\"$3\\\",fontsize=40,position=<$(( $1 + 24 )),$(( $2 + 24 ))>,color=0xFFFF00FF;\""
    echo "    <$1,$2>  $3  --  $4"
    TILE_PADS="$TILE_PADS sink_$PAD::position=<$1,$2> sink_$PAD::dimensions=<1280,720>"
    if [ "$PROBE" = 1 ]; then
        TILES="$TILES $5identity name=probe_$PAD silent=false ! comp.sink_$PAD"
    else
        TILES="$TILES $5comp.sink_$PAD"
    fi
    PAD=$(( PAD + 1 ))
}

# row <index> <name> <dlc> <layer>: one denoise net, teed to one Bicubic++ tile per scale
MODEL_LINES=""
row() {
    local y=$(( $1 * 720 )) x=0 s
    echo "  row $1 denoise $2: $3  (layer $4)"
    TILES="$TILES $(denoise_branch "$3" "$4" "dn$1")"
    MODEL_LINES="${MODEL_LINES}row$1 denoise_$2 $(md5sum "$3" | cut -d' ' -f1) $3 layer=$4
"
    for s in $SCALES; do
        geom "$s"
        tile "$x" "$y" "DN_$2+Bicubic++_${BICUBICPP_VER}_x$s" \
            "crop <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}, trim x=$TRIM_X" "$(sr_branch "dn$1")"
        x=$(( x + 1280 ))
    done
}

MOS_W=$(( 1280 * NSCALES ))
MOS_H=$(( 720 * NROWS ))
echo "  $PORT_MOSAIC  mosaic ${MOS_W}x${MOS_H}, target-bitrate $MOSAIC_BITRATE, latency ${MOSAIC_LATENCY} ms, labels LABEL=$LABEL"
ROW=0
for key in $ROWS; do
    row_cfg "$key"
    row "$ROW" "$R_NAME" "$R_DLC" "$R_LAYER"
    ROW=$(( ROW + 1 ))
done
COL=0
for s in $SCALES; do
    geom "$s"
    MODEL_LINES="${MODEL_LINES}col$COL BicubicPP_${BICUBICPP_VER}_x$s $(md5sum "$BICUBICPP" | cut -d' ' -f1) $BICUBICPP layer=$BPP_LAYER
"
    COL=$(( COL + 1 ))
done

OVERLAY=""
if [ "$LABEL" = 1 ]; then OVERLAY="qtivoverlay strings={$LABELS} !"; fi
# the composer comes before the tiles that link to its request pads
MOSAIC="qtivcomposer name=comp background=255 latency=$(( MOSAIC_LATENCY * 1000000 )) $TILE_PADS ! \
      video/x-raw(memory:GBM),format=NV12,width=$MOS_W,height=$MOS_H ! $OVERLAY \
      qtic2venc control-rate=1 $ENC_OPTS target-bitrate=$MOSAIC_BITRATE ! $(encoder_out mosaic "$PORT_MOSAIC")"

if [ "$RECORD" = 1 ]; then
    printf '# scales %s  denoise crop <%s,%s,1280,720>  mosaic %sx%s  board %s\n# row/col model md5 dlc layer\n%s' \
        "$SCALES" "$DN_X" "$DN_Y" "$MOS_W" "$MOS_H" "$BOARD_IP" "$MODEL_LINES" > "$OUT_DIR/models.txt"
    echo "  recording to $OUT_DIR/ mosaic.$EXT models.txt  (Ctrl-C to finish cleanly)"
else
    echo "  no recording (RECORD=0)"
fi

# --- RTSP relay: gst-rtsp-server serves the mosaic from the loopback RTP feed
if [ "$TRANSPORT" = rtsp ]; then
    command -v gst-rtsp-server >/dev/null 2>&1 || { echo "gst-rtsp-server not found; use TRANSPORT=udp" >&2; exit 1; }
    RTP_CAPS="application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96"
    RELAY_LOG_DIR=${OUT_DIR:-/tmp}
    if ps -ef | grep -a "[g]st-rtsp-server -a 0.0.0.0 -p $PORT_MOSAIC " >/dev/null; then
        echo "ERROR: RTSP port $PORT_MOSAIC is held by an earlier gst-rtsp-server; kill it first (ps -ef | grep gst-rtsp-server)" >&2
        exit 1
    fi
    # background jobs ignore Ctrl-C in a script, so stop them when the pipeline ends
    trap 'kill $(jobs -p) 2>/dev/null' EXIT
    gst-rtsp-server -a 0.0.0.0 -p "$PORT_MOSAIC" -m / \
        "( udpsrc address=127.0.0.1 port=$PORT_MOSAIC caps=\"$RTP_CAPS\" ! rtph264depay ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )" \
        > "$RELAY_LOG_DIR/rtsp_mosaic.log" 2>&1 &
    echo "Mac viewer (VLC, RTSP over TCP):"
    echo "  open -n -a VLC --args --rtsp-tcp rtsp://$BOARD_IP:$PORT_MOSAIC   # mosaic"
fi

gst-launch-1.0 -e -v \
  qtiqmmfsrc camera=0 ! \
  capsfilter caps="video/x-raw(memory:GBM),format=NV12,framerate=30/1,width=1920,height=1080" ! \
  tee name=src \
  $MOSAIC \
  $TILES
