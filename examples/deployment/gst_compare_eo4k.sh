#!/bin/bash
# Compare the EO SR models on the same crop of the 4K USB camera, with a plain
# upscale as the anchor, as one 2x2 mosaic over RTSP so a room sees every tile
# on the same camera frame. No denoise stage: this sensor is clean enough that
# the denoise-then-upscale chain is not part of the comparison (and the
# deployed denoise DLC is fixed at 1280x720 anyway).
#
#   usage: [BICUBICPP_VER=v3] [SVFOCUS_VER=v2] ./gst_compare_eo4k.sh [2|3|4]
#
#   port 30000 : mosaic of 1280x720 tiles, each labelled at its top-left corner (the 4K camera
#                scripts use 30000, the MC-G105 ones 20000; owner, 2026-10-13)
#
#   x2, x4  2560x1440                       x3  1280x1440
#   +--------------------+--------------+   +--------------------+
#   | videoscale         | camera       |   | videoscale         |
#   | (CPU anchor)       | (full 4K)    |   | (CPU anchor)       |
#   +--------------------+--------------+   +--------------------+
#   | SVFocusSRNet_<ver> | Bicubic++_   |   | Bicubic++_<ver>    |
#   |                    | <ver>        |   |                    |
#   +--------------------+--------------+   +--------------------+
#
# Models: $MODEL_DIR/BicubicPP_EO_<W>x<H>_x<s>_$BICUBICPP_VER.dlc (default v3)
# must exist, or the script stops before anything starts; a v4 runs as soon as
# its DLC is there (BICUBICPP_VER=v4). SVFocusSRNet_EO_<W>x<H>_x<s>_$SVFOCUS_VER.dlc
# (default v2) is probed at x2/x4; without it its tile stays empty under a
# "SVFocusSRNet_<ver>_DLC_missing" label. x3 shows the anchor over Bicubic++ only.
# Labels carry no spaces: the overlay strings reach gst-launch word split, as in
# the reference, and a space would cut the property value (hence the version
# names are limited to letters, digits and _).
#
# Front end -- device-measured rules shape this pipeline (violating one leaves
# it at PLAYING with zero frames, or kills it ~1s in):
#   1. Frames are born in CPU memory (MJPEG -> jpegdec -> videoconvert). A
#      Qualcomm element on this path needs a videoconvert feeding it directly
#      (a queue survives the link, a tee does not), so every branch ends its
#      CPU section with its own videoconvert bridge.
#   2. Never force qtivtransform's output with a capsfilter (GBM or plain) --
#      it then self-allocates and dies (bo_create / PRIME import errors). Its
#      consumer must drive the negotiation.
#   3. The model branches must let qtivtransform CROP FROM THE FULL 4K frame
#      (videoconvert ! qtivtransform crop=<...> ! qtimlvconverter). Feeding a
#      pre-cropped 428x240 or 320x180 frame into that hop fails
#      deterministically (0/4 trials; only the 640x360 x2 crop survived),
#      while the crop-inside-qtivtransform form passes 2/2 at every scale --
#      the converter only cooperates on pool setup when qtivtransform owns
#      the geometry. The bicubic anchor keeps CPU geometry (videocrop/
#      videoscale) since it worked at all scales from the start.
#   4. The 4K tee stays sysmem: buffers cross it as shared refs, and only the
#      preview/model bridges lift what they need (an early all-GBM variant
#      OOM-killed at 3.1 GB RSS).
#
# Mosaic tiles keep those branches; only what followed the last element changed,
# from an encoder to a qtivcomposer sink pad:
#   anchor   ... videoconvert ! qtivtransform ! comp.sink_N      (rule 2: no caps;
#   camera   videoconvert ! qtivtransform ! comp.sink_N           the composer
#            (the old 4K preview bridge: the 3840x2160 frame       negotiates)
#            reaches the composer, whose sink pad dimensions=<1280,720> scales it)
#   models   ... trim qtivtransform ! GBM NV12 1280x720 ! comp.sink_N (as before)
# Not yet run on the board: the composer taking the unforced qtivtransform
# outputs (above all the 4K one). If the camera tile breaks the pipeline,
# CAMERA_TILE=0 leaves it empty under its label and runs the rest.
#
# Geometry, all crops centred in the 3840x2160 frame (native pixels, no
# scaling -- note this is a narrow field of view: a 640x360 crop is 1/36 of
# the 4K frame area)
#
#   scale   SR input   crop              model out   output trim
#   ----------------------------------------------------------------
#   2       640x360    1600,900          1280x720    0
#   3       428x240    1706,960          1284x720    2  (1280 not divisible by 3)
#   4       320x180    1760,990          1280x720    0
#
# Colour handling: the tee carries whatever jpegdec emits (I420 sysmem); each
# branch converts as its consumer negotiates, NV12 landing only at the composer.
#
# Composer, labels, encoder, RTSP relay and recording are
# the mosaic output path first run on board 98 on 2026-10-06,
# see its header: one qtivcomposer (waits MOSAIC_LATENCY ms, default 33, for a
# late tile; raise it if the model tiles trail the anchor), one qtivoverlay
# writing the labels (LABEL=0 leaves it out), one qtic2venc at MOSAIC_BITRATE
# (default 24000000), the encoder output teed to the RTP sender and the file
# writer.
# An empty tile shows the composer's background.
#
# TRANSPORT=rtsp (default): gst-rtsp-server on the mosaic port, fed over loopback
# UDP on the same port number, so the viewer pulls it over TCP and a lost packet
# is resent. The banner prints the Mac command:
#   open -n -a VLC --args --rtsp-tcp rtsp://<board>:30000
# TRANSPORT=udp: plain RTP to HOST on the same port. PORT_OFFSET shifts the port
# (PORT_OFFSET=98 -> 30098) to run both boards against one viewer.
#
# RECORD=1 also writes $OUT_ROOT/<STAMP>_gst_compare_eo4k_x<s>/mosaic.mp4 (.h264
# without mp4mux) and models.txt (tile, label, DLC md5 and path). STAMP defaults
# to the board's date-time; an existing folder stops the script. Stop with
# Ctrl-C: gst-launch -e turns that into EOS, which is what lets mp4mux write the
# index -- a killed process leaves an mp4 without one. The boards' clocks are
# not battery backed and come up in 2000; sync first from a machine with the time:
#   ssh root@<board> "date -s '$(date '+%Y-%m-%d %H:%M:%S')'"
# or pass STAMP=20261006_140000 explicitly.
#
# Network knobs:
#   NET_QUEUE  options for the queue in front of the RTP sender (default leaky,
#              so a stalled network drops packets there instead of stalling the
#              encoder, the recording and every tile). NET_QUEUE="" blocks.
#   ENC_OPTS   extra qtic2venc properties; MOSAIC_BITRATE comes after them.
#
# Other knobs: DEVICE (/dev/video4), MODEL_DIR (/data/dlc), SVFOCUS_LAYER and
# BPP_LAYER (final_act), DELEGATE (dsp = Hexagon HTP, the deployed path; gpu =
# Adreno in float, but a quantized DLC keeps its rounded weights there),
# OUT_ROOT (/data), BOARD_IP (first global IPv4, for the printed URL), HOST (UDP).
#
# Run it alone: it refuses to start while any gst-launch-1.0, a v4l2src on DEVICE
# or an snpe-net-run (board evaluation, bench_dlc.sh) runs, since that would hold the
# camera or share the DSP.

set -e

SCALE=${1:-2}
HOST=${HOST:-192.168.100.112}
TRANSPORT=${TRANSPORT:-rtsp}
DEVICE=${DEVICE:-/dev/video4}
MODEL_DIR=${MODEL_DIR:-/data/dlc}
SVFOCUS_LAYER=${SVFOCUS_LAYER:-final_act}
SVFOCUS_VER=${SVFOCUS_VER:-v2}
BICUBICPP_VER=${BICUBICPP_VER:-v3}
DELEGATE=${DELEGATE:-dsp}
BPP_LAYER=${BPP_LAYER:-final_act}
RECORD=${RECORD:-0}
OUT_ROOT=${OUT_ROOT:-/data}
PORT_OFFSET=${PORT_OFFSET:-0}
NET_QUEUE=${NET_QUEUE-leaky=2 max-size-buffers=30 max-size-time=0 max-size-bytes=0}
ENC_OPTS=${ENC_OPTS:-}
MOSAIC_BITRATE=${MOSAIC_BITRATE:-24000000}
MOSAIC_LATENCY=${MOSAIC_LATENCY:-33}
LABEL=${LABEL:-1}
CAMERA_TILE=${CAMERA_TILE:-1}
SRC_W=3840
SRC_H=2160
TILE_W=1280
TILE_H=720

case "$SCALE" in
    2) IN_W=640; IN_H=360 ;;
    3) IN_W=428; IN_H=240 ;;
    4) IN_W=320; IN_H=180 ;;
    *) echo "usage: $0 [2|3|4]" >&2; exit 1 ;;
esac

CROP_X=$(( (SRC_W - IN_W) / 2 ))
CROP_Y=$(( (SRC_H - IN_H) / 2 ))
OUT_W=$(( IN_W * SCALE ))
OUT_H=$(( IN_H * SCALE ))
TRIM_X=$(( (OUT_W - TILE_W) / 2 ))

# --- who am I: the first global IPv4 address, for the viewer URL (BOARD_IP overrides it
# on a board with more than one interface up)
BOARD_IP=${BOARD_IP:-$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)}
BOARD_IP=${BOARD_IP:-<board-ip>}
PORT_MOSAIC=$(( 30000 + PORT_OFFSET ))

# --- models: Bicubic++ is required, SVFocusSRNet is probed (x2/x4 only)
# the version becomes a file and label name
for v in "$BICUBICPP_VER" "$SVFOCUS_VER"; do
    case "$v" in ''|*[!A-Za-z0-9_]*) echo "bad version name: [$v] (letters, digits, _)" >&2; exit 1 ;; esac
done
BICUBICPP=$MODEL_DIR/BicubicPP_EO_${IN_W}x${IN_H}_x${SCALE}_${BICUBICPP_VER}.dlc
[ -f "$BICUBICPP" ] || { echo "model not found: $BICUBICPP (Bicubic++ $BICUBICPP_VER x$SCALE; BICUBICPP_VER picks the version)" >&2; exit 1; }
SVFOCUS=$MODEL_DIR/SVFocusSRNet_EO_${IN_W}x${IN_H}_x${SCALE}_${SVFOCUS_VER}.dlc

# Another pipeline holding the camera makes gst fail with a cryptic "Device busy" (or a
# silent stall), and any other gst or snpe-net-run job shares the DSP budget; name the
# culprit before the recording folder exists. The whole ps line is shown: its PID column
# differs between BusyBox and procps.
BUSY=$(ps -ef | grep -a -E -e "[v]4l2src device=$DEVICE" -e "(^|[ /])[g]st-launch-1\.0( |\$)" -e "[s]npe-net-run" | head -1)
if [ -n "$BUSY" ]; then
    echo "ERROR: another gst pipeline or snpe-net-run holds $DEVICE or shares the DSP; stop it (kill <PID>) first:" >&2
    echo "       $BUSY" >&2
    exit 1
fi

# --- recording target
if [ "$RECORD" = 1 ]; then
    STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
    case "$STAMP" in 20[2-9][0-9]*) ;; *)
        echo "WARNING: board clock says $(date) -- folder will start with $STAMP. Sync with date -s or pass STAMP=" >&2 ;;
    esac
    OUT_DIR=$OUT_ROOT/${STAMP}_$(basename "$0" .sh)_x$SCALE
    if [ -e "$OUT_DIR" ]; then echo "ERROR: $OUT_DIR already exists; pass another STAMP=" >&2; exit 1; fi
    mkdir -p "$OUT_DIR"
    if gst-inspect-1.0 mp4mux >/dev/null 2>&1; then
        EXT=mp4; MUX="h264parse ! mp4mux"
    else
        EXT=h264; MUX="h264parse"
    fi
fi

case "$TRANSPORT" in
    rtsp) SEND_HOST=127.0.0.1; echo "EO4K $DEVICE ${SRC_W}x${SRC_H} MJPEG   scale x$SCALE   board $BOARD_IP -> RTSP (TCP)" ;;
    udp)  SEND_HOST=$HOST;     echo "EO4K $DEVICE ${SRC_W}x${SRC_H} MJPEG   scale x$SCALE   board $BOARD_IP -> $HOST (UDP)" ;;
    *)    echo "TRANSPORT must be rtsp or udp, got $TRANSPORT" >&2; exit 1 ;;
esac
echo "  NET_QUEUE=[$NET_QUEUE]  ENC_OPTS=[$ENC_OPTS]  DELEGATE=$DELEGATE"
echo "  crop   <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}, trimmed to ${TILE_W}x${TILE_H} at x=$TRIM_X"

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

# sr_branch <dlc> <layer>: one model consumer of the source tee, up to the composer --
# the model branch this script always ran, minus its encoder.
sr_branch() {
    printf '%s' "src. ! queue leaky=2 max-size-buffers=2 ! \
      videoconvert ! qtivtransform crop=<$CROP_X,$CROP_Y,$IN_W,$IN_H> ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=$DELEGATE model=$1 layers=<$2> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform crop=<$TRIM_X,0,1280,720> ! \
      video/x-raw(memory:GBM),format=NV12,width=1280,height=720 ! "
}

# videocrop takes margins, not a rectangle
CL=$CROP_X
CR=$(( SRC_W - CROP_X - IN_W ))
CT=$CROP_Y
CB=$(( SRC_H - CROP_Y - IN_H ))
ANCHOR_BRANCH="src. ! queue leaky=2 max-size-buffers=2 ! \
      videocrop left=$CL right=$CR top=$CT bottom=$CB ! \
      videoscale method=4-tap ! video/x-raw,width=$OUT_W,height=$OUT_H ! \
      videocrop left=$TRIM_X right=$TRIM_X top=0 bottom=0 ! \
      videoconvert ! qtivtransform ! "
CAMERA_BRANCH="src. ! queue leaky=2 max-size-buffers=2 ! \
      videoconvert ! qtivtransform ! "

# tile <x> <y> <label> <banner text> [branch]: label the tile at <x,y> and print it; a
# branch (a source tee consumer ending in "! ") becomes the composer's next sink pad
# there, without one the tile stays empty.
TILE_PADS=""; LABELS=""; TILES=""; PAD=0
tile() {
    LABELS="$LABELS${LABELS:+,}(structure)\"Text,contents=\\\"$3\\\",fontsize=40,position=<$(( $1 + 24 )),$(( $2 + 24 ))>,color=0xFFFF00FF;\""
    echo "    <$1,$2>  $3  --  $4"
    [ -n "${5:-}" ] || return 0
    TILE_PADS="$TILE_PADS sink_$PAD::position=<$1,$2> sink_$PAD::dimensions=<$TILE_W,$TILE_H>"
    TILES="$TILES $5comp.sink_$PAD"
    PAD=$(( PAD + 1 ))
}

if [ "$SCALE" = 3 ]; then
    MOS_W=$TILE_W; BPP_X=0
else
    MOS_W=$(( 2 * TILE_W )); BPP_X=$TILE_W
fi
MOS_H=$(( 2 * TILE_H ))
echo "  $PORT_MOSAIC  mosaic ${MOS_W}x${MOS_H}, target-bitrate $MOSAIC_BITRATE, latency ${MOSAIC_LATENCY} ms, labels LABEL=$LABEL"
tile 0 0 videoscale "CPU videocrop -> videoscale 4-tap ${OUT_W}x${OUT_H} -> trim (anchor)" "$ANCHOR_BRANCH"
MODEL_LINES=""
if [ "$SCALE" != 3 ]; then
    if [ "$CAMERA_TILE" = 1 ]; then
        tile "$TILE_W" 0 camera "full ${SRC_W}x${SRC_H} frame, scaled to ${TILE_W}x${TILE_H} by the composer" "$CAMERA_BRANCH"
    else
        tile "$TILE_W" 0 camera_off "empty (CAMERA_TILE=$CAMERA_TILE)"
    fi
    if [ -f "$SVFOCUS" ]; then
        tile 0 "$TILE_H" "SVFocusSRNet_$SVFOCUS_VER" "$SVFOCUS  (layer $SVFOCUS_LAYER)" "$(sr_branch "$SVFOCUS" "$SVFOCUS_LAYER")"
        MODEL_LINES="<0,$TILE_H> SVFocusSRNet_$SVFOCUS_VER $(md5sum "$SVFOCUS" | cut -d' ' -f1) $SVFOCUS
"
    else
        tile 0 "$TILE_H" "SVFocusSRNet_${SVFOCUS_VER}_DLC_missing" "empty: $SVFOCUS not found"
    fi
fi
tile "$BPP_X" "$TILE_H" "Bicubic++_$BICUBICPP_VER" "$BICUBICPP  (layer $BPP_LAYER)" "$(sr_branch "$BICUBICPP" "$BPP_LAYER")"
MODEL_LINES="$MODEL_LINES<$BPP_X,$TILE_H> BicubicPP_$BICUBICPP_VER $(md5sum "$BICUBICPP" | cut -d' ' -f1) $BICUBICPP
"

OVERLAY=""
if [ "$LABEL" = 1 ]; then OVERLAY="qtivoverlay strings={$LABELS} !"; fi
# the composer comes before the tiles that link to its request pads
MOSAIC="qtivcomposer name=comp background=255 latency=$(( MOSAIC_LATENCY * 1000000 )) $TILE_PADS ! \
      video/x-raw(memory:GBM),format=NV12,width=$MOS_W,height=$MOS_H ! $OVERLAY \
      qtic2venc control-rate=1 $ENC_OPTS target-bitrate=$MOSAIC_BITRATE ! $(encoder_out mosaic "$PORT_MOSAIC")"

if [ "$RECORD" = 1 ]; then
    printf '# x%s  crop <%s,%s,%s,%s> -> %sx%s trim x=%s  delegate %s  board %s\n# tile model md5 dlc\n%s' \
        "$SCALE" "$CROP_X" "$CROP_Y" "$IN_W" "$IN_H" "$OUT_W" "$OUT_H" "$TRIM_X" "$DELEGATE" "$BOARD_IP" "$MODEL_LINES" > "$OUT_DIR/models.txt"
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
  v4l2src device="$DEVICE" ! \
  "image/jpeg,width=$SRC_W,height=$SRC_H,framerate=30/1" ! \
  queue ! jpegdec ! queue ! videoconvert ! \
  tee name=src \
  $MOSAIC \
  $TILES
