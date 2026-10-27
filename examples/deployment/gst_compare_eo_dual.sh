#!/bin/bash
# Both EO sensors on one board, one mosaic over RTSP: the MC-G105 EO camera through denoise
# then Bicubic++ (top row) and the 4K USB camera through Bicubic++ only (bottom row), one
# column per scale. RECORD=1 keeps only what the two sensors deliver -- the MC-G105 frames
# as raw NV12 and the 4K camera's MJPEG stream as it arrives -- for later offline use; the
# mosaic itself is not recorded. Owner request 2026-10-13.
#
#   usage: [DENOISE=<dlc>] [RECORD=1] ./gst_compare_eo_dual.sh [2] [3] [4]
#          no scale given: 2 3 4
#
#   port 20000 : mosaic of 1280x720 tiles, each labelled at its top-left corner,
#                (1280 x number of scales) x 1440 -- 3840x1440 for 2 3 4
#
#   +---------------------------------------+-------------------+-------------------+
#   | EO_DN_BicubicPPDenoise[v1]+Bicubic++  | ... _x3           | ... _x4           |
#   | _v3_x2                                |                   |                   |
#   +---------------------------------------+-------------------+-------------------+
#   | EO4K_Bicubic++_v3_x2                  | EO4K_..._x3       | EO4K_..._x4       |
#   +---------------------------------------+-------------------+-------------------+
#
# Models (all must exist, or the script stops before anything starts):
#   DENOISE    $MODEL_DIR/BicubicPPDenoise_EO_1280x720_v1.dlc, DENOISE_LAYER
#              final_act, label DENOISE_NAME (BicubicPPDenoise[v1]); for the deployed v2 use
#              DENOISE=$MODEL_DIR/SVFocusDenoiseNet_1280x720_v2.dlc DENOISE_LAYER=n4_5 DENOISE_NAME=v2
#   Bicubic++  $MODEL_DIR/BicubicPP_EO_<W>x<H>_x<s>_$BICUBICPP_VER.dlc per scale, for both rows
# Names and versions are letters, digits, _ and [] only (they go into the labels).
#
# Top row, MC-G105 (as gst_compare_eo_denoise.sh): qtiqmmfsrc camera=0 1920x1080 NV12 ->
# centred 1280x720 crop -> denoise -> centred SR-input crop -> Bicubic++ -> trimmed to 1280x720.
# Bottom row, 4K camera (as gst_compare_eo4k.sh, whose device-measured front-end rules apply:
# MJPEG -> jpegdec -> videoconvert into a sysmem tee, a videoconvert bridge per branch, and the
# crop done inside qtivtransform from the full 4K frame): centred SR-input crop of the
# 3840x2160 frame -> Bicubic++ -> trimmed to 1280x720. All crops are native pixels; the two
# rows look through different lenses, so they do not show the same field of view.
#
#   scale   SR input   crop in MC-G105 1280x720   crop in 4K 3840x2160   model out   trim
#   ----------------------------------------------------------------------------------------
#   2       640x360    320,180                    1600,900               1280x720    0
#   3       428x240    426,240                    1706,960               1284x720    2
#   4       320x180    480,270                    1760,990               1280x720    0
#
# DSP load per pair of camera frames: one denoise plus two Bicubic++ per scale on the one HTP
# (burst, board 98 on 2026-10-13: half4 W16 denoise 8.2 ms, v2 6.6 ms; Bicubic++ v3 x2 3.9,
# x3 5.5, x4 3.4 ms from results/release_v3/MANIFEST.md), about 34 ms for 2 3 4 and 16 ms for
# 2; the CPU also decodes the 4K MJPEG. Tiles that fall behind drop frames in their leaky
# queues. qtivcomposer waits MOSAIC_LATENCY ms (default 33) for a late tile. The two cameras
# run free, so a row pair is not one instant.
#
# RECORD=1 writes $OUT_ROOT/<STAMP>_gst_compare_eo_dual_x<scales>/ with
#   eo_1920x1080.nv12        MC-G105 frames as delivered (NV12, 3110400 bytes per frame,
#                            back to back; the format the P1 tools read: score_eo_chart.read_frame)
#   eo4k_3840x2160.mjpeg     the 4K camera's JPEG frames as delivered, back to back
#                            (ffmpeg -f mjpeg -i eo4k_3840x2160.mjpeg ...)
#   info.txt                 start time, camera settings, model list
# The recording branches do not drop frames: if storage cannot keep up they hold their camera
# back. That is about 93 MB/s for the NV12 stream plus the MJPEG stream; the banner prints the
# free space. STAMP defaults to the board's date-time (the boards' clocks come up in 2000;
# sync with date -s or pass STAMP=20261013_140000); an existing folder stops the script. Stop
# with Ctrl-C: gst-launch -e turns that into EOS so both files end on a whole frame.
#
# TRANSPORT=rtsp (default): gst-rtsp-server on the mosaic port, fed over loopback UDP; the
# banner prints the Mac command:
#   open -n -a VLC --args --rtsp-tcp rtsp://<board>:20000
# TRANSPORT=udp: plain RTP to HOST on the same port. PORT_OFFSET shifts the port.
#
# Other knobs: MODEL_DIR (/data/dlc), DEVICE (/dev/video4), BICUBICPP_VER (v3), BPP_LAYER
# (final_act), OUT_ROOT (/data), MOSAIC_BITRATE (6 Mbit/s per tile), NET_QUEUE, ENC_OPTS,
# LABEL (1), PROBE=1 (identity probe_<pad> in front of every composer pad for fps counts),
# BOARD_IP, HOST.
#
# Run it alone: it refuses to start while any gst-launch-1.0, another qtiqmmfsrc pipeline, a
# v4l2src on DEVICE or an snpe-net-run runs, since that would hold a camera or share the DSP.

set -e
# The pipeline text is passed to gst-launch unquoted, and a label such as
# BicubicPPDenoise[v1] would otherwise be a filename pattern.
set -f

SCALES=${*:-2 3 4}
HOST=${HOST:-192.168.100.112}
TRANSPORT=${TRANSPORT:-rtsp}
MODEL_DIR=${MODEL_DIR:-/data/dlc}
DEVICE=${DEVICE:-/dev/video4}
DENOISE=${DENOISE:-$MODEL_DIR/BicubicPPDenoise_EO_1280x720_v1.dlc}
DENOISE_LAYER=${DENOISE_LAYER:-final_act}
DENOISE_NAME=${DENOISE_NAME:-BicubicPPDenoise[v1]}
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
EO_W=1920; EO_H=1080
K4_W=3840; K4_H=2160

# geom <scale> <frame w> <frame h>: SR input size, its centred crop in the frame and the output trim
geom() {
    case "$1" in
        2) IN_W=640; IN_H=360 ;;
        3) IN_W=428; IN_H=240 ;;
        4) IN_W=320; IN_H=180 ;;
    esac
    CROP_X=$(( ($2 - IN_W) / 2 ))
    CROP_Y=$(( ($3 - IN_H) / 2 ))
    OUT_W=$(( IN_W * $1 ))
    OUT_H=$(( IN_H * $1 ))
    TRIM_X=$(( (OUT_W - 1280) / 2 ))
    BICUBICPP=$MODEL_DIR/BicubicPP_EO_${IN_W}x${IN_H}_x$1_${BICUBICPP_VER}.dlc
}

case "$BICUBICPP_VER" in ''|*[!A-Za-z0-9_]*) echo "bad version: [$BICUBICPP_VER] (letters, digits, _)" >&2; exit 1 ;; esac
case "$DENOISE_NAME" in ''|*[!][A-Za-z0-9_]*) echo "bad denoise name: [$DENOISE_NAME] (letters, digits, _, [])" >&2; exit 1 ;; esac
NSCALES=0; SCALE_TAG=""
for s in $SCALES; do
    case "$s" in 2|3|4) ;; *) echo "usage: $0 [2] [3] [4]  (got scale [$s])" >&2; exit 1 ;; esac
    case " $SCALE_TAG " in *" $s "*) echo "scale $s given twice" >&2; exit 1 ;; esac
    SCALE_TAG="$SCALE_TAG $s"
    NSCALES=$(( NSCALES + 1 ))
done
SCALE_TAG=$(echo $SCALE_TAG | tr -d ' ')
MOSAIC_BITRATE=${MOSAIC_BITRATE:-$(( 6000000 * 2 * NSCALES ))}

# MC-G105 1920x1080 -> centred 1280x720 fed to the denoise net
DN_X=$(( (EO_W - 1280) / 2 ))
DN_Y=$(( (EO_H - 720) / 2 ))

BOARD_IP=${BOARD_IP:-$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)}
BOARD_IP=${BOARD_IP:-<board-ip>}
PORT_MOSAIC=$(( 20000 + PORT_OFFSET ))

# --- models: every DLC is required
[ -f "$DENOISE" ] || { echo "model not found: $DENOISE (denoise $DENOISE_NAME; DENOISE= picks another)" >&2; exit 1; }
for s in $SCALES; do
    geom "$s" 1280 720
    [ -f "$BICUBICPP" ] || { echo "model not found: $BICUBICPP (Bicubic++ $BICUBICPP_VER x$s; BICUBICPP_VER picks the version)" >&2; exit 1; }
done

# A second pipeline on either camera makes gst fail cryptically (or stall), and any other gst
# or snpe-net-run job shares the DSP; name the culprit before the recording folder exists.
BUSY=$(ps -ef | grep -a -E -e "[q]tiqmmfsrc" -e "[v]4l2src device=$DEVICE" -e "(^|[ /])[g]st-launch-1\.0( |\$)" -e "[s]npe-net-run" | head -1)
if [ -n "$BUSY" ]; then
    echo "ERROR: another pipeline holds a camera or shares the DSP; stop it (kill <PID>) first:" >&2
    echo "       $BUSY" >&2
    exit 1
fi

# --- recording target: the two sensors' streams only
if [ "$RECORD" = 1 ]; then
    STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
    case "$STAMP" in 20[2-9][0-9]*) ;; *)
        echo "WARNING: board clock says $(date) -- folder will start with $STAMP. Sync with date -s or pass STAMP=" >&2 ;;
    esac
    OUT_DIR=$OUT_ROOT/${STAMP}_$(basename "$0" .sh)_x$SCALE_TAG
    if [ -e "$OUT_DIR" ]; then echo "ERROR: $OUT_DIR already exists; pass another STAMP=" >&2; exit 1; fi
    mkdir -p "$OUT_DIR"
fi

case "$TRANSPORT" in
    rtsp) SEND_HOST=127.0.0.1; echo "EO camera=0 ${EO_W}x${EO_H} + EO4K $DEVICE ${K4_W}x${K4_H} MJPEG   scales$(printf ' x%s' $SCALES)   board $BOARD_IP -> RTSP (TCP)" ;;
    udp)  SEND_HOST=$HOST;     echo "EO camera=0 ${EO_W}x${EO_H} + EO4K $DEVICE ${K4_W}x${K4_H} MJPEG   scales$(printf ' x%s' $SCALES)   board $BOARD_IP -> $HOST (UDP)" ;;
    *)    echo "TRANSPORT must be rtsp or udp, got $TRANSPORT" >&2; exit 1 ;;
esac
echo "  NET_QUEUE=[$NET_QUEUE]  ENC_OPTS=[$ENC_OPTS]"
echo "  EO denoise crop <$DN_X,$DN_Y,1280,720>  $DENOISE (layer $DENOISE_LAYER)"

export WAYLAND_DISPLAY=wayland-1
export XDG_RUNTIME_DIR=/run/user/root

# encoder_out <port>: the RTP sender after the qtic2venc (the mosaic is streamed, not recorded)
encoder_out() {
    printf '%s' "queue $NET_QUEUE ! h264parse config-interval=-1 ! rtph264pay name=pay_$1 mtu=1350 ! \
      udpsink name=sink_$1 host=$SEND_HOST port=$1 sync=false async=false"
}

# eo_sr_branch: one Bicubic++ consumer of the denoise tee at the scale geom set last, up to the composer
eo_sr_branch() {
    printf '%s' "dn. ! queue leaky=2 max-size-buffers=2 ! \
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

# k4_sr_branch: one Bicubic++ consumer of the 4K tee at the scale geom set last, up to the
# composer -- gst_compare_eo4k.sh's model branch (videoconvert bridge, crop inside qtivtransform
# from the full frame, no caps forced on its output)
k4_sr_branch() {
    printf '%s' "k4. ! queue leaky=2 max-size-buffers=2 ! \
      videoconvert ! qtivtransform crop=<$CROP_X,$CROP_Y,$IN_W,$IN_H> ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$BICUBICPP layers=<$BPP_LAYER> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform crop=<$TRIM_X,0,1280,720> ! \
      video/x-raw(memory:GBM),format=NV12,width=1280,height=720 ! "
}

# tile <x> <y> <label> <banner text> <branch>: label the tile at <x,y>, print it, and make the
# branch (ending in "! ") the composer's next sink pad there.
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

MOS_W=$(( 1280 * NSCALES ))
MOS_H=1440
echo "  $PORT_MOSAIC  mosaic ${MOS_W}x${MOS_H}, target-bitrate $MOSAIC_BITRATE, latency ${MOSAIC_LATENCY} ms, labels LABEL=$LABEL"
# top row: MC-G105 -> denoise -> tee dn -> Bicubic++ per scale
DN_BRANCH="eo. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop=<$DN_X,$DN_Y,1280,720> ! \
      video/x-raw(memory:GBM),format=RGBA,width=1280,height=720 ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$DENOISE layers=<$DENOISE_LAYER> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=1280,height=720 ! \
      tee name=dn"
MODEL_LINES="denoise $DENOISE_NAME $(md5sum "$DENOISE" | cut -d' ' -f1) $DENOISE layer=$DENOISE_LAYER
"
x=0
for s in $SCALES; do
    geom "$s" 1280 720
    tile "$x" 0 "EO_DN_${DENOISE_NAME}+Bicubic++_${BICUBICPP_VER}_x$s" \
        "MC-G105 denoise -> crop <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}, trim x=$TRIM_X" "$(eo_sr_branch)"
    MODEL_LINES="${MODEL_LINES}sr_x$s BicubicPP_$BICUBICPP_VER $(md5sum "$BICUBICPP" | cut -d' ' -f1) $BICUBICPP layer=$BPP_LAYER
"
    x=$(( x + 1280 ))
done
# bottom row: 4K camera -> Bicubic++ per scale
x=0
for s in $SCALES; do
    geom "$s" "$K4_W" "$K4_H"
    tile "$x" 720 "EO4K_Bicubic++_${BICUBICPP_VER}_x$s" \
        "4K crop <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}, trim x=$TRIM_X" "$(k4_sr_branch)"
    x=$(( x + 1280 ))
done

OVERLAY=""
if [ "$LABEL" = 1 ]; then OVERLAY="qtivoverlay strings={$LABELS} !"; fi
# the composer comes before the tiles that link to its request pads
MOSAIC="qtivcomposer name=comp background=255 latency=$(( MOSAIC_LATENCY * 1000000 )) $TILE_PADS ! \
      video/x-raw(memory:GBM),format=NV12,width=$MOS_W,height=$MOS_H ! $OVERLAY \
      qtic2venc control-rate=1 $ENC_OPTS target-bitrate=$MOSAIC_BITRATE ! $(encoder_out "$PORT_MOSAIC")"

# raw branches: no leaky queue, so no frame is dropped; the copy to system memory comes right
# after a 2-buffer queue (camera GBM buffers are few) and the deep queue holds system-memory copies
RAW_EO=""; RAW_K4=""
if [ "$RECORD" = 1 ]; then
    RAW_EO="eo. ! queue max-size-buffers=2 max-size-time=0 max-size-bytes=0 ! \
      qtivtransform ! video/x-raw,format=NV12,width=$EO_W,height=$EO_H ! \
      queue max-size-buffers=30 max-size-time=0 max-size-bytes=0 ! \
      filesink location=$OUT_DIR/eo_${EO_W}x${EO_H}.nv12 sync=false"
    RAW_K4="mj. ! queue max-size-buffers=60 max-size-time=0 max-size-bytes=0 ! \
      filesink location=$OUT_DIR/eo4k_${K4_W}x${K4_H}.mjpeg sync=false"
    {
        echo "# gst_compare_eo_dual.sh raw capture  start $(date '+%F %T')  board $BOARD_IP"
        echo "eo_${EO_W}x${EO_H}.nv12      MC-G105 qtiqmmfsrc camera=0, NV12 ${EO_W}x${EO_H} 30 fps, $(( EO_W * EO_H * 3 / 2 )) bytes per frame, frames back to back"
        echo "eo4k_${K4_W}x${K4_H}.mjpeg   4K USB camera $DEVICE, MJPEG ${K4_W}x${K4_H} 30 fps as delivered, JPEG frames back to back"
        echo "# mosaic shown while recording: scales $SCALES, models:"
        printf '%s' "$MODEL_LINES"
    } > "$OUT_DIR/info.txt"
    FREE=$(df -m "$OUT_ROOT" | awk 'NR==2 {print $4}')
    echo "  recording raw to $OUT_DIR/ (eo_${EO_W}x${EO_H}.nv12 ~93 MB/s + eo4k_${K4_W}x${K4_H}.mjpeg; ${FREE} MB free; Ctrl-C to finish cleanly)"
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
  capsfilter caps="video/x-raw(memory:GBM),format=NV12,framerate=30/1,width=$EO_W,height=$EO_H" ! \
  tee name=eo \
  v4l2src device="$DEVICE" ! \
  "image/jpeg,width=$K4_W,height=$K4_H,framerate=30/1" ! \
  tee name=mj \
  mj. ! queue ! jpegdec ! queue ! videoconvert ! tee name=k4 \
  $MOSAIC \
  $DN_BRANCH \
  $TILES \
  $RAW_EO \
  $RAW_K4
