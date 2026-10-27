#!/bin/bash
# Compare the deployed IR SR models head to head on the same crop of the IR
# camera, with a plain upscale as the anchor, as one mosaic over RTSP so a room
# sees every tile on the same camera frame.
#
#   usage: [BICUBICPP_VER=v3] [SVFOCUS_VER=v2] ./gst_compare_ir.sh [2|3|4] [720|1024]
#
#   port 20000 : mosaic of 1280x720 tiles (1280x1024 in the 1024 profile), each
#                labelled at its top-left corner
#
#   x2, x4  2560x1440 (1024: 2560x2048)          x3  1280x1440 (1024: 1280x2048)
#   +---------------------+-----------------+    +---------------------+
#   | qtivtransform       | camera          |    | qtivtransform       |
#   | (bilinear anchor)   | (full 640x512)  |    | (bilinear anchor)   |
#   +---------------------+-----------------+    +---------------------+
#   | SVFocusSRNet_<ver>  | Bicubic++_<ver> |    | Bicubic++_<ver>     |
#   +---------------------+-----------------+    +---------------------+
#
# The camera tile shows the whole 640x512 frame at the tile's height: 900x720
# centred between bars in the 720 profile, exactly 1280x1024 in the 1024 one.
#
# Two output profiles exist because the two model families were trained for
# different framings:
#
#   720   640x360 / 320x180 -> 1280x720
#   1024  640x512 / 320x256 -> 1280x1024  (x2 takes the camera's full frame)
#
# Models: $MODEL_DIR/BicubicPP_IR_<W>x<H>_x<s>_$BICUBICPP_VER.dlc (default v3)
# must exist, or the script stops before anything starts; a v4 runs as soon as
# its DLC is there (BICUBICPP_VER=v4). SVFocusSRNet is probed at x2/x4 --
# SVFocusSRNet_IR_<W>x<H>_x<s>_$SVFOCUS_VER.dlc (default v2) in the 720 profile,
# SVFocusSRNet_IR_<W>x<H>_x<s>.dlc (never versioned) in the 1024 one; without it
# its tile stays empty under a "..._DLC_missing" label. x3 shows the anchor over
# Bicubic++ only.
#
# Labels carry no spaces (Bicubic++_v3, not "Bicubic++ v3"): the labels proven on
# the board had none, and gst-launch ends a {...} property value at an unescaped
# space. Versions are therefore limited to letters, digits and _.
#
# No denoise stage. The EO denoise DLC is fixed at 1280x720 and this source is
# 640x512, so it could only run after an upscale, which would defeat the point
# of comparing upscalers. Use gst_compare_eo.sh on the EO camera when the
# denoise-then-upscale chain is what you want to look at.
#
# Memory and colour handling -- the rules gst_compare_eo4k.sh paid for
#   1. THE TEE STAYS IN SYSTEM MEMORY. It used to carry
#      video/x-raw(memory:GBM) and hand the same dma-buf to four consumers, each
#      importing it with DRM_IOCTL_PRIME_FD_TO_HANDLE. On the device that leaked
#      descriptors -- the failing fd numbers climbed 112, 119, 152, 206, 242 --
#      until gst-launch took SIGSEGV, with green flashes on the way down as
#      failed imports were rendered anyway. Buffers now cross the tee as shared
#      system-memory refs and each branch lifts only what it needs into GBM,
#      which is what the EO pipeline has always done.
#   2. A Qualcomm element on this path needs a videoconvert feeding it directly,
#      so every branch opens with its own videoconvert bridge.
#   3. The anchor is the board's own hardware scaler: qtivtransform crops and
#      upscales in one step (bilinear) to the model output size, then the same
#      trim as the models. It is built exactly like a model branch's front end
#      (videoconvert bridge, crop, forced GBM RGBA caps), which negotiates on
#      this tee. Until 2026-10-06 the anchor was CPU videocrop + videoscale
#      4-tap; the owner switched it so the mosaic compares SR against the zoom
#      the device does without SR.
#
#   The model branches keep their RGBA capsfilter after the crop. The EO
#   pipeline does without one, but here it is what the branches were already
#   negotiating correctly -- it was never part of the descriptor leak, and
#   removing it squeezed the model output into the wrong geometry on device.
#
#   Everything downstream of a model stays RGBA until the last qtivtransform
#   before the composer, which takes NV12. Promoting the tee itself to RGBA
#   gains nothing, since the sensor already delivers 4:2:0.
#
# Mosaic tiles keep the branches this script always ran; only what followed them
# changed, from an encoder to a qtivcomposer sink pad:
#   anchor   ... crop+upscale qtivtransform ! trim qtivtransform ! GBM NV12 1280xH ! comp.sink_N
#   camera   the raw preview's videoconvert ! qtivtransform ! comp.sink_N; the
#            sink pad's dimensions make the composer scale 640x512 to the tile
#   models   ... trim qtivtransform ! GBM NV12 1280xH ! comp.sink_N (as before)
# The camera output stays unforced on purpose: gst_compare_eo4k.sh
# rule 2 saw a capsfilter after a bridged qtivtransform make it self-allocate
# and die, so the consumer drives that negotiation, as the encoder did.
# Board 98 ran the composer taking the unforced qtivtransform outputs and scaling the
# camera tile on 2026-10-06 (x2/x3/x4 at 720 and 1024). If the camera tile breaks the
# pipeline, CAMERA_TILE=0 leaves it empty under its label and runs the rest.
#
# Geometry, all crops centred in the 640x512 frame
#   720   x2 <0,76,640,360>    x3 <106,136,428,240>   x4 <160,166,320,180>
#   1024  x2 <0,0,640,512>     x3 <106,84,428,342>    x4 <160,128,320,256>
#
# x3 is the only scale whose output has to be trimmed: neither 1280 nor 1024
# divides by 3, so the model runs on the next size up and the surplus is cut
# off the SR result (never off the input -- the crop the network sees must be
# exactly IN_W x IN_H).
#
#   720   428x240 -> 1284x720   trim 2px per side
#   1024  428x342 -> 1284x1026  trim 2px per side, 2 rows off the bottom
#
# The vertical surplus is dropped asymmetrically on purpose. Splitting 2 rows
# into 1 top + 1 bottom would put the crop at an odd y, and a 4:2:0 crop offset
# must be even or the chroma planes shear against the luma. Same rule applies
# to the input crop offsets, which are rounded down to even below.
#
# Composer, labels, encoder, RTSP relay and recording are
# the mosaic output path first run on board 98 on 2026-10-06,
# see its header: one qtivcomposer (waits MOSAIC_LATENCY ms, default 33, for a
# late tile; raise it if the model tiles trail the anchor), one qtivoverlay
# writing the labels (LABEL=0 leaves it out), one qtic2venc at MOSAIC_BITRATE
# (default 24000000), the encoder output teed to the RTP sender and the file
# writer.
# An empty tile and the camera tile's side bars show the composer's background
# (grey in the reference run; no colour is set here). The 1024 profile's
# 2560x2048 mosaic ran on board 98 at x2/x3/x4 (2026-10-06; the reference ran 2560x1440).
#
# TRANSPORT=rtsp (default): gst-rtsp-server on the mosaic port, fed over loopback
# UDP on the same port number, so the viewer pulls it over TCP and a lost packet
# is resent. The banner prints the Mac command:
#   open -n -a VLC --args --rtsp-tcp rtsp://<board>:20000
# TRANSPORT=udp: plain RTP to HOST on the same port. PORT_OFFSET shifts the port
# (PORT_OFFSET=98 -> 20098) to run both boards against one viewer.
#
# RECORD=1 also writes $OUT_ROOT/<STAMP>_gst_compare_ir_x<s>_<profile>/mosaic.mp4
# (.h264 without mp4mux) and models.txt (tile, model, DLC md5 and path). STAMP
# defaults to the board's date-time; an existing folder stops the script. Stop
# with Ctrl-C: gst-launch -e turns that into EOS, which is what lets mp4mux write
# the index -- a killed process leaves an mp4 without one. The boards' clocks are
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
# Other knobs: DEVICE (/dev/video2), MODEL_DIR (/data/dlc), SVFOCUS_LAYER and
# BPP_LAYER (final_act), OUT_ROOT (/data), BOARD_IP (first global IPv4, for the
# printed URL), HOST (UDP).
#
# Run it alone: it refuses to start while any gst-launch-1.0, a v4l2src on DEVICE
# or an snpe-net-run (board evaluation, bench_dlc.sh) runs, since that would hold the
# camera or share the DSP. FORCE=1 starts anyway next to other gst or snpe-net-run jobs
# (e.g. offline frame processing that never opens the IR camera); they then share the
# DSP, so tile fps and their run time can drop. A pipeline already holding DEVICE still
# stops it, FORCE or not.

set -e

SCALE=${1:-2}
PROFILE=${2:-720}
HOST=${HOST:-192.168.100.112}
TRANSPORT=${TRANSPORT:-rtsp}
DEVICE=${DEVICE:-/dev/video2}
# Every DLC -- denoise and SR, both domains -- lives flat in one folder;
# the file names already carry model, domain, resolution and scale.
MODEL_DIR=${MODEL_DIR:-/data/dlc}
SVFOCUS_LAYER=${SVFOCUS_LAYER:-final_act}
SVFOCUS_VER=${SVFOCUS_VER:-v2}
BICUBICPP_VER=${BICUBICPP_VER:-v3}
# Since v3 every Bicubic++ graph ends on the Hardtanh, so one layer name
# covers all scales -- the same node SVFocusSRNet uses.
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
FORCE=${FORCE:-0}

case "$PROFILE/$SCALE" in
    720/2)  IN_W=640; IN_H=360 ;;
    720/3)  IN_W=428; IN_H=240 ;;
    720/4)  IN_W=320; IN_H=180 ;;
    1024/2) IN_W=640; IN_H=512 ;;
    1024/3) IN_W=428; IN_H=342 ;;
    1024/4) IN_W=320; IN_H=256 ;;
    *) echo "usage: $0 [2|3|4] [720|1024]" >&2; exit 1 ;;
esac

# 4:2:0 wants even crop offsets (see the header note); every pre-x3 geometry
# was already even, so rounding down changes nothing but the 1024/3 case.
CROP_X=$(( (640 - IN_W) / 2 )); CROP_X=$(( CROP_X - CROP_X % 2 ))
CROP_Y=$(( (512 - IN_H) / 2 )); CROP_Y=$(( CROP_Y - CROP_Y % 2 ))
OUT_W=$(( IN_W * SCALE ))
OUT_H=$(( IN_H * SCALE ))

# What every tile shows. Equal to the model output except at x3.
FINAL_W=1280
FINAL_H=$PROFILE
TRIM_X=$(( (OUT_W - FINAL_W) / 2 )); TRIM_X=$(( TRIM_X - TRIM_X % 2 ))
TRIM_Y=$(( (OUT_H - FINAL_H) / 2 )); TRIM_Y=$(( TRIM_Y - TRIM_Y % 2 ))

# Applied only when there is a surplus: at x2/x4 the crop would cover the whole
# frame anyway, and the working pipelines stay byte-for-byte as they were.
TRIM=""
if [ "$OUT_W" != "$FINAL_W" ] || [ "$OUT_H" != "$FINAL_H" ]; then
    TRIM="crop=<$TRIM_X,$TRIM_Y,$FINAL_W,$FINAL_H>"
fi

# The camera tile: the full 640x512 frame at the tile's height, centred in the tile.
CAM_H=$FINAL_H
CAM_W=$(( 640 * FINAL_H / 512 ))
CAM_X=$(( FINAL_W + (FINAL_W - CAM_W) / 2 ))

# --- who am I: the first global IPv4 address, for the viewer URL (BOARD_IP overrides it
# on a board with more than one interface up)
BOARD_IP=${BOARD_IP:-$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)}
BOARD_IP=${BOARD_IP:-<board-ip>}
PORT_MOSAIC=$(( 20000 + PORT_OFFSET ))

# --- models: Bicubic++ is required, SVFocusSRNet is probed (x2/x4 only)
# the version becomes a file and label name
for v in "$BICUBICPP_VER" "$SVFOCUS_VER"; do
    case "$v" in ''|*[!A-Za-z0-9_]*) echo "bad version name: [$v] (letters, digits, _)" >&2; exit 1 ;; esac
done
BICUBICPP=$MODEL_DIR/BicubicPP_IR_${IN_W}x${IN_H}_x${SCALE}_${BICUBICPP_VER}.dlc
[ -f "$BICUBICPP" ] || { echo "model not found: $BICUBICPP (Bicubic++ $BICUBICPP_VER x$SCALE, profile $PROFILE; BICUBICPP_VER picks the version)" >&2; exit 1; }
if [ "$PROFILE" = "1024" ]; then
    SVF_NAME=SVFocusSRNet
    SVFOCUS=$MODEL_DIR/SVFocusSRNet_IR_${IN_W}x${IN_H}_x${SCALE}.dlc
else
    SVF_NAME=SVFocusSRNet_$SVFOCUS_VER
    SVFOCUS=$MODEL_DIR/SVFocusSRNet_IR_${IN_W}x${IN_H}_x${SCALE}_${SVFOCUS_VER}.dlc
fi

# Another pipeline holding the camera makes gst fail with a cryptic "Device busy" (or a
# silent stall), and any other gst or snpe-net-run job shares the DSP budget; name the
# culprit before the recording folder exists. The whole ps line is shown: its PID column
# differs between BusyBox and procps.
CAMERA_BUSY=$(ps -ef | grep -a -E -e "[v]4l2src device=$DEVICE" | head -1)
if [ -n "$CAMERA_BUSY" ]; then
    echo "ERROR: another gst pipeline holds $DEVICE; stop it (kill <PID>) first:" >&2
    echo "       $CAMERA_BUSY" >&2
    exit 1
fi
BUSY=$(ps -ef | grep -a -E -e "(^|[ /])[g]st-launch-1\.0( |\$)" -e "[s]npe-net-run" | head -1)
if [ -n "$BUSY" ]; then
    if [ "$FORCE" = 1 ]; then
        echo "FORCE=1: starting next to another gst pipeline or snpe-net-run (they share the DSP):" >&2
        echo "       $BUSY" >&2
    else
        echo "ERROR: another gst pipeline or snpe-net-run shares the DSP; stop it (kill <PID>) first, or run with FORCE=1:" >&2
        echo "       $BUSY" >&2
        exit 1
    fi
fi

# --- recording target
if [ "$RECORD" = 1 ]; then
    STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
    case "$STAMP" in 20[2-9][0-9]*) ;; *)
        echo "WARNING: board clock says $(date) -- folder will start with $STAMP. Sync with date -s or pass STAMP=" >&2 ;;
    esac
    OUT_DIR=$OUT_ROOT/${STAMP}_$(basename "$0" .sh)_x${SCALE}_$PROFILE
    if [ -e "$OUT_DIR" ]; then echo "ERROR: $OUT_DIR already exists; pass another STAMP=" >&2; exit 1; fi
    mkdir -p "$OUT_DIR"
    if gst-inspect-1.0 mp4mux >/dev/null 2>&1; then
        EXT=mp4; MUX="h264parse ! mp4mux"
    else
        EXT=h264; MUX="h264parse"
    fi
fi

case "$TRANSPORT" in
    rtsp) SEND_HOST=127.0.0.1; echo "IR $DEVICE 640x512   profile $PROFILE   scale x$SCALE   board $BOARD_IP -> RTSP (TCP)" ;;
    udp)  SEND_HOST=$HOST;     echo "IR $DEVICE 640x512   profile $PROFILE   scale x$SCALE   board $BOARD_IP -> $HOST (UDP)" ;;
    *)    echo "TRANSPORT must be rtsp or udp, got $TRANSPORT" >&2; exit 1 ;;
esac
echo "  NET_QUEUE=[$NET_QUEUE]  ENC_OPTS=[$ENC_OPTS]"
if [ -n "$TRIM" ]; then
    echo "  crop   <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}, trimmed to ${FINAL_W}x${FINAL_H} at <$TRIM_X,$TRIM_Y>"
else
    echo "  crop   <$CROP_X,$CROP_Y,$IN_W,$IN_H> -> ${OUT_W}x${OUT_H}"
fi

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
      video/x-raw(memory:GBM),format=RGBA,width=$IN_W,height=$IN_H ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$1 layers=<$2> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform $TRIM ! video/x-raw(memory:GBM),format=NV12,width=$FINAL_W,height=$FINAL_H ! "
}

ANCHOR_BRANCH="src. ! queue leaky=2 max-size-buffers=2 ! \
      videoconvert ! qtivtransform crop=<$CROP_X,$CROP_Y,$IN_W,$IN_H> ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform $TRIM ! video/x-raw(memory:GBM),format=NV12,width=$FINAL_W,height=$FINAL_H ! "
CAMERA_BRANCH="src. ! queue leaky=2 max-size-buffers=2 ! \
      videoconvert ! qtivtransform ! "

# tile <x> <y> <w> <h> <label> <banner text> [branch]: label the tile at <x,y> and print
# it; a branch (a source tee consumer ending in "! ") becomes the composer's next sink
# pad there at <w,h>, without one the tile stays empty.
TILE_PADS=""; LABELS=""; TILES=""; PAD=0
tile() {
    LABELS="$LABELS${LABELS:+,}(structure)\"Text,contents=\\\"$5\\\",fontsize=40,position=<$(( $1 + 24 )),$(( $2 + 24 ))>,color=0xFFFF00FF;\""
    echo "    <$1,$2>  $5  --  $6"
    [ -n "${7:-}" ] || return 0
    TILE_PADS="$TILE_PADS sink_$PAD::position=<$1,$2> sink_$PAD::dimensions=<$3,$4>"
    TILES="$TILES $7comp.sink_$PAD"
    PAD=$(( PAD + 1 ))
}

if [ "$SCALE" = 3 ]; then
    MOS_W=$FINAL_W; BPP_X=0
else
    MOS_W=$(( 2 * FINAL_W )); BPP_X=$FINAL_W
fi
MOS_H=$(( 2 * FINAL_H ))
echo "  $PORT_MOSAIC  mosaic ${MOS_W}x${MOS_H}, target-bitrate $MOSAIC_BITRATE, latency ${MOSAIC_LATENCY} ms, labels LABEL=$LABEL"
tile 0 0 "$FINAL_W" "$FINAL_H" qtivtransform "qtivtransform crop + bilinear upscale ${OUT_W}x${OUT_H}${TRIM:+ -> trim} (anchor, the no-SR zoom path)" "$ANCHOR_BRANCH"
MODEL_LINES=""
if [ "$SCALE" != 3 ]; then
    if [ "$CAMERA_TILE" = 1 ]; then
        tile "$CAM_X" 0 "$CAM_W" "$CAM_H" camera "full 640x512 frame, scaled to ${CAM_W}x${CAM_H} by the composer" "$CAMERA_BRANCH"
    else
        tile "$CAM_X" 0 "$CAM_W" "$CAM_H" camera_off "empty (CAMERA_TILE=$CAMERA_TILE)"
    fi
    if [ -f "$SVFOCUS" ]; then
        tile 0 "$FINAL_H" "$FINAL_W" "$FINAL_H" "$SVF_NAME" "$SVFOCUS  (layer $SVFOCUS_LAYER)" "$(sr_branch "$SVFOCUS" "$SVFOCUS_LAYER")"
        MODEL_LINES="<0,$FINAL_H> $SVF_NAME $(md5sum "$SVFOCUS" | cut -d' ' -f1) $SVFOCUS
"
    else
        tile 0 "$FINAL_H" "$FINAL_W" "$FINAL_H" "${SVF_NAME}_DLC_missing" "empty: $SVFOCUS not found"
    fi
fi
tile "$BPP_X" "$FINAL_H" "$FINAL_W" "$FINAL_H" "Bicubic++_$BICUBICPP_VER" "$BICUBICPP  (layer $BPP_LAYER)" "$(sr_branch "$BICUBICPP" "$BPP_LAYER")"
MODEL_LINES="$MODEL_LINES<$BPP_X,$FINAL_H> BicubicPP_$BICUBICPP_VER $(md5sum "$BICUBICPP" | cut -d' ' -f1) $BICUBICPP
"

OVERLAY=""
if [ "$LABEL" = 1 ]; then OVERLAY="qtivoverlay strings={$LABELS} !"; fi
# the composer comes before the tiles that link to its request pads
MOSAIC="qtivcomposer name=comp background=255 latency=$(( MOSAIC_LATENCY * 1000000 )) $TILE_PADS ! \
      video/x-raw(memory:GBM),format=NV12,width=$MOS_W,height=$MOS_H ! $OVERLAY \
      qtic2venc control-rate=1 $ENC_OPTS target-bitrate=$MOSAIC_BITRATE ! $(encoder_out mosaic "$PORT_MOSAIC")"

if [ "$RECORD" = 1 ]; then
    printf '# profile %s  x%s  crop <%s,%s,%s,%s> -> %sx%s  trim %s  board %s\n# tile model md5 dlc\n%s' \
        "$PROFILE" "$SCALE" "$CROP_X" "$CROP_Y" "$IN_W" "$IN_H" "$OUT_W" "$OUT_H" "${TRIM:-none}" "$BOARD_IP" "$MODEL_LINES" > "$OUT_DIR/models.txt"
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
  v4l2src device="$DEVICE" name=ircam0 ! \
  "video/x-raw,format=NV12,width=640,height=512,framerate=30/1" ! \
  queue ! videoconvert ! \
  tee name=src \
  $MOSAIC \
  $TILES
