#!/bin/bash
# Denoise once, then compare the upscalers available at this scale on the same
# crop of the EO camera, with a plain upscale as the anchor, as one mosaic over
# RTSP so a room sees every tile on the same camera frame.
#
#   usage: [BICUBICPP_VER=v3] [SVFOCUS_VER=v2] ./gst_compare_eo.sh [2|3|4]
#
#   port 20000 : mosaic of 1280x720 tiles, each labelled at its top-left corner
#
#   x2, x4  2560x1440                          x3  1280x1440
#   +---------------------+-----------------+  +---------------------+
#   | qtivtransform       | camera          |  | qtivtransform       |
#   | (anchor, denoised)  | (full 1920x1080,|  | (anchor, denoised)  |
#   |                     |  no denoise)    |  |                     |
#   +---------------------+-----------------+  +---------------------+
#   | SVFocusSRNet_<ver>  | Bicubic++_<ver> |  | Bicubic++_<ver>     |
#   +---------------------+-----------------+  +---------------------+
#
# Models: the denoise DLC and $MODEL_DIR/BicubicPP_EO_<W>x<H>_x<s>_$BICUBICPP_VER.dlc
# (default v3) must exist, or the script stops before anything starts; a v4 runs as
# soon as its DLC is there (BICUBICPP_VER=v4). SVFocusSRNet_EO_<W>x<H>_x<s>_$SVFOCUS_VER.dlc
# (default v2) is probed at x2/x4; without it its tile stays empty under a
# "..._DLC_missing" label. x3 shows the anchor over Bicubic++ only.
#
# Labels carry no spaces (Bicubic++_v3, not "Bicubic++ v3"): the labels proven on
# the board had none, and gst-launch ends a {...} property value at an unescaped
# space. Versions are therefore limited to letters, digits and _.
#
# The denoise net runs once at 1280x720 and its output is teed to every
# downstream branch, so all upscalers see exactly the same pixels. It is not
# optional here: the EO sensor is noisy enough that comparing upscalers on raw
# frames mostly compares how each one amplifies that noise. Only the camera tile
# shows the raw frame.
#
# Colour handling
#   RGBA starts at the denoise input and runs to the last qtivtransform before
#   the composer, so the denoise output, the model outputs and the bicubic
#   upscale all stay at full chroma resolution rather than being resampled to
#   4:2:0 at every hop.
#
#   The source tee deliberately stays NV12. Promoting it to RGBA cannot recover
#   chroma the sensor already delivered as 4:2:0, and at 1920x1080 it costs
#   8.3 MB/frame against 3.1 -- enough to exhaust the GBM pool across four
#   branches, which shows up as DRM_IOCTL_PRIME_FD_TO_HANDLE failures and
#   silently stalls every branch but the first.
#
# Geometry
#   1920x1080 -> centred 1280x720 crop (native pixels, no scaling) -> denoise
#   denoised 1280x720 -> centred SR-input crop -> the models and the bicubic
#   anchor, which upscales by the same exact factor and gets the same trim
#
#   scale   SR input   crop            model out   output trim
#   -------------------------------------------------------------
#   2       640x360    320,180         1280x720    0
#   3       428x240    426,240         1284x720    2   (1280 not divisible by 3)
#   4       320x180    480,270         1280x720    0
#
# Mosaic tiles keep the branches this script always ran; only what followed their
# last NV12 capsfilter changed, from an encoder to a qtivcomposer sink pad. The
# camera tile takes the source tee consumer the 1920x1080 preview encoder had:
# qtivtransform scales the NV12 GBM frame to NV12 1280x720 (16:9 to 16:9, no
# bars), so the tee keeps two consumers and stays NV12. Board 98 ran it at x2 and x4
# (x3 has no camera tile) on 2026-10-06; if the camera tile breaks the pipeline,
# CAMERA_TILE=0 leaves it empty under its label and runs the rest.
#
# CAMERA_TILE=crop shows, instead of the full frame, the camera pixels of the SR
# input region (x2 <640,360,640,360>, x4 <800,450,320,180>) upscaled by the same
# factor with no denoise and no model, so camera settings such as Aperture or 3D NR
# can be judged at the magnification the other tiles show. Not yet run on the board.
#
# Same instant: every tile's buffer carries its camera frame's timestamp, and
# qtivcomposer waits MOSAIC_LATENCY ms (default 33, one frame) for a late tile.
# Here the model tiles run two DSP stages (denoise, then SR) and the camera tile
# none; if the model tiles visibly trail the camera tile, raise MOSAIC_LATENCY
# (not measured).
#
# Composer, labels, encoder, RTSP relay and recording are
# the mosaic output path first run on board 98 on 2026-10-06,
# see its header: one qtivcomposer, one qtivoverlay writing the labels (LABEL=0
# leaves it out), one qtic2venc at MOSAIC_BITRATE (default 24000000), the encoder
# output teed to the RTP sender and the file writer. An empty tile shows the
# composer's background.
#
# TRANSPORT=rtsp (default): gst-rtsp-server on the mosaic port, fed over loopback
# UDP on the same port number, so the viewer pulls it over TCP and a lost packet
# is resent. The banner prints the Mac command:
#   open -n -a VLC --args --rtsp-tcp rtsp://<board>:20000
# TRANSPORT=udp: plain RTP to HOST on the same port. PORT_OFFSET shifts the port
# (PORT_OFFSET=98 -> 20098) to run both boards against one viewer.
#
# RECORD=1 also writes $OUT_ROOT/<STAMP>_gst_compare_eo_x<s>/mosaic.mp4 (.h264
# without mp4mux) and models.txt (tile, model, DLC md5 and path). STAMP defaults
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
# Other knobs: MODEL_DIR (/data/dlc), DENOISE (its DLC), DENOISE_LAYER (final_act),
# SVFOCUS_LAYER and BPP_LAYER (final_act), OUT_ROOT (/data), BOARD_IP (first
# global IPv4, for the printed URL), HOST (UDP).
#
# Run it alone: it refuses to start while any gst-launch-1.0, another qtiqmmfsrc
# pipeline or an snpe-net-run (board evaluation, bench_dlc.sh) runs, since that would
# hold the camera or share the DSP.

set -e

SCALE=${1:-2}
HOST=${HOST:-192.168.100.112}
TRANSPORT=${TRANSPORT:-rtsp}
# Every DLC -- denoise and SR, both domains -- lives flat in one folder;
# the file names already carry model, domain, resolution and scale.
MODEL_DIR=${MODEL_DIR:-/data/dlc}
DENOISE=${DENOISE:-$MODEL_DIR/BicubicPPDenoise_EO_1280x720_v1.dlc}
DENOISE_LAYER=${DENOISE_LAYER:-final_act}
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

case "$SCALE" in
    2) IN_W=640; IN_H=360 ;;
    3) IN_W=428; IN_H=240 ;;
    4) IN_W=320; IN_H=180 ;;
    *) echo "usage: $0 [2|3|4]" >&2; exit 1 ;;
esac

# 1920x1080 -> centred 1280x720 fed to the denoise net
DN_X=$(( (1920 - 1280) / 2 ))
DN_Y=$(( (1080 - 720) / 2 ))
# SR input cropped from the centre of the denoised 1280x720
CROP_X=$(( (1280 - IN_W) / 2 ))
CROP_Y=$(( (720 - IN_H) / 2 ))
OUT_W=$(( IN_W * SCALE ))
OUT_H=$(( IN_H * SCALE ))
TRIM_X=$(( (OUT_W - 1280) / 2 ))

# --- who am I: the first global IPv4 address, for the viewer URL (BOARD_IP overrides it
# on a board with more than one interface up)
BOARD_IP=${BOARD_IP:-$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)}
BOARD_IP=${BOARD_IP:-<board-ip>}
PORT_MOSAIC=$(( 20000 + PORT_OFFSET ))

# --- models: denoise and Bicubic++ are required, SVFocusSRNet is probed (x2/x4 only)
for v in "$BICUBICPP_VER" "$SVFOCUS_VER"; do
    # the version becomes part of a file and a label name
    case "$v" in ''|*[!A-Za-z0-9_]*) echo "bad version name: [$v] (letters, digits, _)" >&2; exit 1 ;; esac
done
BICUBICPP=$MODEL_DIR/BicubicPP_EO_${IN_W}x${IN_H}_x${SCALE}_${BICUBICPP_VER}.dlc
SVFOCUS=$MODEL_DIR/SVFocusSRNet_EO_${IN_W}x${IN_H}_x${SCALE}_${SVFOCUS_VER}.dlc
[ -f "$DENOISE" ] || { echo "model not found: $DENOISE (denoise; DENOISE= picks another)" >&2; exit 1; }
[ -f "$BICUBICPP" ] || { echo "model not found: $BICUBICPP (Bicubic++ $BICUBICPP_VER x$SCALE; BICUBICPP_VER picks the version)" >&2; exit 1; }

# Another pipeline holding the camera makes gst fail cryptically (or stall), and any
# other gst or snpe-net-run job shares the DSP budget; name the culprit before the recording folder
# exists. The whole ps line is shown: its PID column differs between BusyBox and procps.
BUSY=$(ps -ef | grep -a -E -e "[q]tiqmmfsrc" -e "(^|[ /])[g]st-launch-1\.0( |\$)" -e "[s]npe-net-run" | head -1)
if [ -n "$BUSY" ]; then
    echo "ERROR: another gst pipeline or snpe-net-run holds the camera or shares the DSP; stop it (kill <PID>) first:" >&2
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
    rtsp) SEND_HOST=127.0.0.1; echo "EO camera=0 1920x1080   scale x$SCALE   board $BOARD_IP -> RTSP (TCP)" ;;
    udp)  SEND_HOST=$HOST;     echo "EO camera=0 1920x1080   scale x$SCALE   board $BOARD_IP -> $HOST (UDP)" ;;
    *)    echo "TRANSPORT must be rtsp or udp, got $TRANSPORT" >&2; exit 1 ;;
esac
echo "  NET_QUEUE=[$NET_QUEUE]  ENC_OPTS=[$ENC_OPTS]"
echo "  denoise  $DENOISE  (layer $DENOISE_LAYER)  crop <$DN_X,$DN_Y,1280,720>"
echo "  SR crop  <$CROP_X,$CROP_Y,$IN_W,$IN_H> of the denoised frame -> ${OUT_W}x${OUT_H}, trimmed to 1280x720 at x=$TRIM_X"

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

# tile <pad> <x> <y> <label> <banner text>: label the tile at <x,y> and print it, and
# place composer pad sink_<pad> there; an empty <pad> leaves the tile empty. Pads are
# numbered without gaps in the order their branches link below: anchor 0, Bicubic++ 1,
# then camera and SVFocusSRNet when they run (the reference only ever ran gap-free pads).
TILE_PADS=""; LABELS=""
tile() {
    LABELS="$LABELS${LABELS:+,}(structure)\"Text,contents=\\\"$4\\\",fontsize=40,position=<$(( $2 + 24 )),$(( $3 + 24 ))>,color=0xFFFF00FF;\""
    echo "    <$2,$3>  $4  --  $5"
    [ -n "$1" ] || return 0
    TILE_PADS="$TILE_PADS sink_$1::position=<$2,$3> sink_$1::dimensions=<1280,720>"
}

if [ "$SCALE" = 3 ]; then
    MOS_W=1280; BPP_X=0
else
    MOS_W=2560; BPP_X=1280
fi
MOS_H=1440
echo "  $PORT_MOSAIC  mosaic ${MOS_W}x${MOS_H}, target-bitrate $MOSAIC_BITRATE, latency ${MOSAIC_LATENCY} ms, labels LABEL=$LABEL"
tile 0 0 0 qtivtransform "denoised crop -> qtivtransform ${OUT_W}x${OUT_H} -> trim (anchor)"
MODEL_LINES="denoise $(basename "$DENOISE" .dlc) $(md5sum "$DENOISE" | cut -d' ' -f1) $DENOISE
"
CAM_TILE=""; SVF_TILE=""; SVF_PAD=2
if [ "$SCALE" != 3 ]; then
    case "$CAMERA_TILE" in
        1|crop)
            CAM_CROP=""
            if [ "$CAMERA_TILE" = crop ]; then
                CAM_CROP="crop=<$(( DN_X + CROP_X )),$(( DN_Y + CROP_Y )),$IN_W,$IN_H> "
                tile 2 1280 0 camera_crop "camera at the SR input region, no denoise, qtivtransform x$SCALE"
            else
                tile 2 1280 0 camera "full 1920x1080 frame, no denoise, qtivtransform to 1280x720"
            fi
            CAM_TILE="src. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform $CAM_CROP! video/x-raw(memory:GBM),format=NV12,width=1280,height=720 ! comp.sink_2"
            SVF_PAD=3 ;;
        *)
            tile "" 1280 0 camera_off "empty (CAMERA_TILE=$CAMERA_TILE)" ;;
    esac
    if [ -f "$SVFOCUS" ]; then
        tile "$SVF_PAD" 0 720 "SVFocusSRNet_$SVFOCUS_VER" "$SVFOCUS  (layer $SVFOCUS_LAYER)"
        MODEL_LINES="$MODEL_LINES<0,720> SVFocusSRNet_$SVFOCUS_VER $(md5sum "$SVFOCUS" | cut -d' ' -f1) $SVFOCUS
"
        SVF_TILE="dn. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop=<$CROP_X,$CROP_Y,$IN_W,$IN_H> ! \
      video/x-raw(memory:GBM),format=RGBA,width=$IN_W,height=$IN_H ! \
      qtimlvconverter ! neural-network/tensors,type=FLOAT32 ! \
      qtimlsnpe delegate=dsp model=$SVFOCUS layers=<$SVFOCUS_LAYER> ! \
      neural-network/tensors,type=FLOAT32 ! \
      qtimlvsuperresolution module=srnet-float ! \
      video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H ! \
      qtivtransform crop=<$TRIM_X,0,1280,720> ! \
      video/x-raw(memory:GBM),format=NV12,width=1280,height=720 ! \
      comp.sink_$SVF_PAD"
    else
        tile "" 0 720 "SVFocusSRNet_${SVFOCUS_VER}_DLC_missing" "empty: $SVFOCUS not found"
    fi
fi
tile 1 "$BPP_X" 720 "Bicubic++_$BICUBICPP_VER" "$BICUBICPP  (layer $BPP_LAYER)"
MODEL_LINES="$MODEL_LINES<$BPP_X,720> BicubicPP_$BICUBICPP_VER $(md5sum "$BICUBICPP" | cut -d' ' -f1) $BICUBICPP
"

OVERLAY=""
if [ "$LABEL" = 1 ]; then OVERLAY="qtivoverlay strings={$LABELS} !"; fi
# the composer comes before the tiles that link to its request pads
MOSAIC="qtivcomposer name=comp background=255 latency=$(( MOSAIC_LATENCY * 1000000 )) $TILE_PADS ! \
      video/x-raw(memory:GBM),format=NV12,width=$MOS_W,height=$MOS_H ! $OVERLAY \
      qtic2venc control-rate=1 $ENC_OPTS target-bitrate=$MOSAIC_BITRATE ! $(encoder_out mosaic "$PORT_MOSAIC")"

if [ "$RECORD" = 1 ]; then
    printf '# x%s  denoise crop <%s,%s,1280,720>  SR crop <%s,%s,%s,%s> -> %sx%s trim x=%s  board %s\n# tile model md5 dlc\n%s' \
        "$SCALE" "$DN_X" "$DN_Y" "$CROP_X" "$CROP_Y" "$IN_W" "$IN_H" "$OUT_W" "$OUT_H" "$TRIM_X" "$BOARD_IP" "$MODEL_LINES" > "$OUT_DIR/models.txt"
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
  src. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop="<$DN_X,$DN_Y,1280,720>" ! \
      "video/x-raw(memory:GBM),format=RGBA,width=1280,height=720" ! \
      qtimlvconverter ! "neural-network/tensors,type=FLOAT32" ! \
      qtimlsnpe delegate=dsp model=$DENOISE layers="<$DENOISE_LAYER>" ! \
      "neural-network/tensors,type=FLOAT32" ! \
      qtimlvsuperresolution module=srnet-float ! \
      "video/x-raw(memory:GBM),format=RGBA,width=1280,height=720" ! \
      tee name=dn \
  dn. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop="<$CROP_X,$CROP_Y,$IN_W,$IN_H>" ! \
      "video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H" ! \
      qtivtransform crop="<$TRIM_X,0,1280,720>" ! \
      "video/x-raw(memory:GBM),format=NV12,width=1280,height=720" ! \
      comp.sink_0 \
  dn. ! queue leaky=2 max-size-buffers=2 ! \
      qtivtransform crop="<$CROP_X,$CROP_Y,$IN_W,$IN_H>" ! \
      "video/x-raw(memory:GBM),format=RGBA,width=$IN_W,height=$IN_H" ! \
      qtimlvconverter ! "neural-network/tensors,type=FLOAT32" ! \
      qtimlsnpe delegate=dsp model=$BICUBICPP layers="<$BPP_LAYER>" ! \
      "neural-network/tensors,type=FLOAT32" ! \
      qtimlvsuperresolution module=srnet-float ! \
      "video/x-raw(memory:GBM),format=RGBA,width=$OUT_W,height=$OUT_H" ! \
      qtivtransform crop="<$TRIM_X,0,1280,720>" ! \
      "video/x-raw(memory:GBM),format=NV12,width=1280,height=720" ! \
      comp.sink_1 \
  $CAM_TILE \
  $SVF_TILE
