"""Board scripts run against stub commands: the start guards and bench_dlc.sh's output folder.

The board is not reachable from CI, so gst-launch-1.0, ps, ip and snpe-net-run are small stubs on PATH.
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY = os.path.join(REPO, "examples", "deployment")
SNPE = "root 1234 1 0 10:00 ? 00:00:01 snpe-net-run --container /data/dlc/x.dlc"


def _stubs(tmp_path, scripts):
    stub = tmp_path / "stub"
    stub.mkdir()
    for name, body in scripts.items():
        (stub / name).write_text("#!/bin/bash\n" + body + "\n")
        (stub / name).chmod(0o755)
    return {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}"}


@pytest.fixture
def gst_env(tmp_path):
    env = _stubs(tmp_path, {
        "gst-launch-1.0": 'echo "GST-LAUNCH $#"',
        "ps": 'echo "UID PID PPID C STIME TTY TIME CMD"; [ -n "${STUB_PS:-}" ] && echo "$STUB_PS"; exit 0',
        "ip": 'echo "    inet 10.0.0.98/24 scope global eth0"',
        "gst-inspect-1.0": "exit 0",
    })
    dlc = tmp_path / "dlc"
    dlc.mkdir()
    for name in ("BicubicPPDenoise_EO_1280x720_v1.dlc", "BicubicPP_EO_640x360_x2_v3.dlc", "BicubicPP_IR_640x360_x2_v3.dlc"):
        (dlc / name).touch()
    env.update(MODEL_DIR=str(dlc), OUT_ROOT=str(tmp_path / "out"), STAMP="20261017_000000", TRANSPORT="udp",
               RECORD="1", STUB_PS=SNPE)
    return env


@pytest.mark.parametrize("script,args", [("gst_compare_eo.sh", ["2"]), ("gst_compare_eo4k.sh", ["2"]),
                                         ("gst_compare_ir.sh", ["2", "720"])])
def test_gst_refuses_to_start_next_to_snpe_net_run(script, args, gst_env, tmp_path):
    """A board evaluation or bench_dlc.sh shares the DSP; the scripts used to start anyway."""
    r = subprocess.run(["bash", os.path.join(DEPLOY, script), *args], env=gst_env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 1 and "snpe-net-run" in r.stderr and "GST-LAUNCH" not in r.stdout
    assert not (tmp_path / "out").exists()


def test_ir_force_starts_next_to_snpe_net_run(gst_env):
    r = subprocess.run(["bash", os.path.join(DEPLOY, "gst_compare_ir.sh"), "2", "720"],
                       env={**gst_env, "FORCE": "1", "RECORD": "0"}, capture_output=True, text=True, timeout=60)
    assert "FORCE=1" in r.stderr and "GST-LAUNCH" in r.stdout


def test_bench_ignores_out_and_cleans_its_own_folder(tmp_path):
    """OUT used to be an environment variable that the exit trap removed, whatever folder it named."""
    env = _stubs(tmp_path, {
        "snpe-net-run": 'out=""; while [ $# -gt 0 ]; do [ "$1" = "--output_dir" ] && out=$2; shift; done\n'
                        'mkdir -p "$out" && echo log > "$out/SNPEDiag_0.log"',
        "snpe-diagview": 'echo "Total Inference Time: 3456 us"',
    })
    bench, dlc, keep = tmp_path / "bench", tmp_path / "dlc", tmp_path / "keep"
    (bench / "EO_640x360").mkdir(parents=True)
    (bench / "EO_640x360" / "input_list.txt").write_text("EO_640x360/0.raw\n")
    dlc.mkdir()
    (dlc / "BicubicPP_EO_640x360_x2_v3.dlc").touch()
    keep.mkdir()
    (keep / "f").write_text("keep\n")
    env.update(BENCH=str(bench), MODEL_DIR=str(dlc), REPEATS="1", DURATION="1", OUT=str(keep))
    r = subprocess.run(["bash", os.path.join(DEPLOY, "bench_dlc.sh")], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "3.456 ms" in r.stdout
    assert (keep / "f").read_text() == "keep\n" and not (bench / "out").exists()
