"""Environment checks that pass only inside the usdp image (docs/modules/20261019-MOD-DockerEnv.md §4.2).

They fail, never skip, in any other environment, so a wrong image cannot pass the suite silently.
Run them through docker/run.sh without --gpu.
"""

import base64
import hashlib
import importlib
import importlib.metadata as md
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from packaging.version import Version

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIP_TOOLS = {"pip": "26.0.1", "setuptools": "82.0.0", "wheel": "0.46.3"}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins():
    pins = {}
    with open(os.path.join(REPO, "requirements.txt")) as f:
        for line in f:
            name, _, version = line.strip().partition("==")
            pins[_norm(name)] = version
    return pins


def test_TC_DOCK_01_1_DistributionsMatchRequirements():
    installed = {}
    for dist in md.distributions():
        name = _norm(dist.metadata["Name"])
        if name not in PIP_TOOLS:
            installed.setdefault(name, dist.version)
    pins = _pins()
    assert len(pins) == 178
    # Compare as PEP 440 versions, like pip: thop's wheel metadata says 0.1.1-2209072238, pip freeze 0.1.1.post2209072238.
    assert {k: v for k, v in installed.items() if k not in pins or Version(v) != Version(pins[k])} == {}
    assert sorted(set(pins) - set(installed)) == []


def test_TC_DOCK_01_2_PinnedToolsVersionsAndImports():
    assert {name: md.version(name) for name in PIP_TOOLS} == PIP_TOOLS
    assert (torch.__version__, np.__version__, cv2.__version__) == ("2.9.0+cu128", "2.2.6", "4.12.0")
    for module in ("aimet_torch", "aimet_common.defs", "aimet_torch.common.defs", "pyiqa", "torch.utils.tensorboard"):
        importlib.import_module(module)


def test_TC_DOCK_01_3_HeadlessCv2Wins():
    record = {str(f): f.hash.value for f in md.distribution("opencv-python-headless").files if f.hash}
    with open(os.path.join(os.path.dirname(cv2.__file__), "cv2.abi3.so"), "rb") as f:
        digest = base64.urlsafe_b64encode(hashlib.sha256(f.read()).digest()).rstrip(b"=").decode()
    assert digest == record["cv2/cv2.abi3.so"]


def test_TC_DOCK_01_4_PipCheckClean():
    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_TC_DOCK_01_5_KoreanOutputTimezoneAndEnv():
    r = subprocess.run([sys.executable, "-c", "print('한글 출력')"], capture_output=True, timeout=60)
    assert r.returncode == 0 and "한글 출력" in r.stdout.decode("utf-8")
    assert "KST" in time.tzname
    names = ("NUMPY_MADVISE_HUGEPAGE", "PYTHONUNBUFFERED", "NVIDIA_VISIBLE_DEVICES", "LANG", "TZ")
    assert {k: os.environ.get(k) for k in names} == {
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "PYTHONUNBUFFERED": "1",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "LANG": "C.UTF-8",
        "TZ": "Asia/Seoul",
    }
    assert os.stat(os.path.expanduser("~/.cache")).st_uid == os.getuid()


FFMPEG = "/usr/local/bin/ffmpeg"


def test_TC_DOCK_01_6_AptPackagesMatchRecord():
    with open(os.path.join(REPO, "docker", "apt-versions.txt")) as f:
        recorded = f.read().split()
    out = subprocess.run(["dpkg-query", "-W", "-f=${Package}=${Version}\n"], capture_output=True, text=True, timeout=60)
    assert sorted(out.stdout.split()) == sorted(recorded)


def test_TC_DOCK_03_1_FfmpegBinaryMatches():
    with open(FFMPEG, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == "d5c698e0e98e5cf7de03fe14bd1cda6a"
    r = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "7.0.2" in r.stdout.splitlines()[0]


def test_TC_DOCK_03_2_FfmpegEncodesX264():
    assert os.access(FFMPEG, os.X_OK)
    cmd = [FFMPEG, "-hide_banner", "-f", "lavfi", "-i", "testsrc=d=0.1", "-c:v", "libx264", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


QAIRT = "/opt/qairt"
QAIRT_TOOLS = ("qairt-converter", "qairt-quantizer", "snpe-dlc-graph-prepare", "snpe-dlc-info", "qairt-dlc-diff")


def _qairt_elves():
    """ELF files on the conversion path: x86_64 tools and libraries plus the converter's Python extensions."""
    found = []
    for top in ("bin/x86_64-linux-clang", "lib/x86_64-linux-clang", "lib/python/qti/aisw"):
        for dirpath, _, files in os.walk(os.path.join(QAIRT, top)):
            for name in sorted(files):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as f:
                    if f.read(4) == b"\x7fELF":
                        found.append(path)
    return found


def test_TC_DOCK_02_1_QairtToolsInsideEnv():
    r = subprocess.run(["qairt-env", "bash", "-c", "command -v " + " ".join(QAIRT_TOOLS)], capture_output=True,
                       text=True, timeout=60)
    paths = r.stdout.split()
    assert r.returncode == 0 and len(paths) == 5 and all(p.startswith(QAIRT + "/") for p in paths), r.stdout + r.stderr
    r = subprocess.run(["qairt-env", "python3", "-c", "import numpy; print(numpy.__version__)"], capture_output=True,
                       text=True, timeout=60)
    assert r.stdout.strip() == "1.26.4", r.stderr
    r = subprocess.run(["qairt-env", "pip", "freeze", "--all"], capture_output=True, text=True, timeout=60)
    with open(os.path.join(REPO, "docker", "qairt-requirements.txt")) as f:
        assert sorted(r.stdout.lower().split()) == sorted(f.read().lower().split()), r.stdout
    r = subprocess.run(["qairt-env", "qairt-converter", "--help"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_TC_DOCK_02_2_QairtDoesNotLeak():
    assert [t for t in QAIRT_TOOLS if shutil.which(t)] == []
    r = subprocess.run(["bash", "-lc", "command -v qairt-converter"], capture_output=True, text=True, timeout=60)
    assert r.stdout == ""
    assert subprocess.run([sys.executable, "-c", "import qti"], capture_output=True, timeout=60).returncode != 0
    assert [k for k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH") if QAIRT in os.environ.get(k, "")] == []
    assert QAIRT not in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=60).stdout
    assert np.__version__ == "2.2.6"


def test_TC_DOCK_02_3_NoMissingSharedLibraries():
    elves = _qairt_elves()
    assert len(elves) >= 30
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.join(QAIRT, "lib", "x86_64-linux-clang")}
    missing = []
    for path in elves:
        out = subprocess.run(["ldd", path], capture_output=True, text=True, env=env, timeout=60).stdout
        missing += [f"{path}: {line.strip()}" for line in out.splitlines() if "not found" in line]
    assert missing == []


def test_TC_DOCK_02_4_QairtEnvWithoutCommand():
    r = subprocess.run(["qairt-env"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "usage" in r.stderr


# ---- MREQ-DOCK-15: docker/accept.sh reports every selected case even when the jobs it starts fail ----

ACCEPT_CASES = ["TC_DOCK_09_1", "TC_DOCK_09_2", "TC_DOCK_10_3", "TC_DOCK_11_1", "TC_DOCK_13_2", "TC_DRUN_03_3"]


def _accept_with_refusing_jobs(tmp_path, gpu, cases=ACCEPT_CASES, out=None, run_out=""):
    """Run accept.sh in a fake repo whose run.sh refuses every job (printing run_out first) and whose nvidia-smi fails."""
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "docker", "accept.sh"), repo / "docker" / "accept.sh")
    stub = tmp_path / "stub"
    stub.mkdir()
    for path, body in ((repo / "docker" / "run.sh", 'printf "%s" "$STUB_RUN_OUT"; echo "run.sh: refused: stub" >&2; exit 1'),
                       (stub / "nvidia-smi", 'echo "Failed to initialize NVML" >&2; exit 9')):
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    out = out or repo / "results" / "accept"
    args = ["bash", str(repo / "docker" / "accept.sh"), "--only", ",".join(cases), "--out", str(out)]
    if gpu:
        args[2:2] = ["--gpu", gpu]
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "STUB_RUN_OUT": run_out}
    return subprocess.run(args, capture_output=True, text=True, env=env, timeout=120)


def _case_lines(stdout):
    return [line for line in stdout.splitlines() if line.startswith(("PASS ", "FAIL ", "NOT RUN "))]


def test_TC_DOCK_15_2_AcceptReportsEveryCaseWhenJobsFail(tmp_path):
    r = _accept_with_refusing_jobs(tmp_path, "0")
    lines = _case_lines(r.stdout)
    assert [line.split()[1].rstrip(":") for line in lines] == [c.replace("_", "-") for c in ACCEPT_CASES], r.stdout
    assert all(line.startswith("FAIL ") for line in lines), r.stdout
    assert r.stdout.splitlines()[-1].startswith("accept.sh: 0 passed, 6 not passed"), r.stdout + r.stderr
    assert r.returncode == 1 and "aborted" not in r.stderr


def test_TC_DOCK_15_3_AcceptWithoutGpuMarksGpuCasesNotRun(tmp_path):
    r = _accept_with_refusing_jobs(tmp_path, None)
    lines = _case_lines(r.stdout)
    assert [line.split()[0] for line in lines] == ["FAIL"] * 4 + ["NOT", "NOT"], r.stdout + r.stderr
    assert r.stdout.splitlines()[-1].startswith("accept.sh: 0 passed, 6 not passed"), r.stdout + r.stderr
    assert r.returncode == 1


def test_TC_DOCK_15_4_AcceptRefusesOutOutsideMounts(tmp_path):
    out = tmp_path / "elsewhere"
    r = _accept_with_refusing_jobs(tmp_path, None, out=out)
    assert r.returncode == 2 and "must be under" in r.stderr and not out.exists(), r.stdout + r.stderr


def test_TC_DOCK_15_5_PytestFailReasonQuotesSummary(tmp_path):
    log = "3 failed, 10 passed in 1.00s\nsys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute\n"
    r = _accept_with_refusing_jobs(tmp_path, None, cases=["TC_DOCK_09_1"], run_out=log)
    lines = _case_lines(r.stdout)
    assert len(lines) == 1 and lines[0].startswith("FAIL TC-DOCK-09-1") and "3 failed, 10 passed" in lines[0], r.stdout



def test_TC_DOCK_15_6_AcceptAbortReportsSignal(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "docker", "accept.sh"), repo / "docker" / "accept.sh")
    (repo / "docker" / "run.sh").write_text("#!/bin/bash\nsleep 30\n")
    (repo / "docker" / "run.sh").chmod(0o755)
    args = ["bash", str(repo / "docker" / "accept.sh"), "--only", "TC_DOCK_09_1", "--out", str(repo / "results" / "a")]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    time.sleep(1)
    os.killpg(p.pid, signal.SIGTERM)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 143 and "aborted before the summary (exit 143)" in err, err


def test_TC_DOCK_15_7_AcceptThroughSymlinkedRepoPath(tmp_path):
    r = _accept_with_refusing_jobs(tmp_path, None, cases=["TC_DOCK_09_1"])
    link = tmp_path / "repo-link"
    link.symlink_to(tmp_path / "repo")
    r = subprocess.run(["bash", str(link / "docker" / "accept.sh"), "--only", "TC_DOCK_09_1", "--out", str(link / "results" / "b")],
                       capture_output=True, text=True, timeout=120, env={**os.environ, "STUB_RUN_OUT": ""})
    assert r.returncode == 1 and _case_lines(r.stdout)[0].startswith("FAIL TC-DOCK-09-1"), r.stdout + r.stderr
