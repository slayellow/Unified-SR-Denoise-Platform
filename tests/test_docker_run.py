"""docker/run.sh and docker/entry.sh against stub commands (docs/modules/20261019-MOD-DockerRun.md).

docker, nvidia-smi, findmnt, id and ls are stubs on PATH, and a fake torch package on PYTHONPATH stands in for
CUDA, so the start guards, the docker argv and entry.sh's real CUDA check run without a daemon or a GPU.
Every test copies the two scripts into a fake repo under tmp_path, so default logs never land in the real repo.
"""

import itertools
import os
import shutil
import signal
import subprocess
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(REPO, "docker", "entry.sh")
IMAGE_HOME = "/home/jshong"
# Variables a caller (for example a `run.sh --name` job running this suite) may carry into the tests.
CALLER_VARS = ("USDP_IMAGE", "USDP_LOG", "USDP_GPU_CHECK", "TINI_KILL_PROCESS_GROUP", "PYTHONPATH")

STUBS = {
    "docker": r"""
case "$1 $2" in
  "image inspect")
    case "${STUB_IMAGE:-present}" in
      missing) echo "Error: No such image: ${@: -1}" >&2; exit 1 ;;
      daemon) echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1 ;;
      warn) echo "WARNING: Error loading config file: /home/jshong/.docker/config.json: permission denied" >&2 ;;
    esac
    echo "${STUB_LABELS:-1017 1017 /home/jshong}"; exit 0 ;;
esac
case "$1" in
  ps)
    FILTER=${@: -1}
    if [ "$2" = "-a" ]; then
      if [ -n "${STUB_NAME_USED:-}" ] && [ "$FILTER" = "name=^usdp-${STUB_NAME_USED}\$" ]; then echo abc123def456; fi
      exit 0
    fi
    if [ -n "${STUB_PS_FAIL:-}" ]; then echo "Error response from daemon: stub" >&2; exit 1; fi
    for G in ${STUB_USDP_GPUS:-}; do
      if [ "$FILTER" = "label=usdp.gpu=$G" ]; then echo "cafe0000000$G"; fi
    done
    exit 0 ;;
  run) printf '%s\0' "$@" > "$STUB_ARGV"; exit "${STUB_RUN_RC:-0}" ;;
esac
echo "docker stub: unexpected: $*" >&2; exit 99""",
    "nvidia-smi": r"""
case "${STUB_SMI:-ok}" in
  fail) echo "Failed to initialize NVML: Unknown Error" >&2; exit 9 ;;
  empty) exit 0 ;;
  garbage) echo "No devices were found"; exit 0 ;;
esac
case "$*" in
  *--query-gpu=index,uuid*)
    [ "${STUB_SMI:-ok}" = hang ] && sleep 5
    for i in 0 1 2 3 4; do echo "$i, GPU-0000000$i"; done ;;
  *--query-compute-apps=gpu_uuid*)
    case "${STUB_SMI:-ok}" in
      apps_fail) echo "Failed to initialize NVML: Unknown Error" >&2; exit 9 ;;
      apps_hang) sleep 5 ;;
    esac
    for i in ${STUB_BUSY:-}; do echo "GPU-0000000$i"; done ;;
  *) echo "nvidia-smi stub: unexpected: $*" >&2; exit 99 ;;
esac""",
    "findmnt": r"""
case " $* " in
  *" -C "*" -M $STUB_MOUNT_PATH "*) ;;
  *) exit 1 ;;
esac
if [ -n "${STUB_FSTYPE:-}" ]; then echo "$STUB_FSTYPE"; exit 0; fi
exit 1""",
    "id": r"""
case "$1" in
  -u) echo "${STUB_UID:-1017}" ;;
  -g) echo "${STUB_GID:-1017}" ;;
  *) exec /usr/bin/id "$@" ;;
esac""",
    # A hung sshfs leaves the probe in an uninterruptible wait: it ignores signals and outlasts any timeout.
    "ls": r"""
case "${STUB_LS:-real}" in
  hang) trap '' TERM INT; sleep 30 ;;
  eacces) echo "ls: cannot open directory '${@: -1}': Permission denied" >&2; exit 2 ;;
  *) exec /bin/ls "$@" ;;
esac""",
}

# entry.sh's CUDA check imports this instead of torch. hang sleeps first and then reports one GPU,
# so only the timeout can stop the job.
FAKE_TORCH = """
import os
import time

MODE = os.environ.get("STUB_TORCH", "one")
if MODE == "importerror":
    raise ImportError("No module named 'torch'")
if MODE == "hang":
    time.sleep(5)


class cuda:
    @staticmethod
    def is_available():
        return MODE in ("one", "two", "hang")

    @staticmethod
    def device_count():
        return {"one": 1, "two": 2, "hang": 1}.get(MODE, 0)
"""

# /mnt/data_server state -> (findmnt FSTYPE, ls behaviour, folder contents or None for a missing folder)
MOUNT_STATES = {
    "sshfs": ("fuse.sshfs", "real", ["x"]),
    "empty": ("", "real", []),
    "missing": ("", "real", None),
    "files": ("", "real", ["Projects"]),
    "eacces": ("fuse.sshfs", "eacces", ["x"]),
    "stale": ("fuse.sshfs", "hang", ["x"]),
}


def _env(tmp_path, mount="sshfs", **stub_vars):
    """Fake repo with the two scripts, stub commands on PATH, fake torch, HOME and data mount under tmp_path."""
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True, exist_ok=True)
    for name in ("run.sh", "entry.sh"):
        shutil.copy(os.path.join(REPO, "docker", name), repo / "docker" / name)
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        (stub / name).write_text("#!/bin/bash\n" + body.lstrip("\n") + "\n")
        (stub / name).chmod(0o755)
    (tmp_path / "fake_torch" / "torch").mkdir(parents=True, exist_ok=True)
    (tmp_path / "fake_torch" / "torch" / "__init__.py").write_text(FAKE_TORCH)
    (tmp_path / "home").mkdir(exist_ok=True)
    fstype, ls_mode, contents = MOUNT_STATES[mount]
    data = tmp_path / "data"
    if contents is not None:
        data.mkdir(exist_ok=True)
        for entry in contents:
            (data / entry).write_text("x\n")
    env = {k: v for k, v in os.environ.items() if k not in CALLER_VARS}
    env.update(
        {
            "PATH": f"{stub}:{os.environ['PATH']}",
            "PYTHONPATH": str(tmp_path / "fake_torch"),
            "HOME": str(tmp_path / "home"),
            "USDP_DATA_MOUNT": str(data),
            "USDP_PROBE_TIMEOUT": "0.5",
            "USDP_CUDA_TIMEOUT": "0.5",
            "STUB_ARGV": str(tmp_path / "argv.bin"),
            "STUB_FSTYPE": fstype,
            "STUB_LS": ls_mode,
            "STUB_MOUNT_PATH": str(data),
            "USDP_TEST_REPO": str(repo),
        }
    )
    env.update(stub_vars)
    return env


def _argv(env):
    if not os.path.exists(env["STUB_ARGV"]):
        return None
    with open(env["STUB_ARGV"], "rb") as f:
        return [a.decode() for a in f.read().split(b"\0")[:-1]]


def _run(args, env, script=None):
    """Run the fake repo's run.sh (or `script`); return (rc, stdout, stderr, docker run argv or None)."""
    r = subprocess.run(
        ["bash", script or os.path.join(env["USDP_TEST_REPO"], "docker", "run.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    return r.returncode, r.stdout, r.stderr, _argv(env)


def _mounts(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]


def test_TC_DRUN_01_1_RunsAsCallerUid(tmp_path):
    env = _env(tmp_path, STUB_UID="1234", STUB_GID="1235", STUB_LABELS="1234 1235 /home/jshong")
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 0, err
    assert argv[argv.index("--user") + 1] == "1234:1235"


def test_TC_DRUN_01_2_RefusesImageWithoutUidLabel(tmp_path):
    env = _env(tmp_path, STUB_LABELS="  ")
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 1 and argv is None
    assert "run.sh: refused:" in err and "usdp.uid" in err


def test_TC_DRUN_01_4_RefusesGidOnlyMismatch(tmp_path):
    env = _env(tmp_path, STUB_LABELS="1017 1018 /home/jshong")
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 1 and argv is None
    assert "1017:1018" in err and "1017:1017" in err


def test_TC_DRUN_01_5_RootCallerToldToUseOwnUser(tmp_path):
    env = _env(tmp_path, STUB_UID="0", STUB_GID="0")
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 1 and argv is None
    assert "not root" in err and "rebuild" not in err


def test_TC_DRUN_01_6_InspectWarningOnStderrIgnored(tmp_path):
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path, STUB_IMAGE="warn"))
    assert rc == 0 and argv is not None, err


def test_TC_DRUN_02_1_MountsExactlyThreeBinds(tmp_path):
    env = _env(tmp_path)
    repo, home = env["USDP_TEST_REPO"], env["HOME"]
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 0, err
    assert _mounts(argv) == [
        f"type=bind,source={repo},target={repo}",
        "type=bind,source=/mnt,target=/mnt,bind-propagation=rslave",
        f"type=bind,source={home}/.cache/torch,target={IMAGE_HOME}/.cache/torch",
    ]
    assert "-v" not in argv and "--volume" not in argv
    assert argv[argv.index("-w") + 1] == repo
    assert argv[:3] == ["run", "--rm", "--init"] and argv[argv.index("--ipc") + 1] == "host"
    assert argv[argv.index("usdp:latest") + 1] == f"{repo}/docker/entry.sh"


def test_TC_DRUN_02_2_ExtraMountAndKoreanSpacePaths(tmp_path):
    env = _env(tmp_path)
    extra = tmp_path / "a b" / "한글"
    ref = tmp_path / "참고 자료"
    extra.mkdir(parents=True)
    ref.mkdir()
    log = extra / "로그 파일.log"
    rc, _, err, argv = _run(
        ["--name", "k", "--log", str(log), "--mount", str(extra), "--mount", f"{ref}:/x:ro", "--", "true"], env
    )
    assert rc == 0, err
    assert _mounts(argv)[3:] == [
        f"type=bind,source={extra},target={extra}",
        f"type=bind,source={ref},target=/x,readonly",
    ]
    assert f"USDP_LOG={log}" in argv and log.is_file()


def test_TC_DRUN_02_3_CreatesMissingTorchCache(tmp_path):
    env = _env(tmp_path)
    home = tmp_path / "fresh home"
    home.mkdir()
    env["HOME"] = str(home)
    rc, _, err, argv = _run(["--", "true"], env)
    assert rc == 0, err
    cache = home / ".cache" / "torch"
    assert cache.is_dir() and cache.stat().st_uid == os.getuid()
    assert f"type=bind,source={cache},target={IMAGE_HOME}/.cache/torch" in _mounts(argv)


def test_TC_DRUN_02_4_CommandArgvPassedVerbatim(tmp_path):
    cmd = ["python3", "-c", "import sys; print(sys.argv[1:])", "a b", '"q"', "--force", "한글"]
    env = _env(tmp_path)
    rc, _, err, argv = _run(["--", *cmd], env)
    assert rc == 0, err
    assert argv[-len(cmd) :] == cmd
    r = subprocess.run(["bash", ENTRY, *cmd], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == str(cmd[3:]), r.stderr


def test_TC_DRUN_02_5_OfflineAddsNetworkNone(tmp_path):
    rc, _, err, argv = _run(["--offline", "--", "true"], _env(tmp_path))
    assert rc == 0, err
    assert argv[argv.index("--network") + 1] == "none" and argv.count("--network") == 1
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path))
    assert rc == 0, err
    assert "--network" not in argv


@pytest.mark.parametrize("stdin,stdout,flags", [("file", "pipe", ["-i"]), ("tty", "tty", ["-it"]), ("tty", "pipe", [])])
def test_TC_DRUN_02_6_ForegroundStdinAttached(tmp_path, stdin, stdout, flags):
    env = _env(tmp_path)
    master, slave = os.openpty()
    src = tmp_path / "snippet.py"
    src.write_text("print(1)\n")
    with open(src) as snippet:
        r = subprocess.run(
            ["bash", os.path.join(env["USDP_TEST_REPO"], "docker", "run.sh"), "--", "python3", "-"],
            env=env,
            stdin=slave if stdin == "tty" else snippet,
            stdout=slave if stdout == "tty" else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    os.close(master)
    os.close(slave)
    assert r.returncode == 0, r.stderr
    assert [a for a in _argv(env) if a in ("-i", "-t", "-it")] == flags


def test_TC_DRUN_02_7_SymlinkedRepoPathUsesRealPath(tmp_path):
    env = _env(tmp_path)
    repo = os.path.realpath(env["USDP_TEST_REPO"])
    link = tmp_path / "repo-link"
    link.symlink_to(repo)
    rc, _, err, argv = _run(["--name", "s", "--", "true"], env, script=str(link / "docker" / "run.sh"))
    assert rc == 0, err
    assert _mounts(argv)[0] == f"type=bind,source={repo},target={repo}" and argv[argv.index("-w") + 1] == repo
    assert os.path.isfile(os.path.join(repo, "logs", "s.log"))


def test_TC_DRUN_02_8_SymlinkMountKeepsTypedPath(tmp_path):
    env = _env(tmp_path)
    real = tmp_path / "real data"
    real.mkdir()
    link = tmp_path / "data-link"
    link.symlink_to(real)
    rc, _, err, argv = _run(["--mount", str(link), "--", "true"], env)
    assert rc == 0, err
    assert _mounts(argv)[3] == f"type=bind,source={real},target={link}"


def test_TC_DRUN_02_11_LogUnderSymlinkMountUsesTypedPath(tmp_path):
    env = _env(tmp_path)
    real = tmp_path / "real logs"
    real.mkdir()
    link = tmp_path / "log-link"
    link.symlink_to(real)
    rc, out, err, argv = _run(["--name", "s", "--mount", str(link), "--log", str(link / "s.log"), "--", "true"], env)
    assert rc == 0, err
    assert f"USDP_LOG={link / 's.log'}" in argv and (real / "s.log").is_file()


def test_TC_DRUN_02_9_UsdpImageOverride(tmp_path):
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path, USDP_IMAGE="usdp:20261019"))
    assert rc == 0, err
    assert "usdp:20261019" in argv and "usdp:latest" not in argv


def test_TC_DRUN_02_10_ForegroundExitCodePassedThrough(tmp_path):
    rc, _, err, argv = _run(["--", "false"], _env(tmp_path, STUB_RUN_RC="7"))
    assert rc == 7 and argv is not None, err


def test_TC_DRUN_03_1_SingleGpuQuotedDeviceRequest(tmp_path):
    rc, _, err, argv = _run(["--gpu", "2", "--", "true"], _env(tmp_path))
    assert rc == 0, err
    assert argv[argv.index("--gpus") + 1] == '"device=2"' and argv.count("--gpus") == 1
    assert "USDP_GPU_CHECK=1" in argv and argv[argv.index("--label") + 1] == "usdp.gpu=2"


def test_TC_DRUN_03_2_NoGpuPrintsNotice(tmp_path):
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path))
    assert rc == 0, err
    assert "--gpus" not in argv and "USDP_GPU_CHECK=1" not in argv and "--label" not in argv
    assert "GPU: none" in err


def test_TC_DRUN_04_1_DetachedArgvAndLogCreated(tmp_path):
    env = _env(tmp_path)
    log = os.path.join(env["USDP_TEST_REPO"], "logs", "t.log")
    rc, _, err, argv = _run(["--name", "t", "--", "true"], env)
    assert rc == 0, err
    assert "-d" in argv and argv[argv.index("--name") + 1] == "usdp-t"
    assert "TINI_KILL_PROCESS_GROUP=1" in argv and f"USDP_LOG={log}" in argv
    assert os.path.isfile(log)


@pytest.mark.parametrize("code", [0, 3])
def test_TC_DRUN_04_2_EntryAppendsExitCode(tmp_path, code):
    log = tmp_path / "job.log"
    log.write_text("")
    env = _env(tmp_path, USDP_LOG=str(log))
    r = subprocess.run(
        ["bash", ENTRY, "bash", "-c", f"echo hi; exit {code}"], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == code
    assert log.read_text().splitlines() == ["hi", f"[run.sh] exit {code}"]


def test_TC_DRUN_04_3_EntryRecordsSigterm143(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("")
    p = subprocess.Popen(["bash", ENTRY, "sleep", "30"], env=_env(tmp_path, USDP_LOG=str(log)), start_new_session=True)
    time.sleep(0.5)
    os.killpg(p.pid, signal.SIGTERM)
    assert p.wait(timeout=10) == 143
    assert log.read_text().splitlines()[-1] == "[run.sh] exit 143"
    with pytest.raises(ProcessLookupError):
        os.killpg(p.pid, 0)


def test_TC_DRUN_04_5_ForegroundSignalsWholeGroup(tmp_path):
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path))
    assert rc == 0, err
    assert "TINI_KILL_PROCESS_GROUP=1" in argv


@pytest.mark.parametrize("sig,code", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_TC_DRUN_04_6_StopDuringCudaCheckEndsJob(tmp_path, sig, code):
    log = tmp_path / "job.log"
    log.write_text("")
    ran = tmp_path / "ran"
    env = _env(tmp_path, STUB_TORCH="hang", USDP_GPU_CHECK="1", USDP_CUDA_TIMEOUT="30", USDP_LOG=str(log))
    p = subprocess.Popen(["bash", ENTRY, "touch", str(ran)], env=env, start_new_session=True)
    time.sleep(1)
    os.killpg(p.pid, sig)
    assert p.wait(timeout=10) == code
    assert not ran.exists()
    assert log.read_text().splitlines()[-1] == f"[run.sh] exit {code}"


def test_TC_DRUN_04_7_SignalToEntryDuringCheckHonored(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("")
    ran = tmp_path / "ran"
    env = _env(tmp_path, STUB_TORCH="hang", USDP_GPU_CHECK="1", USDP_CUDA_TIMEOUT="30", USDP_LOG=str(log))
    p = subprocess.Popen(["bash", ENTRY, "touch", str(ran)], env=env, start_new_session=True)
    time.sleep(1)
    os.kill(p.pid, signal.SIGINT)
    assert p.wait(timeout=15) == 130
    assert not ran.exists()
    assert log.read_text().splitlines()[-1] == "[run.sh] exit 130"


# ---- MREQ-DRUN-05 sweeps: the expected outcome of every case is written by hand from MOD-DockerRun §4.1 ----

GPU_SPECS = {
    "none": None,
    "2": ["2"],
    "7": ["7"],
    "02": ["02"],
    "2,3": ["2,3"],
    "all": ["all"],
    "-1": ["-1"],
    "empty": [""],
    "uuid": ["GPU-00000002"],
    "twice": ["3", "2"],
}
GPU_BUSY = {
    "free": {},
    "proc": {"STUB_BUSY": "2"},
    "usdp": {"STUB_USDP_GPUS": "2"},
    "gpu3": {"STUB_BUSY": "3"},
    "usdp3": {"STUB_USDP_GPUS": "3"},
}
SMI = ["ok", "fail", "empty", "garbage", "hang", "apps_fail", "apps_hang"]
FORCE = ["none", "before", "after", "nosep", "abbrev"]
MODES = ["fg", "bg"]
# --mount forms for tmp_path/extra: remapped to /x, same path, same path read-only
MOUNT_FORMS = {"remap": "{}:/x", "same": "{}", "ro": "{}::ro"}

G_CASES = list(itertools.product(GPU_SPECS, GPU_BUSY, SMI, FORCE, MODES))
M_CASES = list(itertools.product(MOUNT_STATES, FORCE, MODES))
R_CASES = list(
    itertools.product(
        ["same", "diff"],
        ["present", "missing", "daemon"],
        ["none", "inside", "outside", "in_mount"],
        ["none", "remap", "same", "ro", "missing"],
        ["new", "used"],
        MODES,
    )
)
C_CASES = list(itertools.product(["one", "zero", "importerror", "two", "hang"], MODES))


def _args(opts, force, cmd=("true",)):
    """Assemble run.sh arguments for one --force placement."""
    if force == "before":
        return [*opts, "--force", "--", *cmd]
    if force == "after":
        return [*opts, "--", *cmd, "--force"]
    if force == "nosep":
        return [*opts, "--force", *cmd]
    if force == "abbrev":
        return [*opts, "--forc", "--", *cmd]
    return [*opts, "--", *cmd]


def _check(outcome, rc, err, argv):
    if outcome == "start":
        assert rc == 0 and argv is not None, err
    elif outcome == "usage":
        assert rc == 2 and argv is None and "usage" in err, err
    else:
        assert rc == 1 and argv is None and "run.sh: refused:" in err, err


def _expect_g(spec, busy, smi, force):
    if spec == "twice" or force in ("nosep", "abbrev"):
        return "usage"
    if spec == "none":
        return "start"
    if spec != "2" or smi != "ok":
        return "refuse"
    if busy in ("proc", "usdp") and force != "before":
        return "refuse"
    return "start"


@pytest.mark.parametrize("spec,busy,smi,force,mode", G_CASES)
def test_TC_DRUN_05_1_GpuGuardSweep(tmp_path, spec, busy, smi, force, mode):
    env = _env(tmp_path, STUB_SMI=smi, **GPU_BUSY[busy])
    opts = [a for g in (GPU_SPECS[spec] or []) for a in ("--gpu", g)]
    if mode == "bg":
        opts += ["--name", "g"]
    rc, _, err, argv = _run(_args(opts, force), env)
    _check(_expect_g(spec, busy, smi, force), rc, err, argv)


def _expect_m(state, force):
    if force in ("nosep", "abbrev"):
        return "usage"
    return "start" if state == "sshfs" else "refuse"


@pytest.mark.parametrize("state,force,mode", M_CASES)
def test_TC_DRUN_05_2_DataMountGuardSweep(tmp_path, state, force, mode):
    env = _env(tmp_path, mount=state)
    opts = ["--name", "m"] if mode == "bg" else []
    start = time.monotonic()
    rc, _, err, argv = _run(_args(opts, force), env)
    _check(_expect_m(state, force), rc, err, argv)
    if _expect_m(state, force) == "refuse":
        assert "sshfs -o allow_other" in err
        assert time.monotonic() - start < 5


def _expect_r(uid, image, log, src, name, mode):
    if mode == "fg" and log != "none":
        return "usage"
    if image != "present" or uid == "diff" or src == "missing":
        return "refuse"
    if mode == "bg" and (log == "outside" or (log == "in_mount" and src != "same") or name == "used"):
        return "refuse"
    return "start"


@pytest.mark.parametrize("uid,image,log,src,name,mode", R_CASES)
def test_TC_DRUN_05_3_RunConditionGuardSweep(tmp_path, uid, image, log, src, name, mode):
    env = _env(
        tmp_path,
        STUB_UID="1017" if uid == "same" else "1018",
        STUB_IMAGE=image,
        STUB_NAME_USED="r" if name == "used" else "",
    )
    repo = env["USDP_TEST_REPO"]
    opts = ["--name", "r"] if mode == "bg" else []
    log_path = os.path.join(repo, "logs", "r.log")
    if log == "inside":
        opts += ["--log", log_path]
    elif log == "outside":
        opts += ["--log", str(tmp_path / "outside" / "r.log")]
    elif log == "in_mount":
        opts += ["--log", str(tmp_path / "extra" / "r.log")]
    if src in MOUNT_FORMS:
        (tmp_path / "extra").mkdir()
        opts += ["--mount", MOUNT_FORMS[src].format(tmp_path / "extra")]
    elif src == "missing":
        opts += ["--mount", f"{tmp_path / 'nope'}:/x"]
    os.makedirs(os.path.dirname(log_path))
    with open(log_path, "w") as f:
        f.write("old\n")
    rc, _, err, argv = _run(_args(opts, "none"), env)
    outcome = _expect_r(uid, image, log, src, name, mode)
    _check(outcome, rc, err, argv)
    if outcome == "refuse" and image == "missing":
        assert "not found; build it first with docker/build.sh" in err
    if outcome == "refuse" and image == "daemon":
        assert "Cannot connect to the Docker daemon" in err and "build it first" not in err
    with open(log_path) as f:
        assert f.read() == "old\n"


@pytest.mark.parametrize("torch,mode", C_CASES)
def test_TC_DRUN_05_4_InContainerCudaGuardSweep(tmp_path, torch, mode):
    env = _env(tmp_path, STUB_TORCH=torch, USDP_GPU_CHECK="1")
    log = tmp_path / "job.log"
    if mode == "bg":
        log.write_text("")
        env["USDP_LOG"] = str(log)
    ran = tmp_path / "ran"
    r = subprocess.run(["bash", ENTRY, "touch", str(ran)], env=env, capture_output=True, text=True, timeout=60)
    if torch == "one":
        assert r.returncode == 0 and ran.exists(), r.stderr
    else:
        assert r.returncode == 3 and not ran.exists() and "entry.sh: refused" in r.stderr
    if mode == "bg":
        assert log.read_text().splitlines()[-1] == f"[run.sh] exit {0 if torch == 'one' else 3}"


@pytest.mark.parametrize("where", ["mount", "target", "repo", "home"])
def test_TC_DRUN_05_7_CommaPathsRefused(tmp_path, where):
    env = _env(tmp_path / "c,d" if where == "repo" else tmp_path)
    args = ["--", "true"]
    if where == "mount":
        (tmp_path / "a,b").mkdir()
        args = ["--mount", str(tmp_path / "a,b"), *args]
    elif where == "target":
        (tmp_path / "src").mkdir()
        args = ["--mount", f"{tmp_path / 'src'}:/x,y", *args]
    elif where == "home":
        (tmp_path / "h,ome").mkdir()
        env["HOME"] = str(tmp_path / "h,ome")
    rc, _, err, argv = _run(args, env)
    assert rc == 1 and argv is None and "comma" in err, err


@pytest.mark.parametrize(
    "args",
    [
        ["--name", "--force", "--", "true"],
        ["--name", "-x", "--", "true"],
        ["--name", "a b", "--", "true"],
        ["--name", "n", "--log", "", "--", "true"],
        ["--gpu", "--force", "--", "true"],
        ["--mount", "--offline", "--", "true"],
    ],
)
def test_TC_DRUN_05_8_OptionLookingValuesRejected(tmp_path, args):
    rc, _, err, argv = _run(args, _env(tmp_path))
    assert rc == 2 and argv is None and "usage" in err, err


def test_TC_DRUN_05_9_SweepSizesMatchSpec():
    assert (len(G_CASES), len(M_CASES), len(R_CASES), len(C_CASES)) == (3500, 60, 480, 10)


def test_TC_DRUN_05_10_ForcePrintsNotice(tmp_path):
    rc, _, err, argv = _run(["--gpu", "2", "--force", "--", "true"], _env(tmp_path, STUB_BUSY="2"))
    assert rc == 0 and argv is not None, err
    assert "--force: starting on GPU 2 although it is in use" in err


def test_TC_DRUN_05_11_LogShadowedByFileMountRefused(tmp_path):
    env = _env(tmp_path)
    other = tmp_path / "other.log"
    other.write_text("")
    target = os.path.join(env["USDP_TEST_REPO"], "logs", "n.log")
    rc, _, err, argv = _run(["--name", "n", "--mount", f"{other}:{target}", "--", "true"], env)
    assert rc == 1 and argv is None and "run.sh: refused:" in err, err


@pytest.mark.parametrize("value,outcome", [("abc", "usage"), ("0", "usage"), ("-1", "usage"), ("0.01", "start")])
def test_TC_DRUN_05_12_ProbeTimeoutValidated(tmp_path, value, outcome):
    rc, _, err, argv = _run(["--", "true"], _env(tmp_path, USDP_PROBE_TIMEOUT=value))
    _check(outcome, rc, err, argv)


def test_TC_DRUN_05_13_BusyCheckFailsClosed(tmp_path):
    rc, _, err, argv = _run(["--gpu", "2", "--", "true"], _env(tmp_path, STUB_PS_FAIL="1"))
    assert rc == 1 and argv is None and "docker ps" in err, err
