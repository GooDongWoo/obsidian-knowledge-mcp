from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from knowledge_mcp.config import Settings
from knowledge_mcp.daemon import (
    get_daemon_pid,
    is_daemon_running,
    run_daemon,
    start_daemon_process,
    stop_daemon_process,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        tmp_path / "vault",
        tmp_path / "vault" / ".knowledge",
        Settings.DEFAULT_QDRANT_URL,
        "test_collection",
        Settings.DEFAULT_DENSE_MODEL,
        "codex",
        project_root=tmp_path / "project",
    )


def test_get_daemon_pid(tmp_path):
    runtime_dir = tmp_path / ".knowledge"
    runtime_dir.mkdir(parents=True)
    assert get_daemon_pid(runtime_dir) is None

    (runtime_dir / "daemon.pid").write_text("12345\n", encoding="utf-8")
    assert get_daemon_pid(runtime_dir) == 12345

    (runtime_dir / "daemon.pid").write_text("invalid", encoding="utf-8")
    assert get_daemon_pid(runtime_dir) is None


def test_separate_clients_serialize_daemon_startup(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "start.py"
    script.write_text('''
import sys, time
from pathlib import Path
from types import SimpleNamespace
import knowledge_mcp.daemon as daemon
from knowledge_mcp.config import Settings
root = Path(sys.argv[1])
settings = Settings(root / 'vault', root / 'runtime', Settings.DEFAULT_QDRANT_URL,
                    'test', Settings.DEFAULT_DENSE_MODEL, 'test', project_root=root)
ready = root / 'ready'
daemon.is_daemon_running = lambda **_: ready.exists()
def launch(*args, **kwargs):
    with (root / 'starts').open('a') as log:
        log.write('start\\n')
    time.sleep(0.3)
    (settings.runtime_dir / 'daemon.pid').write_text('4242')
    ready.touch()
    return SimpleNamespace(pid=4242, poll=lambda: None)
daemon.subprocess.Popen = launch
assert daemon.start_daemon_process(settings, timeout=5) == 4242
''', encoding="utf-8")
    children = [subprocess.Popen([sys.executable, str(script), str(tmp_path)]) for _ in range(2)]
    try:
        assert [child.wait(timeout=30) for child in children] == [0, 0]
        assert (tmp_path / "starts").read_text().splitlines() == ["start"]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()


def test_is_daemon_running_success():
    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=DummyResponse()):
        assert is_daemon_running(port=8765) is True


def test_is_daemon_running_failure():
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        assert is_daemon_running(port=8765) is False


def test_start_daemon_process_already_running(settings):
    with patch("knowledge_mcp.daemon.is_daemon_running", return_value=True), \
         patch("knowledge_mcp.daemon.get_daemon_pid", return_value=42):
        pid = start_daemon_process(settings, port=8765)
        assert pid == 42


def test_start_daemon_process_launches(settings):
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = None

    running_states = [False, False, True]

    def fake_is_running(port=8765, host="127.0.0.1"):
        return running_states.pop(0) if running_states else True

    with patch("knowledge_mcp.daemon.is_daemon_running", side_effect=fake_is_running), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        pid = start_daemon_process(settings, port=8765, timeout=5.0)
        assert pid == 9999
        assert mock_popen.called
        assert mock_popen.call_args.kwargs["stdout"] == __import__("subprocess").DEVNULL
        assert mock_popen.call_args.kwargs["stderr"] == __import__("subprocess").DEVNULL


def test_start_daemon_process_win32_headless(settings):
    import subprocess
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.poll.return_value = None

    with patch("knowledge_mcp.daemon.is_daemon_running", side_effect=[False, False, True]), \
         patch("sys.platform", "win32"), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        start_daemon_process(settings, port=8765, timeout=5.0)
        assert mock_popen.called
        kwargs = mock_popen.call_args.kwargs
        assert kwargs["creationflags"] == (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        assert kwargs["startupinfo"] is not None
        assert kwargs["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
        assert kwargs["startupinfo"].wShowWindow == subprocess.SW_HIDE
        args = mock_popen.call_args.args[0]
        assert args[0].endswith("pythonw.exe")


def test_stop_daemon_process(settings):
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    (settings.runtime_dir / "daemon.pid").write_text("9999", encoding="utf-8")

    running_states = [True, False]

    def fake_is_running(port=8765, host="127.0.0.1"):
        return running_states.pop(0) if running_states else False

    with patch("knowledge_mcp.daemon.is_daemon_running", side_effect=fake_is_running), \
         patch("subprocess.run") as mock_run:
        stopped = stop_daemon_process(settings, port=8765, timeout=2.0)
        assert stopped is True
        assert not (settings.runtime_dir / "daemon.pid").exists()


def test_startup_timeout_terminates_owned_child_before_unlock(settings):
    process = MagicMock(pid=9999)
    process.poll.return_value = None
    with patch("knowledge_mcp.daemon.is_daemon_running", return_value=False), \
         patch("subprocess.Popen", return_value=process), \
         patch("subprocess.run") as kill:
        with pytest.raises(TimeoutError):
            start_daemon_process(settings, timeout=0)
    assert kill.call_args.args[0] == ["taskkill", "/F", "/T", "/PID", "9999"]
    process.wait.assert_called_once()


def test_run_daemon_registers_tools_and_runs_streamable_http(settings):
    class FakeIndexer:
        async def sync(self, assume_locked=False):
            from knowledge_mcp.indexer import IndexRunSummary
            return IndexRunSummary(added=1)

    class FakeStore:
        stores = {}

    mcp_mock = MagicMock()
    tools = {}

    def fake_tool(*args, **kwargs):
        def decorator(fn):
            tools[kwargs.get("name", fn.__name__)] = fn
            return fn
        return decorator

    mcp_mock.tool = fake_tool
    mcp_mock.custom_route = MagicMock(return_value=lambda fn: fn)

    with patch("knowledge_mcp.cli.ensure_qdrant"), \
         patch("knowledge_mcp.cli._dependencies", return_value=(FakeStore(), MagicMock(), {"m": FakeIndexer()})), \
         patch("knowledge_mcp.daemon.create_application") as mock_create_app:
        mock_create_app.return_value = MagicMock(mcp=mcp_mock)
        run_daemon(settings, host="127.0.0.1", port=8765)

        assert "knowledge-index-sync" in tools
        assert mcp_mock.run.called
        kwargs = mcp_mock.run.call_args[1]
        assert kwargs["transport"] == "http"
        assert kwargs["path"] == "/mcp"
        assert kwargs["host_origin_protection"] is True
        assert "127.0.0.1" in kwargs["allowed_hosts"]
        assert kwargs["port"] == 8765


def test_daemon_logging_rotates_during_one_process_without_sensitive_text(tmp_path, monkeypatch):
    import logging
    import sys
    import knowledge_mcp.daemon as daemon

    monkeypatch.setenv("KNOWLEDGE_DAEMON_LOG_MAX_BYTES", "256")
    monkeypatch.setenv("KNOWLEDGE_DAEMON_LOG_BACKUP_COUNT", "2")
    with daemon.daemon_logging(tmp_path):
        for _ in range(100):
            logging.getLogger("third_party").error("sensitive-query and private/path.md")
            print("sensitive-document", file=sys.stderr)
        try:
            raise RuntimeError("sensitive-exception")
        except RuntimeError:
            logging.exception("sensitive-error")
        logging.error("ignored", extra={"event_code": "sensitive"})
    files = sorted(tmp_path.glob("daemon.log*"))
    assert len(files) == 3
    assert all(path.stat().st_size <= 256 for path in files)
    stored = "".join(path.read_text(encoding="utf-8") for path in files)
    assert "runtime_message" in stored
    assert "runtime_stderr" in stored
    assert "sensitive" not in stored
    assert "private/path.md" not in stored


def test_daemon_failure_log_has_no_exception_payload(settings):
    with patch("knowledge_mcp.cli.ensure_qdrant", side_effect=RuntimeError("sensitive-vault-path")):
        with pytest.raises(RuntimeError, match="sensitive-vault-path"):
            run_daemon(settings)
    stored = (settings.runtime_dir / "daemon.log").read_text(encoding="utf-8")
    assert "daemon_failed" in stored
    assert "sensitive" not in stored


def test_daemon_logging_restores_streams_and_handlers_after_failure(tmp_path, capsys):
    import logging
    import sys
    import knowledge_mcp.daemon as daemon

    stdout, stderr = sys.stdout, sys.stderr
    root = logging.getLogger()
    handlers = root.handlers[:]
    with pytest.raises(RuntimeError):
        with daemon.daemon_logging(tmp_path, console=True):
            print("interactive diagnostic", file=sys.stderr)
            raise RuntimeError("failure")
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert root.handlers == handlers
    print("after cleanup", file=sys.stderr)
    assert "interactive diagnostic" in capsys.readouterr().err
    assert "interactive diagnostic" not in (tmp_path / "daemon.log").read_text(encoding="utf-8")
