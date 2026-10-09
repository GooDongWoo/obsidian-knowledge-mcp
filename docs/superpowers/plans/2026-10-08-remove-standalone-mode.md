# Standalone 기능 완전 제거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `obsidian-knowledge-mcp`에서 인프로세스로 ML 모델을 단독 로드하던 레거시 `--standalone` 실행 모드 및 관련 분기 코드를 완전 삭제하고, 단일 HTTP 데몬 + 경량 stdio 프록시 아키텍처로 일원화한다.

**Architecture:** `knowledge-mcp serve` 명령을 데몬 자동 기동 및 stdio 프록시 전용으로 단순화한다. `--standalone` CLI 옵션, `daemon.py`의 `sync_tool` 조건부 분기를 제거하고, 관련 단위/와이어 테스트와 문서를 프록시/데몬 단일 모델에 맞게 정비한다.

**Tech Stack:** Python 3.11+, FastMCP 4.0.10, pytest, anyio.

**Spec:** 사용자 요구사항 (독립 standalone 기능 배제 및 관련 코드·테스트·문서 전면 삭제)

## Global Constraints

- 데몬 기반 단일 HTTP 서버(`http://127.0.0.1:8765/mcp`) 및 stdio 프록시(`run_stdio_proxy`)의 정상 동작을 100% 보장한다.
- Vault 원문, Qdrant 벡터 컬렉션, SQLite 메타데이터 상태는 변경하지 않는다.
- 데몬이 제공하는 3대 도구(`qdrant-find`, `knowledge-index-status`, `knowledge-index-sync`)의 계약을 온전히 유지한다.
- 모든 pytest 테스트가 성공적으로 통과해야 한다.

---

### Task 1: CLI 및 데몬 팩토리에서 standalone 제거

**Files:**
- Modify: `src/knowledge_mcp/cli.py:15-22, 71-76, 137-151`
- Modify: `src/knowledge_mcp/daemon.py:435-446`
- Test: `tests/test_cli_paths.py`

**Interfaces:**
- Consumes: `Settings`, `is_daemon_running`, `start_daemon_process`, `run_stdio_proxy`
- Produces: `serve` 명령어에서 `--standalone` 옵션 제거, `create_daemon_application(settings)` 단일화

- [ ] **Step 1: `src/knowledge_mcp/cli.py`에서 `--standalone` 옵션 및 분기 제거**

`src/knowledge_mcp/cli.py`에서 다음을 수정:
1. 상단 `from .daemon import (...)`에서 `create_daemon_application` import 제거.
2. `_parser()`의 `serve_parser` 정의 수정:
```python
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run MCP stdio proxy")
    serve_parser.add_argument("--client", choices=("codex", "claude-code", "antigravity"), default="codex")
    serve_parser.add_argument("--host", default=DEFAULT_DAEMON_HOST, help="Daemon host")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_DAEMON_PORT, help="Daemon port")
```
(`--standalone` 인자 완전 삭제)
3. `main()`의 `command == "serve"` 분기 수정:
```python
        if command == "serve":
            # Ensure daemon is running and proxy stdio
            if not is_daemon_running(port=port, host=host):
                start_daemon_process(settings, port=port, host=host)
            import anyio
            from .proxy import run_stdio_proxy

            anyio.run(run_stdio_proxy, f"http://{host}:{port}/mcp", settings)
            return 0
```

- [ ] **Step 2: `src/knowledge_mcp/daemon.py`에서 `sync_tool` 파라미터 제거**

`src/knowledge_mcp/daemon.py`의 `create_daemon_application` 함수 시그니처 및 도구 등록 조건 수정:
```python
def create_daemon_application(settings: Settings):
    """Register tools/health without touching Qdrant, models or the state DB."""
    runtime = _DaemonRuntime(settings)
    application = create_application(settings, lifespan=runtime.lifespan)
    runtime.application = application
    application.runtime_status = runtime.status

    @application.mcp.tool(name="knowledge-index-sync", description="Trigger synchronization of the local Vault index.")
    async def knowledge_index_sync(rebuild: bool = False, force_full_hash: bool = False) -> dict[str, Any]:
        return await runtime.sync(rebuild=rebuild, force_full_hash=force_full_hash)

    @application.mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        from starlette.responses import JSONResponse
```

- [ ] **Step 3: CLI parser 및 serve 분기 변경 확인**

Run in PowerShell:
```powershell
.venv\Scripts\python -c "from knowledge_mcp.cli import _parser; p = _parser(); args = p.parse_args(['serve']); assert not hasattr(args, 'standalone'); print('CLI Parser OK')"
```
Expected: `CLI Parser OK`

---

### Task 2: 테스트 코드 수정 및 standalone 테스트 제거

**Files:**
- Modify: `tests/test_cli_paths.py:103-154`
- Modify: `tests/test_mcp_wire.py:397-420`
- Test: `tests/test_cli_paths.py`, `tests/test_mcp_wire.py`

**Interfaces:**
- Consumes: `cli._parser`, `create_daemon_application`
- Produces: standalone 플래그 거부 검증 및 데몬 애플리케이션 수명주기 직접 검증 테스트

- [ ] **Step 1: `tests/test_cli_paths.py`에서 standalone 관련 테스트 교체 및 플래그 거부 테스트 추가**

기존 `test_serve_starts_with_partial_file_failures`는 `serve --standalone` CLI 명령을 통해 간접적으로 lifespan을 실행했음. 이를 `create_daemon_application`의 lifespan과 부분 에러 상태를 직접 검증하는 테스트로 전환하고, `serve --standalone` 실행 시 인자 오류가 발생하는지 검증하는 테스트를 추가한다:

```python
def test_serve_rejects_standalone_flag():
    import pytest
    from knowledge_mcp.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["serve", "--standalone"])


def test_daemon_application_starts_with_partial_file_failures(tmp_path, monkeypatch):
    import asyncio
    from fastmcp import Client
    from knowledge_mcp import cli
    from knowledge_mcp.daemon import create_daemon_application
    from knowledge_mcp.indexer import IndexRunSummary
    from knowledge_mcp.config import Settings
    from knowledge_mcp.state import OperationLog
    from test_server import FakeStore

    settings = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")

    class Indexer:
        async def sync(self, **kwargs):
            return IndexRunSummary(failed=1, error_codes=["parse_failed"])

    monkeypatch.setattr(cli, "ensure_qdrant", lambda _: None)
    monkeypatch.setattr(cli, "_dependencies", lambda _: (
        FakeStore(), OperationLog(settings.runtime_dir), {"bge": Indexer()},
    ))

    app = create_daemon_application(settings)

    async def exercise_lifespan():
        async with Client(app.mcp) as client:
            async with asyncio.timeout(2):
                while app.runtime_status["state"] in {"starting", "indexing"}:
                    await asyncio.sleep(.01)
            status = (await client.call_tool("knowledge-index-status", {})).data
            assert status["state"] == "error"
            assert status["last_error"] == "parse_failed"
            assert status["last_index"]["bge"]["failed"] == 1
            assert {tool.name for tool in await client.list_tools()} == {
                "qdrant-find", "knowledge-index-status", "knowledge-index-sync"
            }

    asyncio.run(exercise_lifespan())
```

- [ ] **Step 2: `tests/test_mcp_wire.py`에서 `test_standalone_cli_runs_real_stdio_protocol` 제거**

`tests/test_mcp_wire.py`의 라인 397-420에 위치한 다음 테스트 함수를 완전히 삭제:
```python
@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_standalone_cli_runs_real_stdio_protocol(tmp_path, mode):
    ...
```

- [ ] **Step 3: 수정된 테스트 실행 및 확인**

Run in PowerShell:
```powershell
.venv\Scripts\pytest tests/test_cli_paths.py
```
Expected: All tests pass.

---

### Task 3: 문서 및 운영 가이드 최신화

**Files:**
- Modify: `README.md:28, 272`
- Modify: `docs/user-guide-ko.md:36, 174, 271`

**Interfaces:**
- Consumes: 없음
- Produces: `--standalone` 옵션 및 인프로세스 실행 관련 언급이 제거된 일관된 문서

- [ ] **Step 1: `README.md` 수정**

1. 28행 수정:
기존:
```markdown
2. **Lightweight Stdio Protocol Proxy**: AI clients launch `knowledge-mcp serve` without importing heavy ML libraries. The official FastMCP proxy mirrors the client's protocol era: modern clients use MCP `2026-07-28` on both sides, while legacy clients retain normal initialize-based negotiation. Modern `server/discover` requests are supported without a forced legacy fallback.
```
(이 부분은 유지하되, 만약 본문 다른 곳에 standalone이 있으면 수정)
2. 272행 수정:
기존:
```markdown
`KNOWLEDGE_DAEMON_START_TIMEOUT` (default 900 seconds) now bounds waiting for the HTTP listener, independently of model/index readiness. The proxy's normal tool-call deadline remains 15 seconds. `serve --standalone` uses the same managed initialization lifecycle while preserving its two read tools.
```
수정 후:
```markdown
`KNOWLEDGE_DAEMON_START_TIMEOUT` (default 900 seconds) bounds waiting for the HTTP listener, independently of model/index readiness. The proxy's normal tool-call deadline remains 15 seconds.
```

- [ ] **Step 2: `docs/user-guide-ko.md` 수정**

1. 36행 수정:
기존:
```markdown
daemon 직접 HTTP, 기본 stdio proxy, standalone stdio 모두 최신 protocol을 지원하고, 구 클라이언트는 SDK의 정상 initialize 기반 협상으로 같은 `/mcp` 데몬을 이용합니다.
```
수정 후:
```markdown
daemon 직접 HTTP 및 stdio proxy 모두 최신 protocol을 지원하고, 구 클라이언트는 SDK의 정상 initialize 기반 협상으로 같은 `/mcp` 데몬을 이용합니다.
```

2. 174행의 `--standalone` 플래그 설명 불릿 완전 제거:
기존:
```markdown
- `--standalone` 플래그를 주면 데몬 없이 단독 stdio 프로세스로 실행할 수 있으며 MCP `2026-07-28`과 legacy 협상을 지원합니다. 이 경로는 자체 모델을 로드하므로 여러 클라이언트가 모델을 공유하려면 기본 `serve`를 사용하세요. standalone은 읽기 도구 `qdrant-find`, `knowledge-index-status` 두 개를 제공합니다.
```
삭제.

3. 271행 수정:
기존:
```markdown
실행 중인 네이티브 모델 작업 때문에 정상 종료가 늦어질 수 있습니다. `serve --standalone`도 같은 초기화 수명주기를 사용하며 읽기 도구 두 개를 유지합니다.
```
수정 후:
```markdown
실행 중인 네이티브 모델 작업 때문에 정상 종료가 늦어질 수 있습니다.
```

- [ ] **Step 3: 문서에서 남아있는 standalone 키워드 검증**

Run in PowerShell:
```powershell
git grep -i "standalone" README.md docs/user-guide-ko.md
```
Expected: 아키텍처 다이어그램 HTML 관련 standalone SVG 언급 외에 MCP 서버 standalone 실행 관련 언급이 없어야 함.

---

### Task 4: 회귀 검증 및 최종 테스트 스위트 확인

**Files:**
- Verify: 전체 테스트 및 CLI 동작

- [ ] **Step 1: 전체 단위 및 통합 테스트 실행**

Run in PowerShell:
```powershell
.venv\Scripts\pytest tests/test_cli_paths.py tests/test_daemon.py tests/test_proxy.py
```
Expected: All tests pass.

- [ ] **Step 2: CLI 도움말 확인**

Run in PowerShell:
```powershell
.venv\Scripts\knowledge-mcp.exe serve --help
```
Expected: `--standalone` 옵션이 표시되지 않고 깔끔한 stdio proxy 옵션만 출력됨.

- [ ] **Step 3: Git diff 최종 확인**

Run in PowerShell:
```powershell
git diff --stat
```
Expected: `src/knowledge_mcp/cli.py`, `src/knowledge_mcp/daemon.py`, `tests/test_cli_paths.py`, `tests/test_mcp_wire.py`, `README.md`, `docs/user-guide-ko.md` 변경 내역 확인.
