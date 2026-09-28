# FastMCP 4 및 MCP 2026-07-28 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. 현재 요청은 계획 작성·수정까지이며 실제 코드 변경과 운영 전환은 후속 구현 작업이다.

**Goal:** FastMCP `2.7.0` → `4.0.10`, MCP SDK v1 → v2, 기존 HTTP+SSE → Streamable HTTP, discovery 강제 폴백 → MCP `2026-07-28` 지원을 함께 완료한다.

**Architecture:** 기본 경로는 `MCP 클라이언트 → stdio 프로토콜 프록시 → http://127.0.0.1:8765/mcp → 단일 모델 데몬`이다. FastMCP 4 공식 proxy가 앞쪽 클라이언트와 뒤쪽 데몬의 protocol era를 맞추게 하고, 수동 JSON-RPC 중계와 `server/discover` 차단을 제거한다. 구 클라이언트는 SDK의 정상 legacy 협상을 통해 같은 `/mcp` 서버를 이용할 수 있지만 최신 클라이언트는 modern 프로토콜로 동작해야 한다.

**Tech Stack:** Python `>=3.11`, FastMCP `4.0.10`, MCP SDK `2.2.0` 검증 기준, Streamable HTTP, stdio, AnyIO, Pydantic, Starlette, pytest, Windows, 기존 Qdrant 및 로컬 ML 모델.

**Spec:** 이 문서의 [범위와 완료 조건](#범위와-완료-조건)이 요구사항이다. 2026-09-29 초안의 SSE 유지·강제 discovery 폴백·modern 지원 후속 분리는 이번 수정으로 대체한다.

## Global Constraints

- 조사일 기준 최신 안정판 `fastmcp==4.0.10`을 고정한다. 실행 시작일 최신판 재확인 후 목표가 바뀌면 계획·검증 기준도 갱신한다.
- 최신 지원 프로토콜 `2026-07-28`은 daemon 직접 HTTP, 기본 stdio proxy, standalone stdio 세 경로에서 모두 검증한다.
- 외부 stdio 진입점인 `knowledge-mcp serve`와 `serve --standalone`을 유지한다. daemon 기본 endpoint는 `/sse`에서 `/mcp`로 바꾼다.
- 단일 데몬·단일 모델 로딩·기존 검색 계약·필터·개인정보 정책·CPU/GPU 제한을 보존한다.
- 모델·벡터 차원·컬렉션 이름·Qdrant 버전·SQLite 스키마를 변경하지 않고 rebuild/컬렉션 삭제 없이 전환한다.
- 구환경과 구 checkout을 보존한다. 원래 `.venv`를 제자리 업그레이드하거나 복사·이름 변경해 재사용하지 않는다.
- 새 프로토콜의 필수 동작은 이번 작업에 포함한다. 사용하지 않는 optional 기능인 Apps/tasks/MRTR 도구 등을 모두 새로 구현할 필요는 없다.
- 배포 시 구 proxy를 먼저 종료하고 데몬을 교체한다. 같은 runtime_dir/포트에서 구·신 데몬을 동시에 띄우지 않는다.

---

## 범위와 완료 조건

1. 새 환경 설치와 재설치 및 `pip check`가 통과한다.
2. 새 데몬은 Streamable HTTP `/mcp`와 별도 `/health`를 제공하며 구 `/sse`를 기본 경로로 사용하지 않는다.
3. 최신 클라이언트의 `server/discover`에 정상 응답하고 `2026-07-28`을 사용한다. 지원 가능한 최신 요청을 임의로 `-32601` 처리해 legacy로 강등시키지 않는다.
4. 최신 요청은 initialize와 protocol session ID 없이 처리된다. `_meta` 및 HTTP headers의 protocol version/capabilities/method가 SDK를 통해 올바르게 전달된다.
5. 공식 proxy의 기본 era mirroring으로 modern→modern, legacy→legacy를 유지한다. 구 클라이언트용 협상은 최신 경로와 별도로 시험한다.
6. daemon의 세 도구 `qdrant-find`, `knowledge-index-status`, `knowledge-index-sync`와 standalone의 기존 읽기 도구 두 개를 유지한다.
7. 실제 wire에서 검색 결과·빈 결과·오류, 한글·소스 경로·행 번호·점수·private 필터가 유지된다.
8. 데몬 종료·timeout·cancel·stdio EOF·동시 호출·재기동이 정해진 시간 내 처리되고 요청당 최종 응답은 한 번만 전달된다.
9. 구 코드·구환경·구 client 설정으로 롤백할 수 있다. 최신 클라이언트에서 legacy로 내려가야만 성공하는 상태는 완료로 판정하지 않는다.

## 기존 초안이 잘못 제한한 부분

SSE 유지와 discovery 차단은 package 업그레이드만 먼저 수행하는 임시 전환 전략이었다. 하지만 이 저장소의 proxy는 최신 클라이언트가 보낸 `server/discover`를 의도적으로 거절하므로 FastMCP 4를 설치해도 실제 사용자 경로의 최신 protocol 지원을 막는다. 최신 migration 완료 목표에는 맞지 않아 제거한다.

구 HTTP+SSE 전송과 Streamable HTTP의 응답에 사용되는 SSE는 다르다. `/sse`의 지속 연결·별도 message endpoint를 없애고 `/mcp`의 요청별 POST 및 JSON/SSE 응답을 사용한다. Streamable HTTP 응답에서 SSE가 나타나는 것은 구 전송을 유지한다는 뜻이 아니다. [공식 Streamable HTTP 사양](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http).

최신 사양은 요청별 version 선언과 discovery를 제공하고 legacy initialize 방식도 함께 지원할 수 있다. 따라서 제거 대상은 프로젝트가 직접 만든 **강제 강등 처리**다. SDK의 상호 지원 버전 선택이나 실제 구 서버를 위한 정상 fallback 자체를 무조건 금지하지 않는다. [공식 버전·호환성 사양](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning).

## 조사 기준과 기술 선택

| 항목 | 확인한 상태 / 결정 |
|---|---|
| 현재 환경 | Python `3.13.5`, FastMCP `2.7.0`, MCP `1.30.0`, Pydantic `2.11.10` |
| 목표 의존성 | FastMCP `4.0.10`, MCP `2.2.0`; FastMCP는 같은 버전의 `fastmcp-slim[client,server]` 사용 |
| optional extra 충돌 | `mcp-server-qdrant==0.8.1`은 FastMCP `2.7.0` 및 Pydantic `<2.12.0` 고정; 새 프로젝트 환경에서 분리 |
| 목록 API | 테스트의 `get_tools()` dict 조회 → `list_tools()` list 조회 |
| SDK raw 메시지 | `mcp.shared.session` 제거, SessionMessage import 이동, JSONRPCMessage RootModel→union; 수동 중계 제거로 의존도 축소 |
| protocol proxy | 공식 `create_proxy(target)` 사용, backend `mode`를 강제 지정하지 않아 era mirroring 유지 |
| HTTP 상태 | modern은 요청별 처리. `stateless_http=True`를 서버 전체에 무조건 지정하지 않고 SDK가 protocol era에 따라 modern/legacy를 구분하도록 함 |
| 실제 코드 위험 | standalone async 함수에서 sync `run()` 호출; CLI import만으로 fastembed/onnxruntime이 로딩됨 |

버전·의존성 근거: [FastMCP 4.0.10](https://pypi.org/project/fastmcp/4.0.10/), [fastmcp-slim 메타데이터](https://pypi.org/pypi/fastmcp-slim/4.0.10/json), [MCP 2.2.0](https://pypi.org/pypi/mcp/2.2.0/json), [구 Qdrant 서버 의존성](https://pypi.org/pypi/mcp-server-qdrant/0.8.1/json).

공식 FastMCP proxy는 front/backend의 protocol era를 요청별로 맞춘다. target URL/transport를 전달하고 mode를 생략해야 이 기본 동작을 사용한다. 이미 구성한 Client를 전달하거나 mode를 지정하면 별도 협상 설정을 유지하므로 이번 기본 설계에 사용하지 않는다. provider 오류는 도구 목록이 조용히 비는 일이 없도록 `provider_error_strategy="raise"`로 전달한다. [공식 proxy 문서](https://gofastmcp.com/servers/providers/proxy), [v4.0.10 proxy 소스](https://github.com/PrefectHQ/fastmcp/blob/v4.0.10/fastmcp_slim/fastmcp/server/providers/proxy.py).

구 환경에서 아래 관련 테스트는 **28 passed, 1 Authlib deprecation warning**이었다. 새 환경·새 protocol 실행은 아직 검증하지 않았다.

```powershell
& .venv/Scripts/python.exe -m pytest tests/test_proxy.py tests/test_server.py tests/test_daemon.py tests/test_cli_paths.py -q
```

## 사이드 이펙트와 검증

| 위험 | 대응 / 필수 확인 |
|---|---|
| URL만 `/mcp`로 바꾸고 SSE client를 그대로 사용 | proxy 전체를 공식 protocol proxy로 교체; 실제 HTTP POST 및 응답 시험 |
| discovery 차단만 삭제해 version/header 의미가 어긋남 | SDK front server와 proxy backend mirroring 사용; `_meta`/header 일치와 modern→modern 확인 |
| 최신 요청을 session initialize/GET stream에 의존시킴 | initialize 없이 discovery·tools/list·tools/call 실행; modern에 legacy session ID/독립 GET stream 요구 없음 |
| 구 `/sse` 직접 연결 설정이 끊김 | 저장소·실제 구성에서 URL 검색, `/mcp`로 수정, endpoint 변경을 문서화. 기존 stdio 명령은 유지 |
| 공식 proxy가 실패한 provider를 skip해 빈 tools/list 반환 | `provider_error_strategy="raise"`, 데몬 부재/재기동/연결 실패 테스트 |
| proxy 구현 교체로 RAM과 응답 시간이 증가 | 실제 CLI와 proxy의 cold import/RSS, 동시 호출 p95 비교. 모델 생성은 daemon에만 유지 |
| auto-restart·timeout 정책 소실 | SDK protocol 처리 주변에 좁은 middleware로 health/restart/timeout 정책 보존. 수동 session dispatcher를 재구현하지 않음 |
| timeout/cancel된 sync가 부분 진행됨 | 완료 여부를 status로 확인; 자동 재전송 금지, index lock·재시도 가능성·manifest 보존 검증 |
| protocol ping 제거를 daemon 장애로 오인 | `/health`로 프로세스 health 확인, tools/list/call로 MCP 경로 확인. modern health에 ping 사용 금지 |
| origin/header 엄격성 때문에 localhost 요청 차단 | localhost binding, 허용 Origin/Host 명시, 정상 local request와 잘못된 Origin 거절 테스트 |
| 자동 직렬화/outputSchema 변경 | 구·신 raw content/structuredContent/error 비교, 필요한 경계만 보정 |
| 패키지 해석이 GPU 조합도 변경 | 현재 torch `2.11.0+cu128`, fastembed-gpu `0.8.0`, onnxruntime-gpu `1.26.0` 보존 및 실제 inference |
| FastMCP 설정 전에 import됨 | 기존 `.env` 로딩을 FastMCP import보다 먼저 수행, 자식 daemon 상속 확인 |
| 구 daemon health만 보고 새 버전으로 착각 | proxy 종료→daemon stop→새 환경 start; PID/version/discovery/검색을 함께 확인 |

SDK v2의 HTTP 오류·stdio descriptor·메시지 타입 변경은 [공식 SDK 마이그레이션](https://github.com/modelcontextprotocol/python-sdk/blob/v2.2.0/docs/migration.md)을 기준으로 검증한다. 현대 protocol의 ping 및 callback 제약은 [FastMCP 3→4 가이드](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3)를 확인한다.

## 변경 파일

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | FastMCP·직접 import하는 SDK/AnyIO 선언, 충돌 extra 제거 |
| `constraints/fastmcp4-windows-py313.txt` (신규) | 검증된 새 환경 제약; 다른 플랫폼용 lock으로 간주하지 않음 |
| `src/knowledge_mcp/proxy.py` | 공식 proxy와 health/restart/timeout 정책, 구 수동 중계 제거 |
| `src/knowledge_mcp/daemon.py` | Streamable HTTP `/mcp`, 별도 health, lifecycle |
| `src/knowledge_mcp/cli.py` | `/mcp` URL·status 안내, env 선행 로딩, ML lazy import, standalone async run |
| `src/knowledge_mcp/server.py` | 실제 응답 계약 차이가 확인될 때만 직렬화 경계 보정 |
| `tests/test_proxy.py`, `tests/test_daemon.py`, `tests/test_cli_paths.py`, `tests/test_server.py`, `tests/test_env_and_ignores.py` | 새 동작과 기존 정책 회귀 검증 |
| `tests/test_mcp_wire.py` (신규) | 실제 stdio·HTTP의 modern/legacy 협상·header·오류·취소 |
| `.env.example`, `README.md`, `docs/user-guide-ko.md` | 설치·설정·client 전환·복구 안내 |
| `docs/obsidian-knowledge-architecture.html`, `docs/architecture-dark.png`, `docs/architecture-light.png` | 현재 아키텍처의 HTTP+SSE 표기를 Streamable HTTP로 갱신 |

과거 날짜의 설계 기록은 덮어쓰지 않는다. 실제 client 구성의 변경 파일은 구현 시작 시 설정 inventory에서 찾고 경로와 수정값을 기록한다. credential·개인 설정은 저장소에 커밋하지 않는다.

## Task 1: 구환경 보존과 최신 의존성 설치

**Files:** `pyproject.toml`, 신규 constraints. **Produces:** 별도 새 환경의 `$candidatePython`, 복구할 baseline checkout/환경.

- [x] baseline HEAD·package list·client 명령·env·구 endpoint를 `.knowledge/migration/fastmcp4/`에 기록한다. 구 editable 환경이 참조하는 checkout은 그대로 보존하고 구현은 격리된 checkout에서 수행한다.
- [x] FastMCP pin을 `4.0.10`으로 바꾸고 직접 사용하는 `mcp==2.2.0`, `anyio>=4.9,<5`를 명시한다. 사용하지 않는 충돌 extra `qdrant-mcp`를 제거하고 구 extra 사용자는 별도 FastMCP 2 환경으로 안내한다.
- [x] 새 `.venv-fastmcp4`를 생성하고 `.git/info/exclude`에 추가한다. 아래 명령은 후속 구현 시 실행한다.

```powershell
& .venv/Scripts/python.exe -m venv .venv-fastmcp4
$candidatePython = Join-Path (Get-Location) '.venv-fastmcp4/Scripts/python.exe'
& $candidatePython -m pip install --upgrade pip
& $candidatePython -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& $candidatePython -m pip install -e '.[dev]'
& $candidatePython -m pip check
```

- [x] `pip list --format=freeze`에서 프로젝트 자체의 editable 항목을 제외해 constraints를 저장한다. 자격증명/로컬 경로 포함 여부를 확인하고 별도 새 환경에서 동일 constraints로 재설치해 `pip check`를 통과시킨다.
- [x] 이 단계에서 구 proxy import 실패는 SDK 전환의 예상 실패로 기록한다. dependency/GPU 충돌은 Task 2 이전에 해결한다.

**완료:** 구환경 복구 가능, 새 의존성 설치·재설치 성공. dependency 변경을 독립 커밋한다.

## Task 2: 데몬의 Streamable HTTP 전환

**Files:** `daemon.py`, `cli.py`, `tests/test_daemon.py`, `tests/test_cli_paths.py`, 신규 `tests/test_mcp_wire.py`.

**Interfaces:** daemon 시작 인자 host/port 유지, MCP URL은 `/mcp`, health는 `/health`.

- [x] 기존 daemon mock 테스트의 기대값을 `transport="http"`, `path="/mcp"`로 바꾸고 현재 코드에서 실패하는지 확인한다.
- [x] daemon의 서버 실행을 다음과 같이 변경한다. 기존 custom health route·초기 sync 1회·warmup 1회·PID 정리를 유지한다.

```python
mcp.run(transport="http", host=host, port=port, path="/mcp")
```

- [x] CLI의 proxy URL과 daemon status 메시지에서 `/sse`를 `/mcp`로 교체한다. daemon health 판정은 `/health`만 사용하고 구 `/sse` GET fallback을 제거한다.
- [x] localhost Host/Origin 검증을 목표 버전의 `host_origin_protection`/allowed 설정으로 구성한다. Origin 없는 정상 SDK local 요청은 허용하고 무관한 web Origin은 거절한다. 검증을 꺼서 통과시키지 않는다.
- [x] `tests/test_mcp_wire.py`에 `test_direct_http_modern_discovery`, `test_modern_http_without_initialize`, `test_http_header_body_version_match`, `test_http_invalid_origin`을 추가한다. FakeStore와 별도 포트로 실제 HTTP를 열어 검증한다.
- [x] 현대 요청의 headers `MCP-Protocol-Version`, `Mcp-Method`, tools/call의 `Mcp-Name` 및 body `_meta`를 SDK가 구성하게 한다. 불일치·unsupported version은 사양의 JSON-RPC 오류로 거절되는지 확인한다.
- [x] modern 요청에 protocol session ID/별도 GET stream/initialize가 필요하지 않은지 검사한다. HTTP JSON과 요청별 SSE 응답 모두 처리 가능해야 한다.

```powershell
& $candidatePython -m pytest tests/test_daemon.py tests/test_cli_paths.py tests/test_mcp_wire.py -q
```

**완료:** 실제 modern HTTP 요청으로 세 도구를 사용할 수 있다. 단순 200 health만으로 완료하지 않는다.

## Task 3: protocol proxy와 discovery 강제 폴백 제거

**Files:** `proxy.py`, `tests/test_proxy.py`, `tests/test_mcp_wire.py`.

**Interfaces:** `run_stdio_proxy(daemon_url, settings, timeout)`은 유지하고 기본 URL을 `/mcp`로 바꾼다. SDK가 front protocol과 backend era를 관리한다.

- [x] 기존 discovery 테스트를 교체한다. 목표 assertion은 `server/discover` 성공, modern version 사용, backend도 modern으로 호출됨이다. 더 이상 `-32601` 차단을 기대하지 않는다.
- [x] 구 `sse_client`, `stdio_server` raw stream forwarding, `_make_method_not_found_response`, RootModel wrapper, 직접 in-flight ID dispatch를 제거한다.
- [x] 공식 proxy 기반 실행을 사용한다. 아래는 핵심 생성·실행 경로이며 health/timeout 정책은 다음 단계에서 추가한다.

```python
from fastmcp.server import create_proxy

proxy = create_proxy(
    daemon_url,
    name="obsidian-knowledge-proxy",
    provider_error_strategy="raise",
)
await proxy.run_async(transport="stdio", show_banner=False)
```

- [x] backend `mode="legacy"` 또는 `mode="auto"`를 강제 지정하지 않는다. 설정된 Client 대신 URL 또는 transport target을 전달해 official default mirroring을 사용한다. SDK의 transport 인식·metadata 전달·version 오류를 자체 구현으로 우회하지 않는다.
- [x] 공식 Middleware의 `on_call_tool` 경계에서 기존 정책을 보존한다: health 부재 시 백그라운드 restart 1회와 기존 한국어 tool 오류, 정상 health이면 timeout 내 upstream 호출, timeout 후 상태 조회 및 필요할 때만 restart. retry로 tools/call을 자동 재전송하지 않는다.
- [x] timeout은 `KNOWLEDGE_PROXY_TIMEOUT`/함수 인자/기존 15초 기본값 순서로 적용하고 SDK 연결·discovery의 deadline과 구분한다. SDK 취소가 daemon 작업을 취소할 수도 있으므로 기존 “반드시 백그라운드에서 계속 실행”을 보장하지 않는다. 실제 sync lock·부분 상태를 시험한다.
- [x] provider/list 실패는 명시적 오류로 반환하고 빈 도구 목록을 성공처럼 반환하지 않는다. disconnect/timeout에서 최종 응답은 한 번, EOF에서 proxy 종료, 다른 동시 요청은 유지하는지 검사한다.
- [x] `test_proxy_modern_era_mirrors_backend`, `test_proxy_legacy_era_mirrors_backend`, `test_discover_does_not_force_legacy`, `test_provider_failure_is_visible`, `test_proxy_timeout_and_cancel`, `test_proxy_eof_shutdown`을 실제 transport 시험에 추가한다.

```powershell
& $candidatePython -m pytest tests/test_proxy.py tests/test_mcp_wire.py -q
```

**완료:** 최신 stdio client가 discovery 후 modern backend를 사용하고 legacy client도 독립적으로 동작한다. health/restart/timeout 사용자 정책이 보존된다.

## Task 4: 도구 계약·standalone·경량 실행 경로

**Files:** `cli.py`, 필요 시 `server.py`, `tests/test_server.py`, `tests/test_cli_paths.py`, `tests/test_mcp_wire.py`.

- [x] `get_tools()` 테스트를 새 공개 API에 맞춘다.

```python
tools = {tool.name: tool for tool in await application.mcp.list_tools()}
assert set(tools) == {"qdrant-find", "knowledge-index-status"}
assert "rerank" in tools["qdrant-find"].parameters["properties"]
```

- [x] async standalone 함수의 sync `mcp.run()`을 아래 호출로 교체하고 테스트 double도 async run으로 바꾼다. 실제 stdio subprocess에서 modern discovery와 legacy initialize를 각각 검증한다.

```python
await application.mcp.run_async(transport="stdio")
```

- [x] ML 생성·import가 기본 `serve`와 proxy에 필요하지 않게 CLI의 ML 의존성 import를 daemon/standalone/index 분기로 지연시킨다. 조사 환경의 CLI는 이미 fastembed/onnxruntime을 import하므로 proxy 모듈만 검사해서 경량 실행을 보장했다고 주장하지 않는다.
- [x] clean interpreter에서 `knowledge_mcp.proxy` 및 실제 기본 CLI 경로가 torch/onnxruntime/fastembed/sentence_transformers를 로딩하지 않는지 검증한다. 기본 client 여러 개가 모델 생성 횟수를 늘리지 않아야 한다.
- [x] 구·신 환경에서 같은 FakeStore의 실제 tools/list·tools/call 결과를 캡처한다. 필터 nullable, required query, `limit=8`, `include_private=False`, `rerank=False`, 모델 Literal, outputSchema, content, structuredContent, isError를 비교한다.
- [x] `test_find_wire_contract`, `test_empty_find_wire_contract`, `test_invalid_arguments_wire_error`, `test_status_wire_contract`, `test_standalone_modern_discovery`를 추가한다. Python snake_case와 wire camelCase를 구분한다. 소비자에게 필요한 결과가 달라진 경우에만 ToolResult 경계를 보정한다.

```powershell
& $candidatePython -m pytest tests/test_server.py tests/test_cli_paths.py tests/test_proxy.py tests/test_mcp_wire.py -q
```

**완료:** 최신 protocol 세 경로의 도구 계약·실제 실행·단일 모델 원칙이 유지된다.

## Task 5: 설정·client 구성·현재 문서와 아키텍처 갱신

**Files:** `.env.example`, `cli.py`, `tests/test_env_and_ignores.py`, README/사용자 가이드/현재 architecture HTML·PNG.

- [x] CLI의 daemon/server/FastMCP import 전에 기존 env loader를 실행하고 상속을 검증한다.

```python
from .config import Settings, _load_env_file

_load_env_file()
```

- [x] 예제·실사용 환경에 다음 값을 적용한다. 새 interpreter에서 `.env`를 읽은 FastMCP settings 값을 검사하고 daemon 자식에도 전달되는지 확인한다.

```dotenv
FASTMCP_CHECK_FOR_UPDATES=off
FASTMCP_TELEMETRY_MODE=off
FASTMCP_SHOW_SERVER_BANNER=false
FASTMCP_DEPRECATION_WARNINGS=true
```

- [x] global OTel provider/exporter가 없다면 원격 export는 발생하지 않으므로 dependency 존재 자체를 데이터 유출로 판단하지 않는다. SDK 자체 tracing과 FastMCP 설정은 구분하고 노트·질의의 원격 전송이 없음을 점검한다.
- [x] `FASTMCP_MCP_CAMELCASE_COMPAT=false`로 관련 테스트를 한 번 실행해 deprecated Python 접근이 남지 않았는지 검사한다.
- [x] 실제 client 구성 inventory에서 `/sse` URL을 사용하는 항목은 `/mcp`로 바꾸고 transport 설정도 Streamable HTTP로 갱신한다. stdio 명령은 유지하고 검증된 새 Python 경로로 전환한다. 사용자 앱의 숨겨진 설정을 추정해 변경하지 않는다.
- [x] README의 버전·공식 저장소 링크·설치 제약·extra 중단·modern 지원 버전을 갱신한다. timeout이 작업 완료/미완료를 확정하지 않음과 자동 sync 재시도 금지를 안내한다.
- [x] 현재 architecture HTML과 두 PNG를 Streamable HTTP `/mcp`·protocol proxy 구조로 갱신하고 시각적으로 확인한다. 날짜가 있는 과거 문서는 역사 기록으로 유지한다.

**완료:** 사용 안내와 실제 endpoint/protocol/client 설정이 일치한다.

## Task 6: 통합 검증과 운영 전환·롤백

**Files:** 위 변경 전체. **Evidence:** `.knowledge/migration/fastmcp4/` 로컬 기록.

- [x] API/구 transport 잔재 스캔과 dependency 검증을 실행한다. 활성 src/tests에서 sse_client·discovery 거절·구 endpoint·제거된 SDK wrapper가 남지 않아야 한다.

```powershell
rg -n 'get_tools\(|mcp\.shared\.session|JSONRPCMessage\(|sse_client|/sse|_make_method_not_found_response' src tests
& $candidatePython -m pip check
& $candidatePython -m pytest -q
```

- [x] 전체 테스트는 localhost Qdrant 1.19.1을 준비한 시험 환경에서 실행한다. 기존 통합 테스트 중 Qdrant 부재 시 자동 skip하지 않는 항목이 있다.
- [x] 임시 Vault·별도 runtime·시험 collection으로 실제 모델 index→한글 검색→rerank→증분 sync→status를 검증한다. rebuild를 사용하지 않는다.
- [x] 아래 protocol matrix를 실제 transport에서 통과시키고 front/backend version을 각각 기록한다. 최신 auto client가 legacy로 강등되면 실패다.

| 경로 | 최신 client | legacy client |
|---|---|---|
| daemon 직접 `/mcp` | discovery/modern 요청, 2026-07-28 | initialize 기반 협상 |
| 기본 `serve` stdio proxy | discovery 성공, backend도 modern | front/backend 모두 legacy |
| `serve --standalone` stdio | discovery 성공, modern tools/call | 기존 initialize 및 읽기 도구 |

- [x] 헤더 누락/불일치, unsupported version, 부재 method, JSON/SSE 응답, timeout/cancel, 동시 요청, daemon 재기동, stdio EOF를 검증한다. modern에는 ping·legacy GET 연결·Last-Event-ID 재생이 필수라는 가정을 넣지 않는다.
- [ ] 실제 구성된 Codex/Claude Code/Antigravity UI 재접속 후 협상 protocol을 기록한다. 설치 버전과 설정 command의 SDK 협상은 검증했다. client 자체가 legacy만 지원하면 compatibility 결과로 표시하며 modern 성공의 증거로 사용하지 않는다. 고정 modern SDK client가 자동화 검증을 담당한다.
- [x] 같은 머신·데이터·warmup·client 수로 검색/rerank p50/p95·startup·RSS/VRAM을 비교한다. 기준선 대비 20% 이상 증가하면 원인을 분석한 후 전환한다. 공식 proxy 추가 overhead도 이 비교에 포함한다.
- [x] 운영 전환 직전 baseline 환경·코드·client 설정, SQLite backup API로 일관된 백업, Qdrant 로컬 snapshot을 확보한다.
- [x] 구 client/proxy 종료 → 구 daemon stop 및 포트/PID 해제 → 새 daemon `/mcp` start → health/discovery/tools/search → 새 client 경로·URL로 전환 → 재연결 순서로 진행한다.
- [x] 아래 롤백을 실제 한 번 연습하고 다시 최신 경로를 확인한다. 구환경은 정상 사용과 다음 증분 sync까지 보존하고 제거는 별도 정리 작업으로 남긴다.

### 롤백

1. 새 client/proxy를 종료하고 새 daemon을 stop한다.
2. 구 checkout·구환경·구 client Python 경로 및 직접 URL `/sse` 설정으로 복귀한다.
3. 구환경 daemon을 시작하고 client를 재연결한다.
4. health·도구 목록·동일 공개 노트 검색·index 상태·collection point 수를 확인한다.
5. DB/벡터 포맷은 바뀌지 않았으므로 정상 롤백에서 Qdrant·SQLite·manifest를 삭제하거나 되감지 않는다. 손상이 실제 확인됐을 때만 별도 백업 복구를 수행한다.

**전환 중단/롤백 조건:** 최신 discovery 실패·강제 legacy 강등, protocol/header 불일치, 검색 필드 소실, private 필터 회귀, 모델 중복 생성, daemon 재시작 반복, 원격 노트/질의 전송, 기존 상태 읽기 실패.

**최종 완료:** 최신 protocol matrix, 데이터·검색 계약, lifecycle, 실사용 compatibility, 복구 게이트가 모두 통과한다. 미검증 영역이 있으면 전환을 보류하고 정확히 표시한다.

## 사용하지 않는 protocol 기능

Apps, tasks, sampling, roots, elicitation, MRTR, subscriptions를 이번에 모두 추가하지 않는다. 현재 도구가 요구하는 필수 protocol 동작·version/discovery·transport·metadata·오류·취소는 검증하고, 실제 제공하지 않는 capability는 광고하지 않는다. background task 패키지는 기존 `knowledge-index-sync`가 일반 async 도구이므로 필수 의존성이 아니다.

## 계획 자체 검토

- [x] 사용자 목표에 맞게 package·protocol·transport·proxy 전환을 이번 작업의 필수 범위로 묶었다.
- [x] discovery 강제 거절과 SDK의 정상 하위 호환 협상을 구분했다.
- [x] 구 HTTP+SSE와 최신 HTTP 응답 SSE를 구분했다.
- [x] 목표 릴리스에서 공식 proxy의 era mirroring과 실행 API를 확인했다.
- [x] 데이터 보존·모델 단일 로딩·timeout/cancel·설정 적용·복구를 반영했다.
- [x] 기본 HTTP/stdio/standalone의 최신 지원을 완료 조건으로 명시했다.
- [x] 이전 구환경 테스트 28개 통과와 새 환경 미검증을 구분했다.

## 실행 결과 및 계획 조정

[실행 결과](../../2026-09-29-fastmcp4-migration-results.md)를 확인한다. 구현은 별도 managed worktree의 .venv에서 수행하여 기존 checkout/환경을 보존했다. SSL context 재사용, 프로세스 시작 잠금, 취소 상태 기록, 비동기 sync 직렬화, 별도 900초 시작 제한을 실제 재현에 따라 추가했다. 현재 client 실행 경로인 worktree를 유지하며 main merge/push는 별도 통합 단계다. UI client 재연결은 아직 필요하다. 계획의 단계별 커밋은 완료 검증 후 의존성/구현/문서로 분리한다.
