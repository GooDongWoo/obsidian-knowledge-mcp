# Obsidian Knowledge MCP 한국어 운영 가이드

이 문서는 `obsidian-knowledge-mcp`의 설치, 데몬 관리, 동기화, 검색, 그리고 일상 운영을 위한 상세 한국어 매뉴얼입니다.

---

## 1. 전체 아키텍처 개요

```text
[ Obsidian Vault 원본 문서 ] (.md, .txt, .pdf)
         │
         ▼
[ MCP 클라이언트 ] (Codex, Claude Code, Antigravity 등)
         │  (stdio)
         ▼
[ Stdio Protocol Proxy ]  <-- 클라이언트마다 공식 FastMCP 프록시 실행
         │  (Streamable HTTP: http://127.0.0.1:8765/mcp)
         ▼
[ Single HTTP Daemon (단일 백그라운드 프로세스, /health) ]
   ├── BGE-m3-ko 임베딩 (GPU / CUDA)
   ├── bge-reranker-v2-m3-ko 리랭커 (CPU 워밍업)
   └── FastMCP 도구 제공 (qdrant-find, knowledge-index-status, knowledge-index-sync)
         │
         ├──> [ Docker Qdrant ] (127.0.0.1:6333, 6334)
         │       └── 벡터 임베딩, BM25 인덱스, 문서 청크 및 메타데이터
         │
         └──> [ SQLite State ] (Project/.knowledge/state.sqlite3)
                 └── 파일별 완료 기록, 인덱스 실행 로그, 질의 통계
```

- **Qdrant 컨테이너**: Docker Desktop(WSL2) 환경에서 실행되며, 순수 벡터 및 BM25 저장소 역할을 담당합니다.
- **단일 HTTP 데몬**: FastMCP `4.0.10`과 MCP SDK `2.2.0`을 사용하며, 무거운 딥러닝 모델(`BGE-m3-ko`, 리랭커)을 1벌만 메모리에 상주시킵니다.
- **경량 Stdio protocol 프록시**: AI 에이전트(Codex 등)의 진입점으로, 무거운 ML 라이브러리를 로드하지 않습니다. 공식 FastMCP 프록시가 클라이언트의 protocol era를 그대로 반영하여 modern→modern, legacy→legacy로 연결합니다. 최신 `server/discover` 요청을 강제로 거절하거나 legacy로 강등시키지 않습니다.
- **로컬 보안 원칙**: 외부 Qdrant Cloud나 상용 임베딩 API를 사용하지 않으며, 모든 임베딩과 검색은 PC 내부에서 처리됩니다.

Streamable HTTP는 `/mcp` 하나에서 POST 요청과 JSON 또는 **요청별 SSE 응답**을 처리합니다. 이 SSE는 구 HTTP+SSE 전송의 지속 `/sse` 연결 및 별도 message endpoint와 다릅니다. MCP `2026-07-28` modern 요청은 discovery와 요청별 protocol metadata를 사용하며 `initialize`, protocol session ID, 별도 GET stream을 요구하지 않습니다. daemon 직접 HTTP, 기본 stdio proxy, standalone stdio 모두 최신 protocol을 지원하고, 구 클라이언트는 SDK의 정상 initialize 기반 협상으로 같은 `/mcp` 데몬을 이용합니다.

---

## 2. 빠른 시작 (PowerShell 설정)

Python `3.11` 이상을 사용합니다. 새 checkout에서 가상환경을 만든 뒤 설치하세요. Windows Python `3.13` CUDA 환경을 재현할 때는 다음 constraints를 사용합니다.

```powershell
python -m venv .venv
$candidatePython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $candidatePython -m pip install --upgrade pip
& $candidatePython -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& $candidatePython -m pip install -c constraints/fastmcp4-windows-py313.txt -e '.[dev]'
& $candidatePython -m pip check
```

이 constraints는 해당 Windows/GPU 환경의 재설치 기준이며 다른 플랫폼의 lockfile이 아닙니다. 다른 환경에서는 새 가상환경에 `python -m pip install -e .`로 설치하고 `python -m pip check`를 확인하세요. `qdrant-mcp` extra는 제거되었습니다. 구 `mcp-server-qdrant==0.8.1`이 FastMCP `2.7.0`과 Pydantic `<2.12.0`을 고정하므로 필요하면 별도의 FastMCP 2 환경에서 사용합니다.

기존 설치에서 전환할 때는 **별도 checkout/worktree에 새 `.venv`를 생성**합니다. 원래 checkout과 editable 가상환경 및 클라이언트 설정은 롤백용으로 그대로 보존하고, 원래 환경을 제자리 업그레이드하거나 가상환경을 복사하여 재사용하지 마세요. 프로토콜 전환에 모델 변경, 컬렉션 재생성, SQLite/Qdrant 포맷 변경은 필요하지 않습니다.

Windows PowerShell에서 환경변수를 설정합니다. 이후 명령은 **동일한 PowerShell 세션**에서 실행합니다.

```powershell
# Vault 경로 및 프로젝트 경로 설정 (환경에 맞게 수정)
# 또는 프로젝트 루트의 .env 파일에 설정할 수도 있습니다 (.env.example 참고).
$env:KNOWLEDGE_VAULT_ROOT = 'C:\Path\To\Your\Obsidian'
$env:KNOWLEDGE_PROJECT_ROOT = 'C:\Path\To\obsidian-knowledge-mcp'

$project = $env:KNOWLEDGE_PROJECT_ROOT
$vault = $env:KNOWLEDGE_VAULT_ROOT
$mcp = Join-Path $project '.venv\Scripts\knowledge-mcp.exe'
$python = Join-Path $project '.venv\Scripts\python.exe'
```

프로젝트 `.env`는 FastMCP import보다 먼저 읽히며 데몬 자식 프로세스에도 설정이 상속됩니다. `.env.example`의 기본값을 사용하고 데몬은 localhost에 바인딩합니다.

```dotenv
FASTMCP_CHECK_FOR_UPDATES=off
FASTMCP_TELEMETRY_MODE=off
FASTMCP_SHOW_SERVER_BANNER=false
FASTMCP_DEPRECATION_WARNINGS=true
```

프로젝트는 global OpenTelemetry exporter를 구성하지 않습니다. SDK tracing 의존성의 설치 자체가 원격 export를 의미하지 않으며, 노트 임베딩과 검색은 로컬에서 처리됩니다.

---

## 3. CLI 명령어 및 데몬 관리

`knowledge-mcp` CLI는 인덱싱, 서버 실행, 그리고 백그라운드 데몬 제어를 위한 하위 명령어를 제공합니다.

### 3.1 인덱싱 및 동기화

#### 증분 인덱싱 (`index`)
Qdrant 컨테이너가 꺼져 있으면 자동으로 시작하고, Vault와 인덱스를 비교하여 **추가·수정·삭제된 파일만 증분 동기화**합니다.
```powershell
& $mcp index --client codex
```
- 결과 요약(`added`, `changed`, `deleted`, `unchanged`, `skipped`, `failed`, `error_codes`)을 `stderr`에 출력합니다.
- 오류가 발생하면 종료 코드가 1이며, 다시 실행하면 실패한 파일만 재시도합니다.

#### 전체 재색인 (`rebuild`)
기존 BGE 컬렉션과 파일별 완료 기록을 삭제하고 처음부터 다시 색인합니다.
```powershell
& $mcp rebuild --client codex
```
> [!WARNING]
> 재색인이 완료될 때까지 검색 도구를 사용할 수 없습니다. 실행 전 기존 검색 프로세스를 종료하세요. 질의 로그는 보존됩니다.

---

### 3.2 백그라운드 데몬 제어 (`daemon`)

여러 AI 에이전트가 동시에 접근할 수 있도록 단일 백그라운드 서버를 관리합니다.

```powershell
# 데몬 상태 및 PID 확인
& $mcp daemon status

# 데몬 백그라운드 시작 (Qdrant 자동 확인 + 1회 동기화 + /mcp HTTP 서버 오픈)
& $mcp daemon start

# 데몬 중지 (메모리 완전 해제)
& $mcp daemon stop

# 데몬 포그라운드 실행 (디버깅 및 실시간 로그 확인용)
& $mcp daemon run --port 8765
```

MCP endpoint는 `http://127.0.0.1:8765/mcp`, 프로세스 health endpoint는 `http://127.0.0.1:8765/health`입니다. `/health` 성공만으로 protocol 동작을 확인할 수는 없으므로 discovery, 도구 목록, 검색도 확인하세요.

---

### 3.3 MCP 서버 실행 (`serve`)

```powershell
& $mcp serve --client codex
```
- 기본적으로 **백그라운드 데몬이 켜져 있는지 확인하고, 없으면 자동 기동한 후 Stdio 프록시로 연결**됩니다.
- `--standalone` 플래그를 주면 데몬 없이 단독 stdio 프로세스로 실행할 수 있으며 MCP `2026-07-28`과 legacy 협상을 지원합니다. 이 경로는 자체 모델을 로드하므로 여러 클라이언트가 모델을 공유하려면 기본 `serve`를 사용하세요. standalone은 읽기 도구 `qdrant-find`, `knowledge-index-status` 두 개를 제공합니다.

---

## 4. MCP 클라이언트 등록 가이드

각 에이전트 도구 설정 파일에 `obsidian-knowledge-mcp`를 등록합니다.

### 4.1 Codex 등록 (`~/.codex/config.toml`)
```toml
[mcp_servers.obsidian_knowledge]
command = "C:\\Path\\To\\obsidian-knowledge-mcp\\.venv\\Scripts\\knowledge-mcp.exe"
args = ["serve", "--client", "codex"]
env = { KNOWLEDGE_VAULT_ROOT = "C:\\Path\\To\\Your\\Obsidian", KNOWLEDGE_PROJECT_ROOT = "C:\\Path\\To\\obsidian-knowledge-mcp" }
startup_timeout_sec = 900
```
- 등록 상태 확인: `codex mcp get obsidian_knowledge`

### 4.2 Claude Code 등록
`claude_desktop_config.json` 또는 Claude Code 설정에 추가:
```json
{
  "mcpServers": {
    "obsidian_knowledge": {
      "command": "C:\\Path\\To\\obsidian-knowledge-mcp\\.venv\\Scripts\\knowledge-mcp.exe",
      "args": ["serve", "--client", "claude-code"],
      "env": {
        "KNOWLEDGE_VAULT_ROOT": "C:\\Path\\To\\Your\\Obsidian",
        "KNOWLEDGE_PROJECT_ROOT": "C:\\Path\\To\\obsidian-knowledge-mcp"
      }
    }
  }
}
```

### 4.3 Antigravity 등록
`~/.gemini/antigravity/mcp_config.json` 또는 작업공간 MCP 설정에 `--client antigravity`로 등록합니다.

### 4.4 기존 설정 전환 및 롤백

stdio 등록은 `serve` 명령을 유지하면서 실행 파일 경로를 **새 checkout의 `.venv\Scripts\knowledge-mcp.exe`**로 바꿉니다. 직접 HTTP 등록은 기존 `/sse` URL을 `http://127.0.0.1:8765/mcp`로 바꾸고 transport를 Streamable HTTP로 선택합니다. 구 클라이언트 호환성도 새 `/mcp`에서 제공됩니다.

1. 구 클라이언트와 프록시를 종료하고 구 환경의 데몬을 중지합니다.
2. 기존 포트와 PID가 해제된 뒤 새 환경의 데몬을 시작합니다. 동일 runtime directory/포트에서 구·신 데몬을 함께 실행하지 마세요.
3. health, discovery, 도구 목록, 공개 문서 검색 및 인덱스 상태를 확인한 뒤 새 경로로 클라이언트를 재연결합니다.
4. 문제가 있으면 새 클라이언트/프록시와 데몬을 중지하고 보존한 원래 checkout·가상환경·클라이언트 설정으로 복귀합니다. 구 직접 HTTP 설정은 `/sse`로 복구합니다.
5. 롤백 후 health, 도구 목록, 동일 검색과 인덱스 상태를 확인합니다. 정상 롤백에는 Qdrant/SQLite/manifest 삭제나 재색인이 필요하지 않습니다.

---

## 5. 제공되는 MCP 도구 및 사용법

### 5.1 `qdrant-find` (지식 검색)
Obsidian Vault에서 밀집 벡터(`BGE-m3-ko`)와 희소 BM25를 결합한 하이브리드 검색을 수행합니다.

| 매개변수 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `query` | string | (필수) | 검색 질의문 또는 키워드 |
| `document_type` | list[str] | None | 문서 유형 필터 (예: `["diary", "experience", "paper"]`) |
| `file_type` | list[str] | None | 파일 확장자 필터 (`["md", "txt", "pdf"]`) |
| `created_from` / `created_to` | string | None | 생성일 범위 (`YYYY-MM-DD`) |
| `modified_from` / `modified_to` | string | None | 수정일 범위 (`YYYY-MM-DD`) |
| `include_private` | bool | `false` | `true` 설정 시 비공개 문서 포함 |
| `limit` | int | `8` | 반환할 청크 수 (1~20) |
| `embedding_model` | string | `dragonkue/BGE-m3-ko` | 사용할 임베딩 모델 인덱스 |
| `rerank` | bool | `false` | 한국어 Cross-Encoder(`bge-reranker-v2-m3-ko`) 적용 여부 |

**에이전트 대화창 호출 예시**:
> "obsidian_knowledge의 qdrant-find로 'CUDA 가속' 관련 문서를 찾아줘. 출처 경로와 줄 번호도 포함해줘."  
> "qdrant-find에서 query='논문 아이디어', document_type=['research'], rerank=true, limit=5로 찾아줘."

---

### 5.2 `knowledge-index-status` (인덱스 상태 확인)
컬렉션별 색인 상태(`completed`, `partial`), 등록된 청크 수(`point_count`), 최근 실행의 추가/수정/삭제 건수 및 오류 코드를 반환합니다.

---

### 5.3 `knowledge-index-sync` (즉시 동기화 트리거)
새 문서를 Vault에 추가한 후 데몬을 재시작하지 않고 에이전트 대화창에서 즉시 증분 인덱싱을 수행할 때 사용합니다.
- `rebuild=false` (기본값): 변경분만 증분 동기화.
- `rebuild=true`: 컬렉션 재생성 및 전체 재색인.

프록시의 도구 호출 제한 시간은 기본 15초이며 `KNOWLEDGE_PROXY_TIMEOUT`으로 조절합니다. timeout이나 취소 응답만으로 동기화의 완료 여부를 알 수 없습니다. SDK 취소가 작업을 중단할 수 있고 일부 파일만 처리된 상태일 수도 있습니다. 프록시는 도구 호출을 자동 재전송하지 않습니다. `knowledge-index-status`와 데몬 로그로 진행 상태를 확인한 뒤 증분 동기화 재시도 여부를 판단하세요. timeout 이후 sync/rebuild를 자동으로 재시도하지 마세요.

취소된 색인은 `partial`과 `sync_cancelled`를 기록하며 다음 일반 증분 동기화에서 manifest와 벡터 세대를 복구합니다. 여러 클라이언트의 데몬 시작은 프로세스 간 잠금으로 직렬화하고, 동시 sync 요청은 파일 잠금을 잡기 전에 비동기로 대기합니다. 시작 제한 시간을 넘긴 프로세스는 종료한 뒤 시작 잠금을 해제합니다. 프록시는 인증서 검증용 SSL context만 재사용하며 MCP 세션은 SDK가 요청별로 생성합니다.

모델 로딩·최초 증분 동기화·리랭커 warmup에는 별도 시작 제한 900초를 적용합니다(`KNOWLEDGE_DAEMON_START_TIMEOUT`). MCP 클라이언트의 시작 제한도 이 값 이상으로 설정하세요. 일반 도구 호출 제한 15초와는 별개입니다.

---

## 6. Docker Qdrant 단독 제어

Qdrant 컨테이너만 수동으로 시작하거나 확인할 때 사용합니다:

```powershell
# 컨테이너 상태 확인
docker ps --filter "name=obsidian-knowledge-mcp-qdrant-1"

# Docker Compose 수동 기동
$env:KNOWLEDGE_QDRANT_STORAGE = (Join-Path $project '.knowledge\qdrant').Replace('\', '/')
docker compose -f (Join-Path $project 'docker-compose.yml') up -d
```

---

## 7. 관련 상세 문서 안내

- [Qdrant 상태 키워드 및 진단 가이드 (qdrant-status-and-diagnostics.md)](qdrant-status-and-diagnostics.md): Qdrant REST API, SQLite 상태 테이블 분석, 미색인 파일 추적 및 에러 코드 상세.
- [문서 메타데이터 및 설정 가이드 (metadata-and-configuration.md)](metadata-and-configuration.md): Frontmatter 문법, sidecar yaml, ignore 설정, security 레벨.
- [데몬 아키텍처 및 리소스 상한 설계 문서 (2026-09-20-daemon-architecture-and-resource-limits.md)](2026-09-20-daemon-architecture-and-resource-limits.md): 메모리 프리징 해결 과정과 5중 리소스 가드레일 분석.
- [대화형 아키텍처 다이어그램 (obsidian-knowledge-architecture.html)](obsidian-knowledge-architecture.html): 브라우저에서 인터랙티브하게 확인 가능한 시스템 다이어그램.
