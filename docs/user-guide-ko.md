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
   ├── bge-reranker-v2-m3-ko 리랭커 (CUDA/CPU 워밍업)
   └── FastMCP 도구 제공 (qdrant-find, knowledge-index-status, knowledge-index-sync)
         │
         ├──> [ 로컬 Qdrant 1.19.1 ] (127.0.0.1:6333, 6334)
         │       └── 벡터 임베딩, BM25 인덱스, 문서 청크 및 메타데이터
         │
         └──> [ SQLite State ] (Project/.knowledge/state.sqlite3)
                 └── 파일별 완료 기록, 인덱스 실행 로그, 질의 통계
```

- **로컬 Qdrant**: Windows 네이티브 1.19.1 바이너리를 사용합니다. 정상 실행 중인 로컬 서비스(127.0.0.1:6333)도 자동으로 감지하여 재사용합니다.
- **단일 HTTP 데몬**: FastMCP `4.0.10`과 MCP SDK `2.2.0`을 사용하며, 무거운 딥러닝 모델(`BGE-m3-ko`, 리랭커)을 1벌만 메모리에 상주시킵니다.
- **경량 Stdio protocol 프록시**: AI 에이전트(Codex 등)의 진입점으로, 무거운 ML 라이브러리를 로드하지 않습니다. 공식 FastMCP 프록시가 클라이언트의 protocol era를 그대로 반영하여 modern→modern, legacy→legacy로 연결합니다. 최신 `server/discover` 요청을 강제로 거절하거나 legacy로 강등시키지 않습니다.
- **로컬 보안 원칙**: 외부 Qdrant Cloud나 상용 임베딩 API를 사용하지 않으며, 모든 임베딩과 검색은 PC 내부에서 처리됩니다.

Streamable HTTP는 `/mcp` 하나에서 POST 요청과 JSON 또는 **요청별 SSE 응답**을 처리합니다. 이 SSE는 구 HTTP+SSE 전송의 지속 `/sse` 연결 및 별도 message endpoint와 다릅니다. MCP `2026-07-28` modern 요청은 discovery와 요청별 protocol metadata를 사용하며 `initialize`, protocol session ID, 별도 GET stream을 요구하지 않습니다. daemon 직접 HTTP, 기본 stdio proxy, standalone stdio 모두 최신 protocol을 지원하고, 구 클라이언트는 SDK의 정상 initialize 기반 협상으로 같은 `/mcp` 데몬을 이용합니다.

---

## 2. 빠른 시작 (설치 및 환경 설정)

Python `3.11` 이상을 사용합니다. 새 checkout에서 가상환경을 만든 뒤 설치하세요. `obsidian-knowledge-mcp`는 PEP 508 환경 마커(`sys_platform == 'win32'` vs `sys_platform != 'win32'`)를 사용하여 Windows(GPU 가속)와 Linux(CPU 전용) 의존성을 자동 분기합니다.

### 2.1 가상환경 생성 및 설치

#### Windows (CUDA 가속 권장)
Windows Python `3.13` CUDA 환경을 재현할 때는 다음 constraints를 사용합니다.

```powershell
python -m venv .venv
$candidatePython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $candidatePython -m pip install --upgrade pip
& $candidatePython -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& $candidatePython -m pip install -c constraints/fastmcp4-windows-py313.txt -e '.[dev]'
& $candidatePython -m pip check
```

이 constraints는 해당 Windows/GPU 환경의 재설치 기준이며 다른 플랫폼의 lockfile이 아닙니다. 다른 환경에서는 새 가상환경에 `python -m pip install -e .`로 설치하고 `python -m pip check`를 확인하세요. `qdrant-mcp` extra는 제거되었습니다. 구 `mcp-server-qdrant==0.8.1`이 FastMCP `2.7.0`과 Pydantic `<2.12.0`을 고정하므로 필요하면 별도의 FastMCP 2 환경에서 사용합니다.

#### Linux / WSL (CPU 모드)
Linux 및 WSL 환경에서는 불필요한 대용량 NVIDIA CUDA runtime wheel 다운로드를 방지하기 위해 PyTorch CPU 전용 인덱스를 사전 설치한 후 패키지를 설치합니다:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch CPU 빌드 사전 설치 (대용량 CUDA wheel 방지)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 패키지 설치 (pyproject.toml 마커에 의해 fastembed, onnxruntime CPU 버전 자동 설치)
pip install -e .
pip check
```

> [!NOTE]
> `pyproject.toml`에서 `sys_platform != 'win32'`일 경우 `fastembed-gpu`, `onnxruntime-gpu`, `nvidia-*` 패키지가 제외되고 경량 `fastembed`, `onnxruntime` CPU 휠이 설치됩니다.

기존 설치에서 전환할 때는 **별도 checkout/worktree에 새 `.venv`를 생성**합니다. 원래 checkout과 editable 가상환경 및 클라이언트 설정은 롤백용으로 그대로 보존하고, 원래 환경을 제자리 업그레이드하거나 가상환경을 복사하여 재사용하지 마세요. 프로토콜 전환에 모델 변경, 컬렉션 재생성, SQLite/Qdrant 포맷 변경은 필요하지 않습니다.

### 2.2 환경변수 설정

환경에 맞게 Vault 경로와 프로젝트 경로를 설정합니다:

- **Windows PowerShell**:
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

- **Linux / WSL (bash)**:
  ```bash
  # Vault 경로 및 프로젝트 경로 설정 (환경에 맞게 수정)
  # 또는 프로젝트 루트의 .env 파일에 설정할 수도 있습니다 (.env.example 참고).
  export KNOWLEDGE_VAULT_ROOT="/path/to/your/Obsidian"
  export KNOWLEDGE_PROJECT_ROOT="/path/to/obsidian-knowledge-mcp"

  export MCP="$KNOWLEDGE_PROJECT_ROOT/.venv/bin/knowledge-mcp"
  export PYTHON="$KNOWLEDGE_PROJECT_ROOT/.venv/bin/python"
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
정상 Qdrant가 없으면 설정된 네이티브 실행 파일로 시작하고, Vault와 인덱스를 비교하여 **추가·수정·삭제된 파일만 증분 동기화**합니다.
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

# 데몬 백그라운드 시작 (HTTP 먼저 오픈 → Qdrant 확인·모델 로딩·1회 동기화)
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
| `rerank` | bool | `true` | 한국어 Cross-Encoder(`bge-reranker-v2-m3-ko`) 적용 여부 |

리랭크 기본값은 `true`입니다. 명시적 `false`는 RRF 순서를 보존하며 검색 중 리랭커를 로드하지 않습니다. CUDA를 사용할 수 없으면 CPU를 선택하고, CUDA 모델 초기화가 실패하면 CPU 초기화를 한 번 시도합니다. CPU 리랭크 성공도 `rerank_applied=true`입니다.

초기화·추론·점수 검증 실패 시 기존 RRF 순서와 점수를 반환합니다. 각 결과의 `rerank_requested`, `rerank_applied`, `rerank_error`로 요청과 실제 적용을 구분합니다. 실패 코드는 각각 `reranker_init_failed`, `reranker_inference_failed`, `reranker_invalid_scores`이며 성공이나 명시적 false에는 오류 코드가 없습니다. 초기화 실패는 리랭커 인스턴스에 보존되므로 재초기화하려면 데몬을 재시작하세요. 추론·점수 실패는 다음 검색에서 다시 시도합니다. 점수는 `rerank_applied=true`일 때 CrossEncoder 원점수이고 그 외에는 RRF 점수입니다. 출처별 두 결과 제한과 최종 limit는 순위 결정 뒤 적용됩니다. 빈 결과는 리랭커를 로드하거나 적용하지 않습니다. 필드는 기존 결과별 텍스트와 구조화된 `{"result": [...]}` 양쪽 출력에 포함됩니다.

상태의 리랭커 `device`는 실제 `cuda`/`cpu`이고 초기화 전 또는 초기화 실패 시 `null`입니다. `loaded`와 `last_fallback`도 제공합니다. CPU 선택 코드는 `reranker_cuda_unavailable` 또는 `reranker_cuda_init_failed`로 RRF 폴백과 구분됩니다. 이후 추론이 성공하면 최근 추론 실패 코드는 디바이스 선택 코드(CUDA 성공 시 null)로 돌아갑니다. 이 상태는 Qdrant를 다시 조회하지 않고 매 상태 요청에서 갱신합니다.

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

HTTP/MCP는 Qdrant 확인, 모델 로딩, 최초 증분 동기화, 리랭커 warmup 전에 열립니다. `/health`의 HTTP 200은 서버가 요청을 받는다는 뜻이며 검색 준비 완료를 의미하지 않습니다. 응답의 `status`는 `starting`, `indexing`, `ready`, `error` 중 하나입니다. 초기화·색인 중에도 discovery, 도구 목록, 상태 조회는 사용할 수 있습니다. `knowledge-index-status`는 `state`, 모델별 `progress` 진행 수, `last_completed`, 안정적인 `last_error` 코드를 제공합니다. 모델별 포인트 수와 색인 결과는 초기화·동기화·warmup 후 갱신한 스냅샷입니다.

실제 색인 중에는 `qdrant-find`만 명시적인 “인덱싱 중, 잠시 후 재시도” 도구 오류를 반환합니다. 모델 준비 전에는 starting 또는 초기화 실패로 구분합니다. 색인이 끝나면 최근 동기화가 일부 실패했더라도 유효한 기존 컬렉션을 검색할 수 있습니다. Qdrant 연결 실패는 검색 오류로 반환되며 indexing으로 표시하지 않습니다. 선택적 warmup 실패는 `warmup_error`로 표시하고 기본 검색은 허용합니다.

초기·수동 동기화는 하나의 writer 대기열을 공유합니다. 정상 종료 시 관리 작업을 취소하고, 이미 시작된 파일·SQLite·모델 스레드 작업을 완료한 뒤 중단 상태를 기록하고 Qdrant 클라이언트를 사용하던 이벤트 루프에서 닫습니다. 실행 중인 네이티브 모델 작업 때문에 정상 종료가 늦어질 수 있습니다. `serve --standalone`도 같은 초기화 수명주기를 사용하며 읽기 도구 두 개를 유지합니다.

`KNOWLEDGE_DAEMON_START_TIMEOUT`의 기본 900초는 HTTP 서버가 열릴 때까지 기다리는 제한입니다. 모델·색인 준비 상태 및 일반 도구 호출 제한 15초와는 별개입니다.

---

## 6. 네이티브 Qdrant 운영 및 스토리지

일반 실행에는 서비스 설치가 필요하지 않습니다. Qdrant는 로컬 네이티브 프로세스로 실행되며, 공식 Qdrant 1.19.1 바이너리의 절대 경로를 지정하세요. PATH는 수동 CLI 편의용이며 무인 실행에는 절대 경로를 권장합니다.

```powershell
$env:KNOWLEDGE_QDRANT_EXECUTABLE = "C:/Tools/qdrant-1.19.1/qdrant.exe"
$env:KNOWLEDGE_QDRANT_NATIVE_STORAGE = "C:/Users/YourName/qdrant-native"
```

네이티브 기본 저장소는 사용자 홈의 `%USERPROFILE%/.knowledge-qdrant/<프로젝트 경로 해시>`입니다. 짧은 경로는 Windows Gridstore의 긴 경로 실패를 방지합니다. 빈 폴더에만 `.knowledge-native-owner.json`을 생성하며, 표식 없는 임의의 기존 데이터 폴더는 거부합니다. 기본 경로 외에 별도 디렉터리를 사용하려면 `KNOWLEDGE_QDRANT_NATIVE_STORAGE` 환경 변수를 지정하세요. 정상 응답하는 localhost Qdrant 인스턴스(127.0.0.1:6333)가 이미 실행 중이면 자동으로 감지하여 우선 재사용합니다. Qdrant 6333/6334와 MCP 8765 기본 포트는 유지됩니다.

데이터 백업, snapshot 복원, 검증 및 선택적 서비스 등록에 대한 자세한 내용은 [Qdrant 상태 진단 및 색인 모니터링 가이드](qdrant-status-and-diagnostics.md#native-qdrant-migration)를 참조하세요. 복원이나 롤백이 필요한 경우 백업 데이터를 새로운 네이티브 저장소에 직접 복원합니다.

---

## 7. 관련 상세 문서 안내

### 로컬 기록과 개인정보 보존 정책

`state.sqlite3`의 질의 기록은 시각, 지연 시간, 결과 수, 리랭크 요청/적용 여부와 안정적인 실패 코드만 저장합니다. 검색어, 필터, 클라이언트명, 결과 ID, 결과 경로와 본문은 저장하지 않습니다. 성공한 검색은 해당 결과의 실제 `rerank_applied`를 기록합니다. 명시적 `false` 요청과 빈 결과는 적용 여부를 false로 기록하며, 일반 검색 실패의 적용 여부는 `NULL`로 남습니다. 요청값만으로 성공을 추정하지 않습니다.

질의 지표는 기본 30일, 최대 10,000행을 보존합니다. `.env`의 `KNOWLEDGE_QUERY_RETENTION_DAYS`, `KNOWLEDGE_QUERY_MAX_ROWS`로 조절하며 0이나 잘못된 값은 기본값으로 돌아갑니다. 색인 이력은 기본 30일, 최대 1,000행을 보존합니다(`KNOWLEDGE_INDEX_RETENTION_DAYS`, `KNOWLEDGE_INDEX_MAX_ROWS`). 각 컬렉션의 최신 색인 상태는 이력 상한과 별도로 항상 유지합니다. 시작, 운영 기록 저장, 상태 조회 때 정리합니다. 작은 삭제는 SQLite 페이지를 재사용하고 큰 삭제는 `VACUUM`으로 줄입니다. 파일 manifest는 영구 색인 상태로 보존되어 Vault 크기에 따라 커질 수 있습니다.

기존 DB의 최초 초기화는 런타임 디렉터리에 `state.pre-privacy.sqlite3` 일회성 백업을 만든 뒤 원문 질의/결과 테이블과 과거 예외 텍스트를 정리하고 `VACUUM`을 수행합니다. 스키마 변경 실패는 롤백되며 정리 중단 시 다음 시작에 재시도합니다. 백업에는 기존 민감 정보가 남아 있고 자동 삭제하거나 덮어쓰지 않습니다. 마이그레이션 시 콘솔 경고가 백업 위치를 알려주며 백그라운드 실행에서는 이 문서의 고정 파일명으로 확인할 수 있습니다. 데몬을 중지하고 상태를 검증한 뒤 복구가 필요 없으면 직접 삭제하세요. 복구하려면 모든 데몬/색인기를 중지하고 현재 DB의 사본을 보존한 다음 백업을 `state.sqlite3`로 복사하고 이전 체크아웃을 실행합니다. 이 버전으로 복구 DB를 실행하면 다시 마이그레이션됩니다. 외부 백업과 파일시스템 복구 사본은 정리 대상에 포함되지 않습니다.

`daemon.log`는 실행 중에도 기본 1 MiB에서 회전하며 `daemon.log.1`~`.3` 세 파일을 보관합니다. `KNOWLEDGE_DAEMON_LOG_MAX_BYTES`, `KNOWLEDGE_DAEMON_LOG_BACKUP_COUNT`로 상한을 조절합니다. 파일에는 시각, 심각도, 안정적인 이벤트 코드만 저장하며 라이브러리 메시지, stdout/stderr 원문, traceback은 버립니다. 포그라운드 실행에서는 콘솔 진단을 확인할 수 있습니다. 로그 초기화 전 실패는 종료 코드만 확인 가능할 수 있습니다. 기존 로그에는 과거 원문이 남아 있을 수 있으므로 필요 시 데몬을 중지한 뒤 직접 정리하세요.

- [Qdrant 상태 키워드 및 진단 가이드 (qdrant-status-and-diagnostics.md)](qdrant-status-and-diagnostics.md): Qdrant REST API, SQLite 상태 테이블 분석, 미색인 파일 추적 및 에러 코드 상세.
- [문서 메타데이터 및 설정 가이드 (metadata-and-configuration.md)](metadata-and-configuration.md): Frontmatter 문법, sidecar yaml, ignore 설정, security 레벨.
- [데몬 아키텍처 및 리소스 상한 설계 문서 (2026-09-20-daemon-architecture-and-resource-limits.md)](2026-09-20-daemon-architecture-and-resource-limits.md): 메모리 프리징 해결 과정과 5중 리소스 가드레일 분석.
- [대화형 아키텍처 다이어그램 (obsidian-knowledge-architecture.html)](obsidian-knowledge-architecture.html): 브라우저에서 인터랙티브하게 확인 가능한 시스템 다이어그램.
