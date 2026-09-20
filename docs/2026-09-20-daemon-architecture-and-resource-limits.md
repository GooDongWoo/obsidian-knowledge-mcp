# 세션 문서화: 램 폭증/프리징 원인 분석 및 단일 데몬·리소스 상한 아키텍처 구현

**문서 작성일**: 2026-09-20  
**프로젝트**: `obsidian-knowledge-mcp`  
**관련 커밋**: [`3ab2c77`](https://github.com/GooDongWoo/obsidian-knowledge-mcp/commit/3ab2c77) (`feat: implement single SSE daemon, stdio proxy, and resource limits`)

---

## 1. 문제 상황 및 배경 (Background & Symptoms)

- **보고된 증상**: 사용자가 Obsidian Vault에 문서 몇 개를 추가한 직후, 시스템 램(RAM) 사용량이 급격히 치솟고 윈도우 OS가 멈추는(Freezing) 현상 발생.
- **실행 환경**:
  - OS: Windows 11
  - GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
  - RAM: 32GB
  - Python: 3.13 (PyTorch 2.11.0+cu128, ONNX Runtime GPU 1.26.0)
  - Backend: Docker Desktop (WSL2, Qdrant v1.19.1)
  - MCP Clients: Codex, Claude Code, Antigravity 등 다중 에이전트 환경

---

## 2. 근본 원인 분석 (Root Cause Analysis)

코드베이스와 시스템 동작 메커니즘을 심층 분석한 결과, 다음 5가지 원인이 복합적으로 작용하여 발생한 문제임을 규명했습니다.

### ① Stdio 기반 MCP 프로세스 중복 및 모델 이중 로드
- MCP의 `stdio` 방식 특성상 Codex, Claude Code, Antigravity 등의 클라이언트가 실행될 때마다 각각 독립된 `python.exe (knowledge-mcp)` 자식 프로세스를 새로 생성했습니다.
- 프로세스마다 대형 임베딩 모델 2종(`intfloat/multilingual-e5-large` ~2.2GB, `dragonkue/BGE-m3-ko` ~2.2GB)을 메모리에 중복 로드하여, **프로세스 1개당 최소 4.5GB 이상의 메모리/VRAM을 각자 점유**했습니다.
- 3개 클라이언트 실행 시 파이썬 프로세스만으로 13.5GB+가 잠식되었습니다.

### ② VRAM 한계 초과 및 Windows 공유 GPU 메모리 페이징 (프리징 유발)
- 두 임베딩 모델이 상주한 상태에서 대용량 청크를 한꺼번에 GPU로 인퍼런스할 때 RTX 4070 Ti의 12GB VRAM 한계를 넘나들었습니다.
- Windows WDDM(Windows Display Driver Model)은 VRAM이 부족해지면 **초과분을 시스템 RAM으로 강제 페이징(스왑)**합니다.
- 이때 PCIe 버스 대역폭이 100% 포화되면서 **바탕화면 창 관리자(DWM.exe)가 멈추고 마우스/키보드 입력이 일시적으로 정지되는 프리징**이 발생했습니다.

### ③ PyTorch & ONNX Runtime의 비제한 CPU 스레드 풀가동 (CPU 100% 원인)
- `torch.set_num_threads` 상한이 지정되지 않아, 임베딩 및 토크나이징 연산 시 PC의 **모든 논리 코어(16~24스레드)를 100% 점유**하여 CPU 사용량이 99%~100%로 치솟았습니다.

### ④ 임베딩 배치 및 Qdrant Upsert 상한 부재
- **FastEmbed**: 기본 배치 크기가 **256**으로 설정되어 있어 거대한 중간 텐서 메모리를 소모했습니다.
- **Qdrant Upsert**: 문서 내 모든 청크(본문 텍스트, 1024차원 dense 벡터, BM25 원문 텍스트)를 분할 없이 **단 1회의 거대한 HTTP 요청**으로 Qdrant에 전송했습니다. 이로 인해 Docker WSL2 가상머신(`vmmemWSL`)의 메모리가 순간적으로 급격히 팽창했습니다.

### ⑤ 파일 크기 및 청크 개수 상한 부재
- 파일 수집(`discover_sources`) 및 파싱(`parse_source`) 시 파일 크기 검증이 없어 수십~수백 MB의 파일도 그대로 메모리에 적재되는 취약점이 있었습니다.

---

## 3. 아키텍처 전환 및 해결 방안 (Architecture & Solutions)

기존 문제를 근본적으로 해결하기 위해 **단일 SSE/HTTP 데몬(Daemon) + 경량 Stdio 프록시(Proxy)** 구조로 아키텍처를 전면 개편하고, 5중 리소스 상한을 적용했습니다.

```
[ 기존 구조: 클라이언트마다 개별 프로세스 및 모델 중복 로드 (13.5GB+) ]
Codex --------> Python Process A (E5 2.2GB + BGE 2.2GB)
Claude Code --> Python Process B (E5 2.2GB + BGE 2.2GB)
Antigravity --> Python Process C (E5 2.2GB + BGE 2.2GB)

[ 개선 구조: 단일 백그라운드 데몬 + 초경량 프록시 (~4.6GB 고정) ]
Codex --------> Stdio Proxy A (~15MB) --\
Claude Code --> Stdio Proxy B (~15MB) ----+--> [ Single SSE Daemon (Port 8765) ]
Antigravity --> Stdio Proxy C (~15MB) --/       (E5 + BGE 모델 1벌만 상주: 4.5GB)
                                                       |
                                                       v
                                            Docker Qdrant (Mem Limit: 4G)
```

---

## 4. 상세 변경 내역 (Changes Made)

### Component: Daemon & Proxy Architecture

| 파일 | 변경 구분 | 주요 내용 |
| :--- | :--- | :--- |
| `src/knowledge_mcp/daemon.py` | **[NEW]** | - FastMCP를 `transport="sse"`(기본 포트 8765)로 실행하는 단일 백그라운드 서버.<br>- 프로세스 수명 주기(`run_daemon`, `start_daemon_process`, `stop_daemon_process`, `is_daemon_running`, `get_daemon_pid`) 구현.<br>- 데몬 시작 시 1회만 인덱싱(`sync_all()`) 수행하여 클라이언트 접속 지연 해소.<br>- 도구 등록: `qdrant-find`, `knowledge-index-status`, 신규 동기화 관리 도구 `knowledge-index-sync`.<br>- `/health` 헬스체크 및 `daemon.pid` 추적. |
| `src/knowledge_mcp/proxy.py` | **[NEW]** | - 무거운 ML 라이브러리(`torch`, `onnxruntime`, `fastembed`)를 **전혀 import하지 않는 초경량 프록시** (메모리 약 15MB).<br>- `mcp.client.sse.sse_client`와 `mcp.server.stdio.stdio_server`를 사용하여 Stdio ↔ SSE 간 양방향 JSON-RPC 중계. |
| `src/knowledge_mcp/cli.py` | **[MODIFY]** | - `knowledge-mcp daemon {start\|stop\|status\|run}` 서브커맨드 추가.<br>- `knowledge-mcp serve`의 기본 동작을 **데몬 자동 기동 + Stdio 프록시 연결**로 전환 (기존 설정 100% 무수정 호환).<br>- `--standalone` 플래그 추가 (기존 인프로세스 실행 지원). |

### Component: Resource Limits & Stability

| 파일 | 변경 구분 | 주요 내용 |
| :--- | :--- | :--- |
| `src/knowledge_mcp/embeddings.py` | **[MODIFY]** | - **배치 크기 축소**: FastEmbed `passage_embed(..., batch_size=32)` (기존 256 → 32).<br>- **SentenceTransformer 분할**: `self.model.encode(..., batch_size=32)`.<br>- **CPU 스레드 제한**: `torch.set_num_threads(min(4, os.cpu_count() or 4))` 설정으로 CPU 100% 점유 방지. |
| `src/knowledge_mcp/qdrant_store.py` | **[MODIFY]** | - `replace_generation()`에서 `points`를 **64개 단위로 분할하여 `upsert`** 수행. WSL2 Docker 메모리 팽창 억제. |
| `src/knowledge_mcp/documents.py` | **[MODIFY]** | - `MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024` (30MB) 상한 설정.<br>- 30MB 초과 파일은 `discover_sources`에서 제외하고, `parse_source`에서 `skipped_large_file`로 본문 로드 없이 스킵. |
| `src/knowledge_mcp/indexer.py` | **[MODIFY]** | - `skipped_large_file` 상태를 정상 스킵(`summary.skipped += 1`)으로 처리하도록 추가. |
| `docker-compose.yml` | **[MODIFY]** | - Qdrant 서비스에 `deploy.resources.limits.memory: 4G` 메모리 상한 지정. |

---

## 5. 검증 결과 (Verification & Results)

1. **자동화 테스트 검증**:
   - 신규 테스트 스위트 작성:
     - `tests/test_daemon.py`: 데몬 라이프사이클, PID 관리, 헬스체크, 도구 등록 검증.
     - `tests/test_proxy.py`: 경량 import 검증(torch/onnx 미포함 확인) 및 Stdio ↔ SSE 중계 검증.
     - `tests/test_resource_limits.py`: 30MB 파일 스킵, Qdrant 64개 분할 업서트, PyTorch 스레드 상한 검증.
   - 전체 테스트 실행: `.venv\Scripts\pytest tests/` 실행 결과 **113개 테스트 전원 통과 (100% Pass)**.
2. **시스템 자원 사용량 비교**:
   - **RAM 점유율**: 다중 클라이언트(3개 기준) 구동 시 **기존 13.5GB+ → 약 4.6GB**로 대폭 절감.
   - **클라이언트 접속 속도**: 클라이언트 연결 시 매번 수행되던 무거운 인덱싱 검사가 제거되어 **즉각 연결(Instant Connect)** 실현.
   - **시스템 안정성**: 배치 크기 축소 및 CPU 스레드 제한으로 인해 문서 추가 시 CPU 100% 치솟음 및 VRAM 초과로 인한 윈도우 프리징 현상 원천 차단.

---

## 6. 운영 및 사용 가이드 (Operation Guide)

### ① 일반 사용 (기존 설정 그대로 사용)
- 기존 Codex(`config.toml`), Claude Code, Antigravity 설정(`knowledge-mcp serve --client ...`)을 **수정할 필요가 없습니다**.
- 클라이언트를 실행하면 백그라운드에 데몬이 자동으로 기동되고, 클라이언트는 초경량 프록시(~15MB)로 데몬에 자동 연결됩니다.

### ② 데몬 수동 제어 명령어
PowerShell에서 다음과 같이 데몬을 수동으로 관리할 수 있습니다:

```powershell
# 데몬 상태 확인
& .\.venv\Scripts\knowledge-mcp.exe daemon status

# 데몬 백그라운드 기동
& .\.venv\Scripts\knowledge-mcp.exe daemon start

# 데몬 종료 (모델 메모리 완전 언로드)
& .\.venv\Scripts\knowledge-mcp.exe daemon stop

# 데몬 포그라운드 실행 (로그 직접 확인용)
& .\.venv\Scripts\knowledge-mcp.exe daemon run
```

### ③ 신규 문서 추가 시 인덱싱 동기화
- 데몬이 실행 중일 때 새 문서를 추가한 경우, 다음 두 가지 방법으로 즉시 반영할 수 있습니다:
  1. **에이전트 도구 호출**: 대화창에서 `knowledge-index-sync` 도구를 호출하도록 요청.
  2. **수동 명령어 실행**: 터미널에서 `& .\.venv\Scripts\knowledge-mcp.exe index` 실행.
