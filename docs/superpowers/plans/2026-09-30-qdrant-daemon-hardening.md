# Qdrant 및 MCP 데몬 개선 Implementation Plan

> **For agentic workers:** 이 계획은 논리 단위별 검토용이다. 구현을 시작할 때는 `superpowers:executing-plans`에 따라 각 단위를 테스트와 함께 진행한다.

**Goal:** Qdrant 운영 비용과 동기화 비용을 낮추고, 기동 중 MCP 응답성과 검색 품질을 개선하며, 민감한 운영 기록과 플랫폼 종속성을 정리한다.

**Architecture:** 단일 MCP 데몬과 단일 Qdrant HTTP 서버 구조를 유지한다. 초기 색인은 데몬의 관리 작업으로 실행하며, manifest와 Qdrant의 일괄 점 목록을 비교해 증분 인덱싱한다. Windows 네이티브 Qdrant 전환은 데이터 이행과 롤백을 포함한 마지막 독립 배포 단계로 둔다.

**Tech Stack:** Python 3.11+, FastMCP 4.0.10, MCP SDK 2.2.0, SQLite, Qdrant 1.19.1, PyTorch/SentenceTransformer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-qdrant-daemon-hardening-design.md`

## 공통 제약

- Vault 원문과 검색어 원문을 로그, 테스트 결과, 계획에 출력하지 않는다.
- `127.0.0.1:6333` Qdrant와 `127.0.0.1:8765` MCP 바인딩을 유지한다.
- collection/manifest 교체의 복구 계약을 보존한다. 실패한 업서트는 이전 완료 세대를 잃지 않아야 한다.
- `tools/list` 및 discovery의 modern/legacy 프로토콜 계약을 깨지 않는다.
- Qdrant 기존 저장소는 전환 검증 전 삭제하거나 원본 위치에서 네이티브 프로세스로 열지 않는다.

## 순서와 의존관계

| 단위 | 우선순위 | 선행 | 검토 가능한 결과 |
| --- | --- | --- | --- |
| 0. 기준선 | P0 | 없음 | 실제 호출·메모리·기동 시간과 점 수 기록 |
| 1. 로그 최소화 | P0 | 0 | 새 평문 기록 중단, 과거 데이터 정리, 상한 적용 |
| 2. 비동기 초기 색인 | P0 | 0 | 기동 중 health·목록·발견 응답 |
| 3. 프록시 연결 가드 | P1 | 2 | 오프라인 목록·발견 오류 설명과 단일 재기동 |
| 4. 증분 동기화 비용 | P1 | 0, 2 | 불변 파일 무본문 읽기, Qdrant 일괄 검증 |
| 5. 리랭커 계약 | P1 | 0, 2 | 기본 사용, CUDA→CPU, 실제 적용 결과 표시 |
| 6. 자원 제어 | P2 | 0, 5 | 실제 모델 경로의 설정·측정 |
| 7. 네이티브 Qdrant | P2 | 0~6 | 스냅샷 이행, 서비스 운영, 롤백 검증 |
| 8. Linux 호환성 | P2 | 2~4 | Linux import·잠금·CPU 경로 통과 |

### 단위 0: 기준선과 사실 확인

**Files:** 기존 코드 수정 없음. 결과는 `docs/`의 날짜별 측정 기록으로 남긴다.

- [ ] 현재 Qdrant 버전, collection별 점 수·스키마, manifest 수, health→discovery→tools/list→검색 시간, 무변경 sync 시간과 Qdrant 요청 수를 기록한다. 테스트 데이터와 실제 Vault 결과를 구분한다.
- [ ] 같은 입력에서 프로세스 RSS, GPU 전용 메모리, `vmmemWSL` 메모리, 디스크 I/O를 Docker 유휴/검색/증분 색인 상태별로 측정한다. 네이티브 전환의 이득은 이 기준선과 비교한다.
- [ ] 질의 원문은 읽거나 출력하지 않고 SQLite `filters.rerank`의 선택값만 집계한다. 이미 확인한 190건의 `111 true / 78 false / 1 누락` 및 최근 10건 `10 true`를 기준값으로 기록한다. 이 수치는 리랭커 추론 성공의 증거가 아니라는 점을 명시한다.

### 단위 1: 질의 기록과 파일 로그 제한

**Files:** `src/knowledge_mcp/state.py`, `src/knowledge_mcp/server.py`, `src/knowledge_mcp/daemon.py`, `tests/test_state.py`, `tests/test_server.py`, `.env.example`, `README.md`, `docs/user-guide-ko.md`.

- [ ] 민감 정보가 들어 있는 `queries.query`, `queries.filters`, `query_results.source_path`의 사용처를 확인하고, 새 기록 스키마를 `시각·지연·결과 수·rerank_requested·rerank_applied·실패 코드`로 제한한다. 기본 보존 기간은 30일, `daemon.log`는 크기 기반 회전과 보관 파일 수 상한으로 설계한다.
- [ ] 평문이 신규 SQLite 행과 `daemon.log`에 나타나지 않는 실패 테스트를 만든다. 기존 `test_state.py`의 평문 보존 기대값은 새 정책에 맞춰 변경한다.
- [ ] 기존 DB는 사전 백업 후 일회성 데이터 정리와 SQLite 재작성 또는 `VACUUM`으로 처리한다. 백업 파일까지 자동 삭제하지 않으며 잔존 위치를 사용자에게 알린다. 마이그레이션을 두 번 실행해도 안전해야 한다.
- [ ] 30일 경계, 로그 회전, DB 재시작 후 상태 조회, 기존 `index_runs` 보존 여부를 테스트한다. 새 로그의 민감 문자열 부재와 DB 크기 상한 동작을 확인한다.

### 단위 2: 초기 색인과 MCP 기동 분리

**Files:** `src/knowledge_mcp/daemon.py`, `src/knowledge_mcp/cli.py`, `src/knowledge_mcp/server.py`, `src/knowledge_mcp/indexer.py`, `tests/test_daemon.py`, `tests/test_server.py`, `tests/test_mcp_wire.py`, `README.md`, `docs/user-guide-ko.md`.

- [ ] 데몬 상태를 `starting`, `indexing`, `ready`, `error`로 정의한다. `/health`는 HTTP 서버 생존과 현재 상태를 반환하고, `knowledge-index-status`는 진행 상태 및 마지막 완료/오류를 반환한다.
- [ ] 모델과 도구 등록을 경량화해 HTTP/MCP 서버를 먼저 연다. 초기 증분 색인과 리랭커 warmup은 FastMCP lifespan에 소속된 관리 작업에서 실행하고, 종료 시 취소·기록·정리를 완료한다. 별도 프로세스로 모델을 중복 로드하지 않는다.
- [ ] `KnowledgeIndexer`의 파일 읽기, parse, SQLite, 파일 잠금 획득이 이벤트 루프를 장시간 막지 않게 옮긴다. Qdrant async client를 다른 이벤트 루프에 무심코 전달하지 않는다. 수동 sync와 초기 sync는 하나의 writer 큐를 공유한다.
- [ ] 인덱싱 중 `qdrant-find`만 명시적 “인덱싱 중, 잠시 후 재시도” 도구 오류를 반환한다. 인덱싱이 없고 기존 collection이 유효하면 검색을 허용한다. 초기 실패와 Qdrant 연결 실패는 `indexing`으로 위장하지 않는다.
- [ ] 인위적으로 느린 색인에서 `/health`, discovery, `tools/list`, 상태 조회가 빠르게 완료되는 통합 테스트를 작성한다. 검색의 상태별 응답, 동시 sync 직렬화, 중단 후 manifest 복구를 검증한다.

### 단위 3: 프록시의 목록·발견 연결 실패 처리

**Files:** `src/knowledge_mcp/proxy.py`, `tests/test_proxy.py`, `tests/test_mcp_wire.py`.

- [ ] 현재 `_DaemonGuard.on_call_tool()` 계약을 기준으로 FastMCP 4의 목록 및 discovery 확장 지점을 확인한다. 프로토콜 버전별 호출을 작은 백엔드 서버로 재현한다.
- [ ] 데몬이 없으면 단일 재기동 작업을 시작하고 `tools/list`와 discovery에 설명 가능한 오류를 반환한다. 정상 목록을 빈 목록으로 바꾸지 않는다. backend 세션 생성 실패도 같은 메시지 경로로 처리한다.
- [ ] 재기동 중 동시 목록/발견/도구 호출이 중복 데몬을 만들지 않으며, 복구 후 같은 stdio 연결에서 도구 목록과 검색이 다시 되는 테스트를 작성한다.

### 단위 4: 불변 파일과 Qdrant 요청 수 줄이기

**Files:** `src/knowledge_mcp/state.py`, `src/knowledge_mcp/indexer.py`, `src/knowledge_mcp/qdrant_store.py`, `src/knowledge_mcp/cli.py`, `tests/test_indexer.py`, `tests/test_qdrant_integration.py`, `tests/test_state.py`.

- [ ] `collection_files`에 `mtime_ns`와 크기, sidecar/분류 설정 입력의 서명, 인덱싱 설정 버전을 추가한다. 기존 manifest 행은 첫 sync에서 전체 해시를 통해 안전하게 채운다.
- [ ] stat/메타데이터/설정이 동일하면 파일 본문 읽기와 SHA-256을 건너뛴다. 하나라도 달라지면 기존 `_fingerprint()`와 parse 전후 변경 검사를 수행한다. 같은 크기·같은 mtime으로 되돌린 파일을 확인할 수 있도록 명시적 강제 전체 해시 경로를 제공한다.
- [ ] 파일마다 두 차례 수행하는 `count+scroll` 대신 collection 전체를 페이지 단위로 한 번 순회해 `(source_path, generation, point IDs)`를 manifest와 비교한다. 기존 orphan 복구 순서를 보존하고, 커밋 실패 시 cleanup을 미루는 규칙을 유지한다.
- [ ] `unchanged` sync에서 본문 읽기 0회와 파일당 `count+scroll` 2회 반복 제거를 계측 테스트로 확인한다. Qdrant 검증 요청은 페이지당 순회 요청으로 집계한다. 메타데이터만 변경, 점 누락, generation swap 중 취소, 커밋 실패, 고아점 삭제의 기존 회귀 테스트를 통과시킨다.

### 단위 5: 리랭커 기본값, 디바이스 폴백, 적용 여부

**Files:** `src/knowledge_mcp/search.py`, `src/knowledge_mcp/server.py`, `src/knowledge_mcp/reranker.py`, `src/knowledge_mcp/retrieval.py`, `src/knowledge_mcp/state.py`, `tests/test_retrieval.py`, `tests/test_server.py`, `tests/test_mcp_wire.py`, `README.md`, `docs/user-guide-ko.md`.

- [ ] `rerank` 기본값을 검색 요청과 MCP 도구에서 `true`로 맞춘다. 명시적 `false`는 RRF 전용 검색으로 보존한다. 도구 스키마와 README 기본값도 함께 바꾼다.
- [ ] CUDA 사용 가능 여부를 확인하고 CUDA CrossEncoder 생성 실패 시 CPU로 한 번 폴백한다. CPU도 실패하거나 추론이 실패하면 RRF 결과로 폴백하되 `rerank_requested=true`, `rerank_applied=false`, 안정적인 실패 코드를 사용자에게 드러낸다. 리랭커 실패를 일반 검색 실패로 오인하지 않게 한다.
- [ ] status에 실제 리랭커 디바이스와 마지막 폴백 상태를 표시한다. 로그에는 요청·적용 여부만 기록해 “true 요청이 실제로 동작했는가”를 추후 집계할 수 있게 한다.
- [ ] CUDA 성공, CPU 폴백, RRF 폴백, 명시적 `false`, 비정상 점수 및 기존 출처 제한 순서를 테스트한다. 기존 도구 출력 형식 변경은 `tests/test_mcp_wire.py`에서 검증한다.

### 단위 6: 실제 임베딩 경로의 자원 제한

**Files:** `src/knowledge_mcp/embeddings.py`, `src/knowledge_mcp/config.py`, `src/knowledge_mcp/reranker.py`, `tests/test_embeddings.py`, `tests/test_embedding_providers.py`, `tests/test_resource_limits.py`, `.env.example`, `README.md`.

- [ ] 단위 0 측정값으로 기본 SentenceTransformer/PyTorch 경로의 배치 크기, CPU 스레드, GPU peak를 먼저 조정한다. 설정 범위와 기본값을 검증하고 과도하게 작은 배치의 처리 시간도 비교한다.
- [ ] ONNX/FastEmbed 경로를 유지할 경우, 해당 경로의 실제 `InferenceSession` 생성 지점까지 옵션이 전달되는지 테스트한 뒤 `gpu_mem_limit`, `arena_extend_strategy`, `intra_op_num_threads`를 노출한다. 현재 기본 BGE 경로에 이 옵션이 적용된다고 문서화하지 않는다.
- [ ] 유휴·검색·색인 구간의 RSS, CPU, GPU memory, 처리 시간을 기준선과 비교한다. GPU 메모리 상한은 CUDA arena에만 적용된다는 제한을 문서에 명시한다.

### 단위 7: Docker Qdrant에서 Windows 네이티브 Qdrant로 이행

**Files:** `src/knowledge_mcp/cli.py`, `src/knowledge_mcp/config.py`, `tests/test_cli_paths.py`, `tests/test_qdrant_storage.py`, `tests/test_qdrant_integration.py`, `.env.example`, `README.md`, `docs/user-guide-ko.md`, `docs/qdrant-status-and-diagnostics.md`, `docker-compose.yml`.

- [ ] 버전 `1.19.1` Windows 바이너리의 실행, `qdrant/bm25` Document 처리, localhost 바인딩, 절대 저장소 경로를 빈 임시 디렉터리에서 확인한다. PATH는 수동 CLI 편의용으로만 안내한다.
- [ ] 실행 중인 기존 컨테이너에서 collection snapshot과 SQLite 일관성 백업을 만든다. 원본 저장소는 보존하고 새 네이티브 저장소로 snapshot을 복원한다. Qdrant 문서의 snapshot 버전 호환 조건을 확인한다.
- [ ] 점 수, collection metadata/index, 한글 dense+BM25 검색, rerank, 증분 sync를 기존과 비교한다. 원본 Docker와 네이티브 서버를 동일 포트/저장소에서 동시에 실행하지 않는다.
- [ ] `ensure_qdrant()`를 네이티브 프로세스 health 확인 및 명시적 실행 파일 설정으로 교체한다. 자동 시작이 필요한 Windows 설치에는 서비스 래퍼/등록 명령, 계정 권한, 절대 경로, 로그 회전, 제거·복구 명령을 문서화한다. 일반 실행에는 서비스 설치를 강제하지 않는다.
- [ ] 전환 전후의 `vmmemWSL`, 전체 RSS, 부팅 시간, 색인 시간을 비교한다. 이득이 확인되지 않거나 복원 검증에 실패하면 Docker 구성을 다시 가리키는 롤백 절차를 적용한다. `docker-compose.yml` 제거 여부는 롤백 보존 기간 후 별도 결정한다.

### 단위 8: Linux 호환성

**Files:** `src/knowledge_mcp/state.py`, `src/knowledge_mcp/daemon.py`, `pyproject.toml`, `tests/test_state.py`, `tests/test_daemon.py`, `tests/test_cli_paths.py`, CI 설정 파일, `README.md`.

- [ ] 파일 잠금을 Windows의 `msvcrt`, POSIX의 `fcntl`로 분기하고 동일한 비차단 재시도·해제 계약을 테스트한다. Linux에서 `import knowledge_mcp.cli`가 즉시 성공해야 한다.
- [ ] 플랫폼별 subprocess 종료와 GPU 패키지 설치 가능성을 확인한다. CPU 전용 Linux 설치 경로를 분리해 import 이후 모델 준비 단계에서도 실패하지 않게 한다.
- [ ] Linux CI에서 설치, 전체 unit tests, 임시 Qdrant를 이용한 인덱스→검색→중단 복구를 실행한다. Windows 테스트도 유지한다.

## 최종 검증과 반영 기준

- [ ] `pytest -q` 및 실제 Qdrant 통합 테스트가 통과한다. 외부 서비스가 없어서 건너뛴 테스트 수는 별도로 보고한다.
- [ ] 기동 중 health·discovery·tools/list·status가 응답하고, 검색만 인덱싱 상태를 정확히 알린다.
- [ ] 무변경 sync에서 파일 본문 재읽기가 없고 파일당 4회 수행하던 Qdrant 검증 요청이 컬렉션 페이지 순회로 대체된다.
- [ ] 기본 검색의 리랭커 요청/실제 적용 여부가 구분되고, CUDA 불가 시 CPU 또는 명시적 RRF 폴백이 작동한다.
- [ ] 새 질의 원문이 SQLite와 파일 로그에 없고, 기존 데이터 정리와 보존 상한이 확인된다.
- [ ] 네이티브 Qdrant 전환은 snapshot 복원·검색 비교·롤백 리허설이 완료된 뒤에만 운영 환경에 적용한다.
