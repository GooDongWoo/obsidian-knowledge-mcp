# Qdrant 및 MCP 데몬 개선 검증 기록

## 기준선 — 2026-10-01

- 작업 브랜치: `codex/qdrant-hardening`, 기준 커밋: `3a5846c`.
- 프로젝트 기본 `.venv`는 FastMCP 2.7.0 / MCP 1.30.0이었다. 테스트는 기존 마이그레이션 환경의 FastMCP 4.0.10 / MCP 2.2.0 / Python 3.13.5로 수행하고, 새 작업 공간의 `src`를 사용한다.
- 전체 기준선: **147 passed, 1 failed, 292.00초**. 실패는 Qdrant에서 임시 시험 collection을 삭제할 때 Windows 바인드 마운트의 `Permission denied`가 발생한 것이다. 같은 두 parameter case만 다시 실행하면 **2 passed, 17.67초**였다. 테스트 기대값을 완화하지 않았다.
- 운영 Qdrant 버전: 1.19.1. 기본 collection: 5,290 points, status=green. 운영 데몬 health=200, PID=36132.
- 실행 중 관측한 WSL working set 약 1,231 MB, 모델 데몬 working set 약 356 MB. GPU는 RTX 4070 Ti 12,282 MiB이며 당시 전체 GPU 사용량은 8,627 MiB였다. 다른 프로세스와 공유되는 전체 GPU/WSL 수치이므로 이 프로젝트만의 사용량이나 절감량으로 해석하지 않는다.
- 실제 운영 HTTP 호출 한 번의 관측: discovery 0.664초, tools/list 0.003초, rerank=false 검색 0.169초, rerank=true 검색 0.830초. 검색은 각각 8개 결과를 반환했다. 질의와 결과 원문은 기록하지 않았다. 반복 표본의 p50/p95가 아니다.

## 임시 데이터의 저장소 시험

모델 추론 비용을 배제한 결정적 1,024차원 시험 벡터와 한글 문서 20개를 사용했다. 운영 collection에는 쓰지 않았다.

| 환경 | 초기 색인 | 무변경 sync | count | scroll | 무변경 점 수 |
|---|---:|---:|---:|---:|---:|
| 기존 Docker Qdrant | 30.888초 | 0.325초 | 40회 | 41회 | 20 |
| Windows 네이티브 Qdrant | 25.752초 | 0.343초 | 40회 | 41회 | 20 |

두 환경 모두 초기 added=20, 이후 unchanged=20, failed=0, 검색 결과=8이었다. 무변경 sync에는 위 요청 외에 collection_exists/get_collection 각 1회가 포함된다. 네이티브 전환만으로 파일별 검증 요청은 줄지 않으므로 동기화 알고리즘 개선은 별도 필요하다.

Windows 공식 바이너리 `qdrant-x86_64-pc-windows-msvc.zip`은 GitHub release v1.19.1의 SHA-256 `9b6f69bd85f6abed4bc13f943099f55c6ffd55f5dd90388635320d8fbb569eb0`와 일치했다. 네이티브는 별도 포트 16333, 독립 저장소를 사용하고 시험 후 종료했다.

이 PC의 `LongPathsEnabled=0`에서 긴 작업 공간 아래 저장소는 Gridstore 경로 오류를 일으켰고, 짧은 사용자 홈 시험 디렉터리를 사용하면 collection 및 모든 payload index 생성, 한글 BM25 동작이 성공했다. 최종 설치는 짧은 저장 경로를 사용하거나 긴 경로 환경을 별도 검증해야 한다.

## 구현 및 최종 검증

각 단위의 결과와 최종 검증은 구현 진행에 따라 이 문서에 추가한다.

### 단위 1 — 운영 기록 최소화

- 검색 기록은 시간·지연·결과 수·리랭커 요청/적용 여부·안정적인 오류 코드만 저장한다. 과거 DB는 SQLite 일관성 백업 후 원문 테이블/컬럼을 제거하고 VACUUM한다. 중단된 마이그레이션/압축을 재시작에서 복구한다.
- 질의 이력은 기본 30일/10,000행, 과거 색인 이력은 30일/1,000행으로 제한한다. 컬렉션별 마지막 상태와 manifest는 보존한다. 데몬 파일 로그는 기본 1MiB + 회전본 3개이며 고정 이벤트 코드만 저장한다.
- 관련 상태·서버·데몬·색인·MCP wire 테스트 **80 passed**. 단위 검토에서 spec compliant / quality approved, 차단 이슈 없음. 운영 DB는 아직 변경하지 않았다.
- `state.pre-privacy.sqlite3` 및 중단 시 임시 sibling에는 기존 민감 정보가 남을 수 있다. 자동으로 삭제하지 않으며 운영자가 복구 필요성을 확인한 뒤 정리해야 한다.

### 실제 PyTorch/CUDA 배치 기준선

별도 프로세스에서 모델별로 합성 문서 40개, CPU 스레드 4개, CUDA allocator fraction 0.24, 1개 입력 warmup 후 측정했다. 모델은 로컬 cache에서 읽고 두 모델을 동시에 새로 로드하지 않았다. 실제 운영 질의·문서는 사용하지 않았다.

| 모델 | 배치 | 처리 시간 | CUDA peak allocated | peak reserved | 추론 후 RSS |
|---|---:|---:|---:|---:|---:|
| BGE-m3-ko | 32 | 0.3844초 | 2,550.57MiB | 2,782MiB | 1,497MiB |
| BGE-m3-ko | 8 | 0.3937초 | 2,277.54MiB | 2,356MiB | 1,500MiB |
| BGE-m3-ko | 4 | 0.4234초 | 2,229.11MiB | 2,262MiB | 1,500MiB |
| 한국어 리랭커 | 32 | 0.3874초 | 2,556.08MiB | 2,784MiB | 1,400MiB |
| 한국어 리랭커 | 8 | 0.3849초 | 2,270.36MiB | 2,316MiB | 1,402MiB |
| 한국어 리랭커 | 4 | 0.4139초 | 2,224.09MiB | 2,260MiB | 1,403MiB |

배치 32 대비 8의 임베딩 최대 절대 차이는 8.95e-7, 리랭커 점수 차이는 2.02e-9였다. 이 합성 표본은 배치 8의 기본값 후보를 뒷받침한다. 단일 관측이며 실제 Vault 처리 시간의 p50/p95나 시스템 전체 메모리 상한을 뜻하지 않는다. 각 프로세스의 모델 직후 유휴 RSS는 약 1,114MiB였다. 현재 기본 경로는 SentenceTransformer/PyTorch이므로 ONNX CUDA arena 옵션은 이 측정에 적용되지 않는다.

### 단위 2 — 초기 색인과 MCP 기동 분리

- HTTP/도구를 먼저 등록하고 FastMCP lifespan이 초기 준비·색인·warmup·종료 정리를 소유한다. starting/indexing/ready/error, 진행 상태 및 마지막 결과를 응답한다. 초기/수동 sync는 하나의 writer 큐를 공유한다.
- 디스크·SQLite·모델 작업은 이벤트 루프 밖에서 수행하며, 취소돼도 진행 중인 native 작업이 끝난 뒤 파일 잠금과 클라이언트를 정리한다. warmup과 첫 추론은 모델 생성 잠금을 공유한다.
- 상태·색인·MCP 호환성 테스트 **90 passed**. 검토에서 발견된 모델 중복 생성과 조기 취소 정리는 재현 후 수정했고, 해당 검증 **24 passed**, 재검토에서 두 항목 모두 해결됐다.
- 기존 검증 환경이 제거된 뒤 Windows 검증은 main의 `.venv`를 사용한다. 해당 환경은 FastMCP4.0.10/MCP2.2.0/Torch2.11.0+cu128/pytest9.1.1이며, 각 명령에서 구현 worktree/src를 PYTHONPATH로 지정한다.

### 단위 3 — 목록·discovery 연결 가드 (2026-10-02)

- 목록·discovery·도구 호출은 재기동 작업 하나를 공유한다. health=200이어도 MCP 세션이 503으로 실패하는 경우를 요청별 신호로 감지하며, 공식 전송/세션을 그대로 사용한다. 빈 목록으로 연결 실패를 숨기지 않는다.
- 프록시·wire **40 passed**, 최종 프록시 **22 passed** 및 동일 stdio 연결의 modern/legacy 장애→복구 **2 passed**. 단위 검토 spec/quality approved, 차단 이슈 없음.
- 확인된 SDK2.2.0 한계: 데몬이 완전히 끊긴 순간 새 auto 연결을 시작하면, discovery의 복구 메시지를 SDK가 legacy initialize fallback의 프로토콜 오류로 바꿀 수 있다. SDK stream은 middleware 이전에 이 handshake를 거부한다. 일반 CLI는 데몬 생존 확인/시작 후 프록시를 열며, HTTP가 이미 열려 있는 초기 준비/색인 중 동작 및 기존 연결의 장애 복구는 검증됐다. SDK dispatch를 교체하거나 프로토콜 오류 코드를 변조하지 않았다.

### 단위 4 — 무변경 sync 실측 (2026-10-02)

기준선과 같은 합성 문서 20개 및 결정적 시험 벡터로 측정했다. 운영 collection은 변경하지 않고, 임시 UUID collection을 삭제한 뒤 독립 네이티브 프로세스도 종료했다.

| 환경 | 무변경 sync 이전 → 이후 | count 이전 → 이후 | scroll 이전 → 이후 |
|---|---:|---:|---:|
| Docker | 0.325초 → 0.047초 | 40 → 0 | 41 → 1 |
| Windows 네이티브 | 0.343초 → 0.043초 | 40 → 0 | 41 → 1 |

양쪽 모두 unchanged=20, failed=0, 점 수=20, 검색 결과=8이었다. collection_exists/get_collection 각 1회는 유지된다. 본문 읽기 생략은 별도 동작 테스트로 검증하며, inventory 페이지 수는 실제 점 수에 비례한다. 처리 시간은 단일 관측으로 반복 성능 분포를 뜻하지 않는다. 같은 크기와 mtime을 보존한 편집은 `force_full_hash`로 검증해야 한다. 단위 검토 및 관련 복구 테스트 결과는 확정 후 추가한다.

단위 4 검토는 spec compliant / quality approved이며 차단 이슈가 없었다. 관련 게이트 110 passed와 기존 Docker 삭제 권한 오류의 단일 재실행 1 passed, 최종 정수 ID·복구 회귀 7 passed를 확인했다. 전체 재색인 시 inventory 갱신의 이차 시간 복잡도는 비차단 개선 사항으로 최종 검토에 전달했다.

### 단위 5 — 기본 리랭킹 및 폴백

- 검색과 MCP 스키마의 기본값은 `rerank=true`다. CUDA 생성 실패 시 CPU로 한 번 재시도하며, 성공한 CPU 리랭킹은 실제 적용으로 표시한다. 초기화·추론·비정상 점수 실패는 원래 RRF 순서와 점수를 유지하고 안정적인 오류 코드를 전달한다.
- 각 결과의 `rerank_requested`, `rerank_applied`, `rerank_error`가 요청 결과와 함께 이동한다. 상태는 실제 디바이스·폴백을 보여 주며 운영 기록에도 요청과 실제 적용을 구분한다. 명시적 false와 빈 후보는 리랭킹하지 않는다.
- 집중 검증 104개 통과, spec compliant / quality approved이며 지적 사항이 없었다. 넓은 게이트의 프록시 복구 정체는 단독 실행에서 재현되지 않았다(1 passed, 5.49초). 전체 Windows/Linux 실행으로 테스트 순서에 따른 문제까지 확인해야 하며 아직 최종 통합 통과로 주장하지 않는다.

### 단위 6 — 실제 provider 리소스 설정 검증

설정값을 전달한 프로젝트 `LocalSentenceTransformerProvider`를 별도 프로세스에서 실행했다. 로컬 cache 모델과 합성 입력만 사용했으며 CUDA 디바이스, 배치 8, 실제 PyTorch 스레드 4, allocator fraction 약 0.24를 확인했다. 문서 40개의 1,024차원 벡터와 질의 벡터가 생성됐다.

| 구간 | 시간 | RSS | CUDA peak allocated |
|---|---:|---:|---:|
| 모델 준비 | 18.907초 | 1,139.97MiB | 2,165.94MiB |
| 유휴 | 0.200초 | 1,139.99MiB | 2,165.94MiB |
| 질의 임베딩 | 0.204초 | 1,564.73MiB | 2,175.24MiB |
| 문서 임베딩 | 0.417초 | 1,568.67MiB | 2,277.54MiB |

문서 임베딩 피크는 앞서 측정한 배치 8과 일치했다. 첫 호출과 별도 프로세스의 RSS 차이가 있어 이전 정상 상태 시간과 직접 비교하지 않는다. 전체 Qdrant 검색·색인 시간이 아닌 추론 구간의 단일 관측이다. CPU 사용률은 한 코어를 100%로 한 프로세스 CPU 시간/경과 시간으로 기록했으며 시스템 전체 점유율이나 CPU quota가 아니다.

리소스 설정·임베딩·검색·취소·초기화의 집중 게이트는 71 passed였다. 실제 FastEmbed의 SessionOptions/providers 경계를 검증하며, 기본 BGE 모델에 ONNX 옵션이 적용된다고 표시하지 않는다. 단위 검토 결과는 확정 후 추가한다.

단위 6 검토에서 허용한 정수 allocator fraction `1`이 실제 PyTorch API에 거부되는 문제를 발견했다. 설정값을 검증 후 실수로 정규화했고 실제 API 경계에서 임베딩·리랭커 모두 RED 2 → GREEN 20을 확인했다. 재검토에서 지적 사항이 해결됐고 새 문제는 없었다.

### 단위 7 — 네이티브 실행과 이행 리허설 (2026-10-03)

- 네이티브 실행은 health 확인, 명시적 바이너리 또는 PATH, 별도 짧은 저장소 및 소유 표식을 사용한다. 새 `KNOWLEDGE_QDRANT_NATIVE_STORAGE`와 기존 Docker용 `KNOWLEDGE_QDRANT_STORAGE`를 분리한다. 기존 Docker 저장소는 직접 열지 않았다.
- 기존 collection snapshot과 읽기 전용 SQLite 일관성 백업을 writer 잠금 아래 확보했다. snapshot은 1,015,198,208바이트, SHA-256 `cb3f4c537ea69a8e3453767efa9a264b55e9e24ece82cc60dfae4a58bfe9a6a6`이며 원본 그대로 보관한다.
- **Snapshot 복원 실패:** WAL `first-index`가 18개 NUL 바이트여서 같은 버전 1.19.1 복원도 실패했다. 유사 증상이 [Qdrant 공식 이슈 #7956](https://github.com/qdrant/qdrant/issues/7956)에 보고되어 있다. 원인 수정이나 snapshot 복원 성공으로 주장하지 않는다.
- 대신 별도 새 저장소에 읽기 전용 API export/import를 검증했다. 현재 소스와 대상은 정확히 1,607개 point이며 ID·generation·payload·dense/sparse 벡터·collection config·7개 payload index가 동일했다. dense 최대 절대 오차는 0이었다. 이전 5,290개 기준선은 과거 관측이며 변경 원인을 추정하지 않았다.
- 실제 CUDA 질의 임베딩과 리랭커로 한국어 질의 3개를 비교했다. dense/BM25 상위 20개 및 리랭커 최종 8개 순서는 일치하고 모든 리랭킹이 적용됐다. RRF 후보 ID별 점수는 완전히 같았으나 동점 안의 순서는 달랐다. 서로 다른 점수의 순서 역전은 없었다.
- API 복제본 재시작은 5.074초, 준비 직후 RSS 약 174MiB, 검색 후 약 205MiB였다. 전체 RSS와 vmmemWSL은 다른 프로세스 및 기존 Docker를 포함하므로 전환 후 메모리 절감량으로 해석하지 않는다. API import 시간은 기록되지 않아 재측정하거나 추정하지 않았다.
- 합성 문서 20개에서 양쪽 초기 added=20, 무변경 unchanged=20, 변경 단계 changed=1/added=1/deleted=1/unchanged=18, 다음 무변경 unchanged=20이었다. 모두 실패 0, 점 수 20을 유지했다. 초기 Docker 29.621초/네이티브 23.898초, 무변경 0.035/0.031초, 변경 1.901/1.818초의 단일 관측이었다.
- 관련 게이트 **74 passed**. 단위 검토는 진행 중이다. 실제 운영 전환에는 최신 export와 SQLite 백업 쌍이 필요하며, 이번 리허설 자료를 이후에도 최신이라고 가정하지 않는다.
- 백업·private export는 `C:/Users/dongwoo/.qdrant-rehearsal/20261003`에 보존한다. 기존 질의 정보가 포함된 SQLite 백업과 문서 payload를 포함한 snapshot/export는 공개·커밋 대상이 아니다. 시험용 네이티브 프로세스는 모두 종료했고 운영 Docker는 정상 유지했다.
