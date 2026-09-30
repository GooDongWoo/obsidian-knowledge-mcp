# Qdrant 및 MCP 데몬 개선 설계 초안

## 목표와 범위

현재의 단일 로컬 Qdrant, 단일 모델 데몬, stdio 프록시 구조를 유지하면서 기동 지연, 불필요한 동기화 비용, 민감한 검색 기록, 리랭커 정책, 플랫폼 종속성을 개선한다. Qdrant의 Windows 네이티브 실행 전환은 별도 배포 변경으로 다룬다. 각 작업은 독립적으로 검증하고 되돌릴 수 있어야 한다.

## 코드에서 확인한 현황

- `cli.ensure_qdrant()`는 매번 Docker Compose를 실행한다. `daemon.run_daemon()`과 standalone 경로는 초기 증분 동기화와 리랭커 warmup이 끝난 뒤에 MCP 서버를 연다.
- `indexer._fingerprint()`는 매번 원본 전체를 읽고 SHA-256을 계산한다. 변경 없는 파일도 `generation_matches()`의 `count`와 `scroll`을 실행하고, 마지막 검증에서 다시 같은 두 요청을 실행한다. 전역 orphan 검사도 별도 `scroll`을 수행한다.
- 기본 임베딩 모델은 `LocalSentenceTransformerProvider`를 사용한다. 코드의 FastEmbed/ONNX 경로는 기본 모델에서 선택되지 않는다. 현재 배치 크기 32와 PyTorch 스레드 상한 4가 적용된다.
- 리랭커는 `CrossEncoder(..., device="cuda")`만 생성한다. 문서의 CPU 설명은 오래되었다. `rerank` 기본값은 요청 모델과 도구 모두 `false`다.
- `state.sqlite3`는 질의 원문, 필터, 검색 결과의 경로를 보관하며 보존 기한이 없다. `daemon.log`도 append만 수행한다. `state.py`는 `msvcrt`를 무조건 import한다.
- 프록시의 재기동 가드는 `tools/call`에만 있다. `tools/list` 연결 실패를 설명 없이 예외로 올리는 동작을 테스트로 고정해 두었다.
- 로컬 질의 기록을 원문 없이 집계하면 전체 190건 중 `rerank=true` 111건, `false` 78건, 누락 1건이며 최근 10건은 모두 `true`다. 이것은 기록된 호출의 선택값이며 실제 리랭커 성공 여부를 증명하지는 않는다.

## 설계 결정

1. **기동과 인덱싱 상태를 분리한다.** HTTP/MCP 라우트를 먼저 열고 초기 동기화는 데몬 내부의 관리되는 백그라운드 작업으로 실행한다. `health`는 프로세스 생존과 `starting/indexing/ready/error` 상태를 보여 준다. `tools/list`와 discovery는 인덱싱과 무관하게 응답한다. 검색만 실제 인덱싱 실행 중이면 명시적 재시도 메시지를 반환한다. 기존 인덱스가 있고 동기화가 실패한 경우 검색 가능 여부와 오류 상태를 따로 표시한다. 디스크 읽기와 파일 잠금 획득은 이벤트 루프를 막지 않게 처리한다.
2. **프록시의 연결 실패를 프로토콜별로 처리한다.** `tools/call`은 기존 의미를 보존하고, `tools/list`와 discovery는 데몬 재기동을 한 번만 요청하면서 이해 가능한 연결 실패를 전달한다. 빈 도구 목록을 정상 결과처럼 반환하지 않는다. SDK의 modern/legacy 협상은 그대로 유지한다.
3. **동기화는 저비용 후보 판별과 일괄 무결성 검사를 결합한다.** 완료된 manifest에 `mtime_ns`, 크기, sidecar 및 분류 입력의 서명을 저장한다. 서명이 같으면 원본 파일 읽기를 생략하고, 달라지면 기존 전체 해시와 parse 전후 검사를 유지한다. Qdrant의 파일별 `count+scroll` 두 차례를 컬렉션별 한 번의 페이지 순회에서 얻은 generation/point ID 대조로 통합한다. 명시적 강제 검증 모드에서 전체 해시를 다시 계산할 수 있게 한다. 타임스탬프와 크기를 보존한 외부 편집은 일반 증분 동기화가 놓칠 수 있음을 문서화한다.
4. **리랭커 정책과 실제 적용 결과를 분리한다.** 기본 요청은 `rerank=true`로 바꾸는 안을 제안한다. CUDA를 사용할 수 없으면 CPU CrossEncoder를 사용한다. 리랭커 초기화 또는 추론이 실패하면 RRF 결과를 반환하되, 요청 여부와 실제 적용 여부 및 안정적인 실패 코드를 상태/응답에 드러낸다. 폭넓은 예외 삼키기로 검색 품질 저하를 숨기지 않는다.
5. **민감한 운영 기록을 최소화한다.** 새 질의 원문과 결과 경로는 기록하지 않는다. 필요한 경우 시간, 지연, 요청/실제 리랭크 여부, 결과 수, 실패 코드만 기록한다. 기존 원문 데이터는 백업과 복구 절차를 명시한 일회성 정리로 제거하고, SQLite 페이지 재사용 및 외부 백업의 잔존 가능성을 구분한다. 운영 기록과 파일 로그에 보존량/기간 상한을 둔다.
6. **자원 제어는 실제 모델 경로에 맞춘다.** PyTorch/SentenceTransformer의 CPU 스레드, 임베딩 배치, 리랭커 동시성, GPU 메모리 사용량을 측정하고 설정 가능한 범위를 정한다. ONNX의 `gpu_mem_limit`, `arena_extend_strategy`, `intra_op_num_threads`는 ONNX 모델 경로를 실제로 사용할 때만 적용한다. `gpu_mem_limit`는 CUDA arena의 제한이며 프로세스 전체 VRAM 한도가 아니다.
7. **Qdrant 네이티브 전환은 별도 이행 작업이다.** 먼저 현재 버전과 호환되는 Windows 바이너리, BM25 기능, 저장소 경로, localhost 바인딩을 검증한다. 새 데이터 디렉터리에 같은 버전 Qdrant를 띄워 collection snapshot으로 복원하고 점 수·스키마·검색을 비교한다. 전환 시 한 프로세스만 포트 6333과 쓰기 권한을 가지게 한다. 성공 후 `ensure_qdrant()`의 Docker 의존을 제거하고, Windows 서비스 등록은 별도 설치 절차로 제공한다. PATH 추가는 대화형 CLI 편의 설정이며 서비스 실행은 절대 경로를 사용한다.
8. **Linux 지원은 독립 범위다.** `msvcrt`/`fcntl` 분기뿐 아니라 GPU 의존성, subprocess 제어, 잠금 계약을 검증한다. Linux CPU 환경의 import, CLI, 잠금, 기본 검색 경로를 CI에서 실행한다.

## 검증 원칙

- 현재 점 수와 검색 결과를 기준선으로 기록하고 각 변경 후 비교한다. 사용자의 Vault 원문이나 질의 원문은 테스트 출력과 계획 문서에 넣지 않는다.
- 인덱싱 중 health, discovery, tools/list, status가 응답하고 검색만 재시도 메시지를 돌려주는지 확인한다.
- 변경 없음, 메타데이터만 변경, 파일 삭제, 업서트 후 중단, manifest 커밋 실패, Qdrant 점 누락, 복구 실패를 회귀 테스트로 확인한다.
- 네이티브 전환은 스냅샷 복원과 되돌리기를 같은 버전으로 리허설한 뒤 진행한다. 저장소 디렉터리를 Docker와 네이티브 프로세스가 동시에 열지 않는다.

## 외부 근거

- Qdrant 설치 및 Windows/WSL 마운트 경고: https://qdrant.tech/documentation/operations/installation/
- Qdrant Windows 바인드 마운트 문제: https://qdrant.tech/documentation/operations/common-errors/
- Qdrant collection snapshot 복원 호환성: https://qdrant.tech/documentation/operations/snapshots/
- Qdrant localhost 바인딩: https://qdrant.tech/documentation/security/
- ONNX Runtime CUDA provider 옵션: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime 스레드 설정: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
