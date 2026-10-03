# Qdrant 및 데몬 하드닝 아키텍처 개편 종합 요약 (2026-10-03)

## 1. 개요 및 목적

본 문서는 `obsidian-knowledge-mcp`의 아키텍처 안정성, 검색 응답성, 프라이버시 보호 및 크로스 플랫폼(Windows/Linux) 호환성을 확보하기 위해 수행된 **Qdrant 및 데몬 하드닝(Hardening) 개편 작업(단위 0 ~ 단위 8)**의 전체 결과와 기술적 세부 사항을 정리합니다.

기존 시스템은 WSL2 기반 Docker Qdrant에 의존하여 약 4GB의 WSL 가상 메모리를 상시 점유했고, 동기식 초기 색인으로 인한 클라이언트 타임아웃, 검색 쿼리 평문 저장, Windows 전용 잠금(`msvcrt`)으로 인한 Linux 비호환 등의 문제가 존재했습니다. 본 작업을 통해 시스템 전반의 견고성과 성능을 대폭 강화했습니다.

---

## 2. 단위별 주요 변경 내역 (단위 0 ~ 단위 8)

### 단위 0: 기준선 측정 및 벤치마크 환경 수립
- **커밋**: `fd03ff3`
- 기존 FastMCP 2.7 / MCP 1.30 환경과 FastMCP 4.0.10 / MCP 2.2.0 환경의 동작 비교 기준선 확립.
- 전체 기준선 테스트 147 passed 및 20개 합성 문서를 활용한 정량적 벤치마크 하네스 구성.

### 단위 1: SQLite 및 데몬 로그 프라이버시 최소화
- **커밋**: `679f557`
- 검색 기록 테이블에서 사용자 검색 쿼리 평문 저장을 전면 제거하고 지연 시간, 결과 수, 리랭커 적용 여부, 안정적인 에러 코드만 저장.
- 질의 이력 30일/10,000행, 색인 이력 30일/1,000행 보존 한도(Retention) 적용 및 VACUUM 자동화.
- 데몬 파일 로그 1MiB 크기 제한 및 최대 3개 회전(Rotation) 적용.

### 단위 2: 비동기 초기 색인 및 FastMCP 수명주기(Lifespan) 분리
- **커밋**: `412b6c6`, `4af0ce6`
- 데몬 기동 시 초기 증분 색인을 백그라운드 태스크로 분리하여 `/health`, `server/discover`, `tool/list`가 즉각(Instant Connect) 응답하도록 개선.
- FastMCP lifespan에 모델 로드 및 백그라운드 색인 수명주기를 완벽히 통합하여 좀비 프로세스 방지.
- 색인 중 검색 도구 호출 시 블로킹 없이 "현재 색인 진행 중" 안내 반환.

### 단위 3: 프록시 데몬 장애 가드 및 자가 치유(Self-Healing)
- **커밋**: `8f4e3b5`
- 데몬 헬스체크 성공 직후 발생할 수 있는 503 Service Unavailable 및 비정상 세션 단절에 대비한 재시도 및 프록시 재기동 가드 추가.
- 클라이언트(Codex, Antigravity, Claude Code)가 무한 행(Hang)에 빠지는 현상을 원천 차단.

### 단위 4: 무변경 Sync 캐싱 및 Qdrant 페이지 순회 최적화
- **커밋**: `97410c2`
- 전체 파일을 매번 Read/SHA256 해싱하던 병목을 `LastModifiedTime(mtime)` 및 `FileSize` 기반 1차 캐시 검증으로 스킵.
- 파일별 개별 포인트 검증(Count/Scroll)을 인벤토리 일괄 비교로 전환하여 Qdrant 네트워크 왕복을 획기적으로 축소.
- **성과**: 무변경 sync 소요 시간 **0.32초 → 0.04초 (약 7배 개선)**, Qdrant 호출 횟수 **40회 → 0회**.

### 단위 5: 리랭커 기본 활성화 및 다계층 폴백(Fallback)
- **커밋**: `77e5867`
- `qdrant-find` 도구의 기본값을 `rerank: true`로 승격.
- CUDA 가용성 실패 시 CPU 리랭커로 1회 안전 재시도, 리랭커 전체 실패 시 기존 RRF(Reciprocal Rank Fusion) 하이브리드 점수와 순서를 완벽히 유지.
- 검색 결과 메타데이터에 `rerank_requested`, `rerank_applied`, `rerank_error`를 명시하여 상태 투명성 확보.

### 단위 6: PyTorch / CUDA 인퍼런스 리소스 상한 제어
- **커밋**: `2199a97`, `5c707f3`
- `LocalSentenceTransformerProvider` 추론 배치 크기를 8로 축소하고 CPU 스레드 수를 4로 제한.
- PyTorch CUDA allocator fraction(기본 약 0.24)을 정규화하여 다른 GPU 프로세스와의 경합 방지 및 VRAM 폭증 억제.

### 단위 7: Windows 네이티브 Qdrant 프로세스 관리 및 복제 검증
- **커밋**: `e02cc0f`, `3f67210`
- 시스템 PATH에 위치한 `qdrant.exe`를 직접 백그라운드로 구동하는 프로세스 관리자 구현 (`ensure_qdrant`).
- 기존 Docker 스토리지와 네이티브 스토리지를 분리하고, 스냅샷 WAL 결함(Qdrant #7956)을 우회하여 API 기반 읽기 전용 Export/Import로 1,607개 포인트를 100% 무손실 복제 검증.

### 단위 8: 표준 라이브러리 크로스 플랫폼 파일 락 및 Linux/CI 호환성
- **커밋**: `8f750be`, `799f327`, `5cf1492`, `24e9191`, `eed4e20`, `443980c`
- **표준 라이브러리 크로스 플랫폼 파일 락**: `msvcrt` 단독 임포트를 제거하고 Windows(`msvcrt.locking`)와 POSIX(`fcntl.flock`) 분기를 표준 라이브러리만으로 구현하여 무의존성 및 충돌 안전성 달성.
- **POSIX 프로세스 제어**: POSIX 환경에서 `start_new_session=True` 및 `os.kill(pid, signal.SIGTERM)` 표준 시그널 전파 적용.
- **Linux CPU 패키징 분리**: `pyproject.toml`에 PEP 508 환경 마커를 적용하여 Linux 환경에서 불필요한 CUDA/NVIDIA 의존성 배제 및 경량 CPU 휠 자동 선택.
- **GitHub Actions CI 매트릭스**: Windows 및 Ubuntu(Linux) x Python 3.11, 3.12 전수 자동 검증 워크플로(`.github/workflows/ci.yml`) 구성.

---

## 3. 정량적 검증 및 성능 개선 결과

| 항목 | 변경 전 (Baseline) | 변경 후 (Hardened) | 개선율 / 효과 |
|---|---|---|---|
| **무변경 Sync 소요 시간** | 0.325초 | 0.043초 | **약 7.5배 속도 향상** |
| **무변경 Sync 시 Qdrant 요청** | 40회 | 0회 | **불필요한 네트워크 I/O 제거** |
| **검색 쿼리 프라이버시** | 평문 질의 저장 | 평문 저장 전면 제거 (메트릭만 보존) | **개인정보 및 보안 강화** |
| **질의/색인 로그 보존** | 무제한 누적 | 30일 / 10,000행 상한 자동 회전 | **디스크 용량 누수 방지** |
| **데몬 시작 블로킹** | 색인 완료까지 수 초~수십 초 대기 | 즉각 응답 (`< 50ms`), 백그라운드 색인 | **클라이언트 타임아웃 100% 해소** |
| **리랭킹 기본값 및 안정성** | 비활성화 또는 GPU 전용 | 기본 활성화 (`rerank=true`) + CPU/RRF 폴백 | **검색 품질 및 복원력 확보** |
| **Qdrant 런타임** | Docker Compose / WSL2 | Windows 네이티브 / POSIX 지원 | **WSL2 vmmem 오버헤드 격리 가능** |
| **플랫폼 호환성** | Windows 전용 (`msvcrt` 크래시) | Windows + Linux/POSIX 완벽 호환 | **CI 자동화 및 서버 배포 가능** |

---

## 4. 테스트 및 품질 게이트 통과 현황

- **Windows 11 환경 (Python 3.13.5)**:
  - 실행 명령: `pytest tests/ -q`
  - 결과: **277 passed in 100%** (모든 단위 테스트, 통합 테스트, 동시성 락, 데몬 수명주기 전수 통과)
- **Linux WSL Ubuntu 환경 (Python 3.12 CPU venv)**:
  - 실행 명령: `wsl pytest tests/ -q`
  - 결과: **276 passed, 1 skipped** (Docker Compose 라이브 테스트 1건 제외 전수 100% 통과)
- **CI 파이프라인**:
  - Windows & Ubuntu 매트릭스 상에서 단위 테스트 자동 실행 확인.

---

## 5. 운영 마이그레이션 안내 및 향후 과제

1. **운영 Qdrant 네이티브 전환**:
   - `KNOWLEDGE_QDRANT_MODE=native` 및 `KNOWLEDGE_QDRANT_NATIVE_STORAGE` 환경 변수를 지정하여 네이티브 바이너리로 완전 전환 가능.
   - 기존 Docker 컨테이너는 필요 시 즉각 롤백할 수 있도록 안전하게 유지됨.
2. **개인 Obsidian Vault 연계**:
   - Vault 내 작업 로그(`20_Projects/[경험] Qdrant 하드닝 및 데몬 안정화 아키텍처 개편.md`) 및 메인 프로젝트(`[개인] Obsidian Qdrant RAG 구축.md`)에 반영 완료.
3. **향후 모니터링**:
   - 네이티브 Qdrant의 장기 가동 시 메모리 추이 및 수만 개 포인트 규모에서의 증분 색인 성능 지속 모니터링.
