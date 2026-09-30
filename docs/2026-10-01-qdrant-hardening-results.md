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
