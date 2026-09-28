# FastMCP 4 / MCP protocol 마이그레이션 결과

2026-09-29, Windows Python 3.13.5. 구현 브랜치: `codex/fastmcp4-protocol`.

## 적용 결과

- FastMCP `2.7.0` → `4.0.10`, 직접 사용하는 MCP SDK `2.2.0` 고정.
- 데몬을 Streamable HTTP `http://127.0.0.1:8765/mcp`로 전환. `/health`는 별도 readiness 검사로 유지.
- 수동 SSE/stdio 중계와 `server/discover`의 강제 `-32601` 응답을 제거. 공식 proxy가 frontend/backend protocol era를 요청별로 반영한다.
- 최신 client의 `auto` discovery는 `2026-07-28`을 사용한다. legacy는 같은 `/mcp`에서 정상 initialize 협상으로 `2025-11-25`를 사용한다.
- standalone의 중첩 sync runner를 async runner로 수정. 기본 CLI/proxy에서는 torch, ONNX Runtime, fastembed, sentence-transformers를 import하지 않는다.
- 검색의 결과별 TextContent JSON 형식과 기존 필터·기본값·한글·위치·score를 보존하고 structured output을 추가했다. 새 SDK schema는 title을 생략하고 additionalProperties=false를 명시한다. modern 인자 오류는 `-32602` protocol 오류로 반환한다.
- 충돌하는 `qdrant-mcp` extra 제거. Windows/GPU 재설치용 constraints 및 README·한국어 가이드·현재 architecture HTML/PNG 갱신.

## 실제 부작용과 보완

1. AnyIO task group을 FastMCP lifespan의 yield에 걸치면 종료 시 cancel scope 순서가 깨졌다. restart task를 asyncio가 관리하고 lifespan 종료에서 정리하도록 수정했다.
2. 새 SDK의 취소는 색인을 중간에 중단할 수 있다. `partial / sync_cancelled`를 기록하고, Qdrant generation swap 후 취소된 경우에도 다음 일반 증분 sync로 복구됨을 실제 Qdrant에서 검증했다. 도구 자동 재전송은 없다.
3. 여러 proxy의 동시 시작을 Windows 파일 잠금으로 직렬화했다. 준비되지 않은 자식은 시작 timeout에서 종료하고 기다린 뒤 잠금을 해제한다. PID 파일 정리는 소유 프로세스만 수행한다.
4. 동시 sync는 asyncio.Lock에서 비동기로 기다린 뒤 파일 writer 잠금을 잡는다. 대기 요청이 event loop를 막지 않는다.
5. Windows trust-store 로딩이 새 HTTP 클라이언트마다 약 403ms를 추가했다. 검증된 SSL context만 한 번 생성해 재사용한다. SDK의 독립 세션과 자동 era mirroring은 유지된다. 실제 모델 proxy warm 호출은 약 430ms에서 약 67ms로 감소했다.
6. 실사용 Vault의 초기 sync가 기존 60초 시작 제한을 넘었다. 시작 제한을 별도 `KNOWLEDGE_DAEMON_START_TIMEOUT`, 기본 900초로 조정했다. 일반 도구 제한은 기존 15초다. 긴 sync에는 proxy와 client의 도구 제한도 별도로 조정해야 한다.
7. 같은 시험 프로세스에서 Uvicorn 서버를 반복 실행하면 sse-starlette의 전역 shutdown watcher가 다음 서버의 legacy SSE 응답을 닫았다. 시험 harness에서만 automatic drain을 조정했다. 운영 SSE 처리는 변경하지 않았다.

## 검증 근거

- 구환경 전체: 122 passed, Authlib deprecation warning 1건.
- 새환경 최종 전체: 148 passed (286.19초). 시작 deadline 보완 후 관련 daemon/CLI/proxy 32 passed.
- HTTP JSON 및 요청별 SSE, 실제 subprocess stdio proxy, standalone, daemon entrypoint의 modern/legacy 경로 검증.
- daemon 직접 HTTP의 세 도구, 초기 dependencies 생성/warmup 각 1회, 동시 sync 직렬화 검증.
- 헤더/body protocol 일치, Mcp-Method/Mcp-Name, session 없는 modern 요청, 누락·미지원 version·Origin 거절 검증.
- provider 실패를 빈 도구 목록으로 은폐하지 않음, timeout 이후 정상 요청, 동시 요청 중 취소, 재전송 없음, stdio 종료 검증.
- 구·신 FakeStore 검색의 TextContent는 정확히 일치. schema의 required/query 및 nullable/filter/default/model 계약 비교.
- 실제 모델로 임시 Vault와 별도 collection의 index → 한글 검색/rerank → 변경 sync → unchanged sync → status 검증. rebuild 미사용.
- constraints로 별도 환경에 실제 재설치 성공, 양쪽 새 환경에서 pip check 성공.
- `FASTMCP_MCP_CAMELCASE_COMPAT=false`에서도 protocol/관련 검사 통과. 원격 update/telemetry 설정 off, 배너 off.
- 독립 리뷰의 startup/cancel 문제를 수정하고 재검토했으며 최종 Critical/Important 코드 지적은 없다.

로그·벤치마크 원본은 구현 checkout의 `.knowledge/migration/fastmcp4/`에 보관한다. snapshot, SQLite 백업, 기존 client 설정은 원래 checkout의 같은 경로에 보관한다. 개인 설정은 Git에 포함하지 않는다.

## 실사용 성능

같은 머신의 기존 Vault에 대해 공개 검색어 `프로젝트`, limit=2, 실제 기본 stdio 경로를 사용했다. 각 모드/옵션 8회 중 첫 호출을 제외한 warm p50이다. 표본이 작아 장기 성능 보장은 아니다.

| 경로 | Protocol | 검색 p50 | rerank p50 |
|---|---|---:|---:|
| 구환경 auto | 2025-11-25 | 60.38ms | 515.14ms |
| 새환경 auto | 2026-07-28 | 59.24ms | 523.51ms |
| 새환경 legacy | 2025-11-25 | 60.63ms | 531.57ms |
| 롤백 auto | 2025-11-25 | 61.57ms | 528.53ms |
| 최종 복귀 auto | 2026-07-28 | 66.53ms | 531.33ms |
| 최종 복귀 legacy | 2025-11-25 | 63.68ms | 522.51ms |

초기 단일 노트 시험에서는 일부 p50/p95가 20% 이상 증가했으므로 즉시 전환하지 않고 client 생성 비용을 분석했다. SSL context 보완 후 실제 Vault에서는 modern 검색 -1.9%, rerank +1.6%로 관찰됐다. 최종 복귀 시점의 p50은 baseline 대비 검색 +10.2%, rerank +3.1%였다. 시험 모델의 CUDA allocated는 구·신 모두 약 4,341MiB, RSS는 약 1,954/1,980MiB였다. proxy의 TLS context가 추가되므로 고정 15MB RSS는 주장하지 않는다.

## 운영 전환과 롤백

- 원래 checkout/환경을 유지하고 별도 managed worktree의 `.venv`를 설치·사용했다. 현재 main의 구환경을 덮어쓰지 않았다.
- Codex config와 확인된 Antigravity 설정 두 곳의 stdio 명령을 새 환경으로 변경했다. `/sse` 직접 client 설정은 inventory에서 발견되지 않았다.
- SQLite backup API로 일관된 백업, Qdrant collection snapshot, client/env byte-copy 백업을 확보했다.
- 구 proxy/daemon 종료 후 새 daemon을 시작하고 discovery·세 도구·실제 검색·rerank·status를 검증했다.
- 구 client/env 설정 및 구 daemon으로 실제 롤백하여 같은 검색과 status, 5,265개 point 보존을 확인했다. 데이터/manifest를 삭제하거나 되감지 않았다.
- 최신 설정과 `/mcp` daemon으로 다시 복귀했다. 최종 재확인 로그는 `live-final.json`에 기록한다.
- 최초 baseline의 point count는 5,240, 새 incremental sync 후 5,265였다. 초기 sync는 added=3, changed=5, deleted=0, unchanged=355, skipped=2였다. 기존 parse_failed 3건 때문에 status=partial은 남아 있다. 이를 이번 migration 성공으로 숨기지 않는다.

Codex CLI `0.158.0-alpha.2.1`, Claude Code `1.0.117`, Antigravity `2.17.0.0`을 확인했다. Claude Code에서 해당 server 구성은 찾지 못했으므로 추정해 추가하지 않았다. 현재 채팅의 구 MCP 연결은 종료돼 `Transport closed`가 확인됐다. Codex/Antigravity의 MCP 연결 재시작이 필요하다. 위 protocol 성공은 실제 설정의 명령을 실행한 SDK client 및 시험 transport의 근거이며, 아직 재접속하지 않은 UI client의 협상 결과를 주장하지 않는다.

이 worktree는 현재 client 실행 경로이므로 운영 전환이 다른 경로로 완료되기 전에는 archive하지 않는다. main merge/push는 수행하지 않았다.

## 기준 문서

- [FastMCP 4.0.10](https://pypi.org/project/fastmcp/4.0.10/)
- [FastMCP 4 upgrade guide](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3)
- [공식 proxy](https://gofastmcp.com/servers/providers/proxy)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Python SDK 2.2.0 migration](https://github.com/modelcontextprotocol/python-sdk/blob/v2.2.0/docs/migration.md)
