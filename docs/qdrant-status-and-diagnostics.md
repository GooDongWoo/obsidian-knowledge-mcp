# Qdrant 상태 진단 및 색인 모니터링 가이드

<a id="native-qdrant-migration"></a>
## Windows 네이티브 Qdrant 이행과 롤백

### 실행 파일과 저장소

[공식 v1.19.1 Windows 릴리스](https://github.com/qdrant/qdrant/releases/tag/v1.19.1)의 `qdrant-x86_64-pc-windows-msvc.zip`을 사용합니다. 검증된 archive SHA-256은 `9b6f69bd85f6abed4bc13f943099f55c6ffd55f5dd90388635320d8fbb569eb0`입니다. `Get-FileHash -Algorithm SHA256 <archive>`로 확인하고 도구 폴더에 압축을 풉니다.

`KNOWLEDGE_QDRANT_EXECUTABLE`에는 실행 파일의 절대 경로를 지정합니다. PATH 조회도 지원하지만 무인 실행에는 절대 경로를 권장합니다. localhost health가 정상이면 기존 서버를 자동으로 재사용합니다. 일반 CLI 실행에는 서비스 등록이 필요하지 않습니다.

`KNOWLEDGE_QDRANT_NATIVE_STORAGE`는 새 네이티브 저장소의 절대 경로입니다. 생략하면 사용자 홈의 `.knowledge-qdrant/<프로젝트 경로 SHA-256 앞 12자리>`를 사용합니다. Windows의 깊은 checkout/임시 폴더 아래에서는 Gridstore가 긴 경로 오류를 낼 수 있으므로 짧은 경로를 사용하세요. 런처는 빈 저장소에만 `.knowledge-native-owner.json`을 만들고 재시작 시 확인합니다. 표식 없는 nonempty 저장소는 거부합니다. Docker 폴더를 이름 변경하거나 표식을 직접 만들어 우회하지 마세요.

기존 `KNOWLEDGE_QDRANT_STORAGE`는 Docker bind mount 전용입니다(기본 `.knowledge/qdrant`). 두 환경이 저장소를 공유하지 않습니다. 런처는 loopback, 절대 storage/snapshots/tmp 경로, telemetry off를 설정합니다. 네이티브 저장소의 `native.log`와 `native.previous.log`는 현재/직전 실행의 WARN 이상 진단을 보관합니다. 일반 런처 로그는 시작할 때 교체되며 실행 중 크기 회전은 하지 않습니다. 상시 서비스는 아래 WinSW 회전을 사용하세요.

### 백업과 별도 포트 복원

원본 Docker 6333과 MCP 8765를 유지한 리허설에서는 snapshot 생성과 SQLite backup API를 같은 writer lock 범위에서 실행합니다. 운영 runtime에 새 `OperationLog`를 생성하면 DB 마이그레이션이 실행될 수 있으므로 사용하지 않습니다. 다음 예시의 경로를 실제 원본/새 백업 경로로 바꿉니다. 백업은 private 문서와 과거 질의 기록을 포함할 수 있으므로 접근을 제한하고 Git/공유 로그에 넣지 않습니다.

```python
from pathlib import Path
from contextlib import closing
import json, sqlite3
import httpx
from knowledge_mcp.state import index_lock

live = Path("C:/Path/To/original-project/.knowledge")
backup = Path("C:/Users/YourName/qdrant-backup/2026-10-03")
backup.mkdir(parents=True, exist_ok=False)
name = "obsidian_knowledge_bge_m3_ko_v1"
with index_lock(live), httpx.Client(base_url="http://127.0.0.1:6333", timeout=600) as client:
    with closing(sqlite3.connect((live / "state.sqlite3").as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(backup / "state.sqlite3")) as target:
            source.backup(target)
            assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    aliases = client.get("/aliases")
    aliases.raise_for_status()
    (backup / "aliases.json").write_text(json.dumps(aliases.json()), encoding="utf-8")
    response = client.post(f"/collections/{name}/snapshots")
    response.raise_for_status()
    snapshot = response.json()["result"]["name"]
    with client.stream("GET", f"/collections/{name}/snapshots/{snapshot}") as response:
        response.raise_for_status()
        with (backup / "collection.snapshot").open("xb") as output:
            for block in response.iter_bytes():
                output.write(block)
```

[Snapshot 호환 조건](https://qdrant.tech/documentation/operations/snapshots/)은 같은 minor 또는 다음 minor 버전 복원입니다. 여기서는 1.19.1 → 1.19.1을 검증합니다. Collection snapshot에는 alias가 없으므로 별도 목록을 확인하고 필요한 alias를 대상에 재생성합니다. [복원 디스크 요구량](https://qdrant.tech/documentation/migration-recovery-options/)에 따라 collection 크기의 약 2배 여유 공간을 확보하세요.

리허설은 새 빈 저장소와 별도 포트 16333/16334를 사용합니다. `Settings.from_env()`의 운영 URL 정책은 유지하고 시험 코드에서만 포트를 바꿉니다.

```python
from dataclasses import replace
from pathlib import Path
from knowledge_mcp.config import Settings
from knowledge_mcp.cli import ensure_qdrant

settings = replace(
    Settings.from_paths(vault_root=Path("C:/test/vault"), project_root=Path("C:/test/project")),
    qdrant_url="http://127.0.0.1:16333",
    qdrant_executable="C:/Tools/qdrant-1.19.1/qdrant.exe",
    qdrant_native_storage=Path("C:/Users/YourName/qdrant-rehearsal"),
)
ensure_qdrant(settings)
```

```powershell
curl.exe --fail -X POST "http://127.0.0.1:16333/collections/obsidian_knowledge_bge_m3_ko_v1/snapshots/upload?priority=snapshot&wait=true" -F "snapshot=@C:/Users/YourName/qdrant-backup/2026-10-03/collection.snapshot"
$env:KNOWLEDGE_TEST_QDRANT_URL = "http://127.0.0.1:16333"
python -m pytest tests/test_qdrant_integration.py -q
```

새 collection 복원은 `priority=snapshot`을 명시합니다. 복원 후 exact count, dense 1024/Cosine, BM25 IDF, collection metadata, payload index 7개, alias, ID/세대/본문·벡터 일치, 한글 dense/BM25/RRF 및 실제 rerank 적용·순서를 비교합니다. 결과 원문 대신 개수와 일치 여부만 기록하세요. GPU 여유가 작으면 실제 질의 벡터를 별도 프로세스에서 계산하고 종료한 뒤 리랭커를 로드합니다. 증분 sync는 별도 UUID collection과 합성 Vault에서 검증합니다. 운영 복제 collection/manifest를 합성 파일만으로 sync하면 원본 데이터가 삭제될 수 있습니다.

### Snapshot 실패 시 API 복제 검증

2026-10-03 리허설에서는 snapshot의 `0/wal/first-index`가 18개의 NUL byte여서 네이티브 1.19.1이 WAL JSON 오류로 복원을 거부했습니다. [관련 upstream 보고](https://github.com/qdrant/qdrant/issues/7956)가 있지만 이 설치에서의 해결은 확인되지 않았습니다. 버전 일치만으로 snapshot 복원 가능성을 보장하지 못합니다. 원본 snapshot/hash를 보존하고 WAL을 직접 고치지 않습니다. 이 경로의 운영 전환은 복원 검증을 통과할 때까지 보류합니다.

대안은 [공식 migration 문서의 stream/upsert 방식](https://qdrant.tech/documentation/migration-recovery-options/)입니다. writer lock 아래 `scroll(limit=128, with_payload=True, with_vectors=True)`를 offset이 없어질 때까지 호출하여 ID·payload·dense/sparse vector를 로컬 JSONL로 보존합니다. 같은 lock 범위에서 read-only SQLite backup, exact count, collection config/index/alias 및 비교용 검색 결과를 기록합니다. 출력에는 본문을 포함하지 않습니다.

대상은 또 다른 새 owned 저장소여야 합니다. 원본 `get_collection().config`에서 vectors, sparse_vectors, shard/replication/write_consistency, on_disk_payload, HNSW, optimizer, WAL, quantization, metadata 설정을 `create_collection()`에 전달하고 payload index를 원래 타입으로 만듭니다. 클라이언트의 HNSW/optimizer/WAL 생성 인수는 Diff 모델을 받으므로 조회한 전체 모델은 `.model_dump(exclude_none=True)`로 변환해 전달합니다. 저장된 각 레코드로 `models.PointStruct(id=record["id"], vector=record["vector"], payload=record["payload"])`를 만들어 `upsert(..., wait=True)`에 64개씩 전달합니다. 재임베딩하지 않으며 HNSW 재구축 시간/자원이 추가로 필요합니다. [Scroll API](https://api.qdrant.tech/api-reference/points/scroll-points), [Upsert API](https://api.qdrant.tech/api-reference/points/upsert-points)를 따릅니다.

복제 후 위 검증을 모두 수행합니다. Cosine 벡터는 upsert 때 정규화될 수 있으므로 float32 오차가 있으면 최대 절대 차이와 검색 점수·순서를 함께 검증하고 허용 오차를 기록합니다. ID/payload/세대와 sparse 정보는 별도로 정확히 비교합니다. API 복제가 성공해도 snapshot 실패가 해결된 것으로 기록하지 않습니다.

### 검증 후 전환 및 롤백

1. MCP와 모든 writer를 중지하고 최종 백업 쌍을 보존합니다. 원본 checkout/runtime/환경은 그대로 두고 새 runtime은 백업 SQLite의 복사본을 사용합니다.
2. 검증한 방법으로 최종 데이터를 새 네이티브 저장소에 복원하고 비교를 다시 통과시킵니다. 오래된 리허설 백업으로 바로 전환하지 않습니다.
3. 리허설 프로세스를 종료하고 기존 서비스가 실행 중이라면 중지합니다.
4. 새 checkout 환경에 executable/native storage를 명시하고 네이티브 6333/6334, MCP 8765를 시작합니다. 일반 런처라면 실행 파일/저장소를 확인한 해당 PID만 관리합니다.
5. health, count, 검색/rerank, modern/legacy 도구 목록, 색인 상태를 재확인합니다. `vmmemWSL`, 전체 프로세스 RSS, native RSS, 기동/증분 색인 시간을 같은 조건에서 비교합니다. 다른 WSL 작업의 메모리까지 제거됐다고 해석하지 않습니다.

복원/동작 검증 실패 시 새 MCP/네이티브 프로세스를 중지하고, 백업본을 새로운 클린 네이티브 저장소에 직접 복원하여 복구합니다. 전환 후 변경분이 있다면 양쪽 백업을 먼저 보존하고 재동기화 범위를 결정합니다. 다른 시점의 SQLite와 Qdrant를 임의로 섞지 않습니다.

### 선택적 Windows 자동 시작 서비스

자동 시작이 필요할 때만 [WinSW 2.12.0 공식 릴리스](https://github.com/winsw/winsw/releases/tag/v2.12.0)의 x64 wrapper를 `C:/Tools/qdrant-service/QdrantService.exe`로 저장하고 같은 이름의 XML을 만듭니다. [2.12 XML 설정](https://github.com/winsw/winsw/blob/v2.12.0/doc/xmlConfigFile.md)과 [로그 회전](https://github.com/winsw/winsw/blob/v2.12.0/doc/loggingAndErrorReporting.md)을 사용하는 예이며 서비스 설치는 별도 운영 단계입니다.

```xml
<service>
  <id>KnowledgeQdrant</id>
  <name>Knowledge Qdrant</name>
  <executable>C:/Tools/qdrant-1.19.1/qdrant.exe</executable>
  <arguments>--config-path C:/Tools/qdrant-service/qdrant.yaml</arguments>
  <workingdirectory>C:/Tools/qdrant-service</workingdirectory>
  <startmode>Automatic</startmode>
  <delayedAutoStart>true</delayedAutoStart>
  <serviceaccount><domain>NT AUTHORITY</domain><user>LocalService</user></serviceaccount>
  <logpath>C:/ProgramData/KnowledgeQdrant/logs</logpath>
  <log mode="roll-by-size"><sizeThreshold>1024</sizeThreshold><keepFiles>3</keepFiles></log>
  <stoptimeout>60sec</stoptimeout>
  <onfailure action="restart" delay="10 sec"/>
</service>
```

```yaml
# qdrant.yaml: 위에서 초기화·복원·검증한 동일 저장소를 지정합니다.
storage:
  storage_path: C:/Users/YourName/qdrant-native
  snapshots_path: C:/Users/YourName/qdrant-native/snapshots
  temp_path: C:/Users/YourName/qdrant-native/tmp
service:
  host: 127.0.0.1
  http_port: 6333
  grpc_port: 6334
telemetry_disabled: true
log_level: WARN
```

LocalService 또는 별도의 제한된 계정에 실행 파일/설정은 읽기·실행, storage/snapshots/tmp/logs에는 수정 권한만 부여합니다. Vault 권한은 필요하지 않습니다. 예를 들어 관리자 셸에서 `icacls "C:/Tools/qdrant-service" /grant "*S-1-5-19:(OI)(CI)RX"`, `icacls "C:/Users/YourName/qdrant-native" /grant "*S-1-5-19:(OI)(CI)M"`을 적용하고 실행 파일 폴더/로그 폴더에도 각각 RX/M을 부여합니다. 상위 경로 통과 권한을 확인하고 LocalSystem 기본값에 맡기지 않습니다.

기존 포트 사용 프로세스를 해당 절차에 따라 중지한 뒤 관리자 셸에서 등록합니다. 제거 시 데이터는 보존합니다.

```powershell
& 'C:/Tools/qdrant-service/QdrantService.exe' install
& 'C:/Tools/qdrant-service/QdrantService.exe' start
& 'C:/Tools/qdrant-service/QdrantService.exe' status
# 복구/제거 후 수동 프로세스로 전환하거나 백업을 복원합니다.
& 'C:/Tools/qdrant-service/QdrantService.exe' stop
& 'C:/Tools/qdrant-service/QdrantService.exe' uninstall
```

이 문서는 `obsidian-knowledge-mcp`의 백엔드 저장소인 Qdrant와 상태 추적 데이터베이스(`state.sqlite3`)의 동작 원리, 상태 키워드, REST API 진단 방법, 그리고 미완료 파일 추적 기법을 다룹니다.

---

## 1. 색인 상태 키워드 및 오류 코드

인덱싱 작업(`knowledge-mcp index`, `rebuild`, `knowledge-index-sync`) 또는 `knowledge-index-status` 도구 호출 시 반환되는 주요 상태 및 오류 코드의 의미는 다음과 같습니다.

### 1.1 실행 상태 (`status`)

| 상태 키워드 | 의미 및 진단 |
| :--- | :--- |
| `completed` | 대상 파일 전체가 정상적으로 처리되어 Qdrant 및 SQLite에 색인이 완료됨. |
| `partial` | 대상 문서 중 일부 파일이 파싱 실패 또는 텍스트 미추출 등으로 색인되지 못했으나, 나머지 정상 파일은 모두 색인됨. (가장 흔하게 발생하며, 원인 파일 확인 후 조치 가능) |
| `failed` | 데이터베이스 연결 불가, Qdrant 데몬 비정상 등 치명적 오류로 인해 인덱싱 작업 전체가 중단됨. |

### 1.2 세부 오류 코드 (`error_code`) 및 스킵 사유

| 코드 / 키워드 | 분류 | 상세 설명 및 대응 방법 |
| :--- | :--- | :--- |
| `parse_failed` | 오류 (Error) | 파일 파싱에 실패함. 주로 손상된 PDF 파일, 암호화된 PDF, 또는 YAML 문법 오류가 있는 Markdown frontmatter에서 발생합니다. |
| `skipped_no_text` | 스킵 (Skipped) | PDF 파일에 추출 가능한 텍스트 레이어가 없음 (스캔 이미지 PDF 등). OCR을 수행하지 않는 정책에 따라 정상적으로 스킵 처리되며 색인 완료로 간주됩니다. |
| `skipped_large_file` | 스킵 (Skipped) | 파일 크기가 안전 상한선인 **30MB**를 초과함. 시스템 메모리 및 VRAM 고갈을 방지하기 위해 본문 로드 없이 스킵됩니다. |
| `unchanged` | 정상 (Unchanged) | 파일의 내용 해시(Hash)가 변경되지 않아 재색인을 건너뜀. |
| `added` | 완료 (Added) | 새로 추가된 파일이 성공적으로 청크화 및 임베딩되어 Qdrant에 저장됨. |
| `changed` | 완료 (Changed) | 내용이 수정된 파일의 이전 세대 청크가 삭제되고 새 청크로 교체됨. |
| `deleted` | 완료 (Deleted) | Vault에서 삭제된 파일의 청크가 Qdrant 및 SQLite 기록에서 완전히 제거됨. |

> [!NOTE]
> `point_count`는 **청크(Chunk)의 수**이며 원본 파일의 수가 아닙니다. 한 개의 긴 문서는 여러 개의 청크로 분할되어 저장됩니다.

---

## 2. Qdrant REST HTTP API 진단

Qdrant는 기본적으로 로컬 포트 `6333`(HTTP REST)과 `6334`(gRPC)에 바인딩됩니다. PowerShell에서 `Invoke-RestMethod`를 사용하여 Qdrant 자체의 상태를 직접 검사할 수 있습니다.

### 2.1 기본 헬스체크 및 컬렉션 목록
```powershell
# 1. Qdrant 서비스 헬스체크 (정상 시 {"title":"qdrant - vector search engine",...} 반환)
Invoke-RestMethod 'http://127.0.0.1:6333/healthz'

# 2. 존재하는 모든 컬렉션 확인
Invoke-RestMethod 'http://127.0.0.1:6333/collections'
```

### 2.2 컬렉션 상세 정보 및 저장된 청크 수 확인
현재 주로 사용되는 컬렉션은 `obsidian_knowledge_bge_m3_ko_v1` (BGE-m3-ko)입니다.
```powershell
# 컬렉션 상태, 벡터 설정 및 포인트 수 확인
$info = Invoke-RestMethod 'http://127.0.0.1:6333/collections/obsidian_knowledge_bge_m3_ko_v1'
$info.result | Select-Object status, points_count, indexed_vectors_count

# 정확한 포인트(청크) 개수 카운트
$count = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:6333/collections/obsidian_knowledge_bge_m3_ko_v1/points/count' `
  -ContentType 'application/json' `
  -Body '{"exact":true}'
$count.result.count
```

- `status=green`: Qdrant 컨테이너 및 컬렉션 엔진이 건강함을 의미합니다. (개별 파일 인덱싱 성공 여부와는 무관)
- `points_count`: 저장된 총 벡터 청크 수입니다.

### 2.3 페이로드(Payload) 구조 검사 (Scroll API)
Qdrant에 저장된 실제 청크 메타데이터(제목, 경로, 태그, 보안수준 등)를 확인하려면 읽기 전용 `scroll` 요청을 사용합니다.
```powershell
$body = @'
{
  "filter": {
    "must": [
      { "key": "metadata.security_level", "match": { "value": "public" } }
    ]
  },
  "limit": 2,
  "with_payload": true,
  "with_vector": false
}
'@

$sample = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:6333/collections/obsidian_knowledge_bge_m3_ko_v1/points/scroll' `
  -ContentType 'application/json' `
  -Body $body

$sample.result.points | Select-Object id, payload
```

> [!CAUTION]
> Qdrant HTTP API는 MCP 레벨의 `include_private=false` 필터를 강제하지 않습니다. 로컬 포트에 접근할 수 있는 요청은 private 청크도 직접 조회할 수 있으므로, Qdrant 포트(6333, 6334)를 외부에 노출하지 마세요.

---

## 3. SQLite 상태 데이터베이스 (`state.sqlite3`) 구조

프로젝트의 `.knowledge/state.sqlite3`는 파일 수준의 동기화 상태와 실행 이력을 관리합니다.

### 3.1 스키마 개요

```text
┌──────────────────────────────────────────────────────────┐
│ collection_files                                         │
├───────────────────┬──────────────────────────────────────┤
│ collection_name   │ 컬렉션 식별자 (예: ...bge_m3_ko_v1)    │
│ path              │ Vault 상대 경로                      │
│ content_hash      │ SHA256 내용 해시 (증분 비교용)       │
│ point_count       │ 해당 파일에서 추출된 청크 수         │
│ updated_at        │ 색인 완료 시각                       │
└───────────────────┴──────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ index_runs                                               │
├───────────────────┬──────────────────────────────────────┤
│ id                │ 실행 고유 ID                         │
│ started_at        │ 작업 시작 시각                       │
│ completed_at      │ 작업 완료 시각                       │
│ collection_name   │ 대상 컬렉션                          │
│ status            │ completed / partial / failed         │
│ added / changed   │ 추가 / 수정 / 삭제 / 스킵된 파일 수  │
│ deleted / skipped │                                      │
│ failed            │ 실패 파일 수                         │
│ error_code        │ 첫 번째 발생 오류 코드               │
└───────────────────┴──────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ query_logs                                               │
├───────────────────┬──────────────────────────────────────┤
│ id / timestamp    │ 질의 발생 시각                       │
│ query / filters   │ 검색어 및 필터 조건                  │
│ result_count      │ 반환된 결과 청크 수                  │
│ client_name       │ 요청한 MCP 클라이언트 (codex 등)     │
│ elapsed_ms        │ 검색 소요 시간 (밀리초)              │
└───────────────────┴──────────────────────────────────────┘
```

---

## 4. 진행 현황 및 미색인 파일 추적 스크립트

현재 색인 진행률과 색인에 실패하거나 누락된 파일 목록을 정확히 파악하려면 다음 PowerShell/Python 명령을 사용합니다.

### 4.1 모델별 완료 기록 및 최근 실행 요약
```powershell
$vault = $env:KNOWLEDGE_VAULT_ROOT
$project = $env:KNOWLEDGE_PROJECT_ROOT
$python = Join-Path $project '.venv\Scripts\python.exe'
$db = Join-Path $project '.knowledge\state.sqlite3'

& $python -c @'
import sqlite3, sys
db_path = sys.argv[1]
c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

print("=== 모델별 색인 완료 통계 ===")
for row in c.execute("SELECT collection_name, COUNT(*), COALESCE(SUM(point_count),0) FROM collection_files GROUP BY collection_name"):
    print(f"컬렉션: {row[0]} | 파일 수: {row[1]}개 | 총 청크: {row[2]}개")

print("\n=== 최근 3회 색인 실행 기록 ===")
for row in c.execute("SELECT started_at, collection_name, status, added, changed, deleted, failed, error_code FROM index_runs ORDER BY id DESC LIMIT 3"):
    print(f"시작: {row[0]} | {row[1]} | 상태: {row[2]} | 추가:{row[3]} 수정:{row[4]} 삭제:{row[5]} 실패:{row[6]} | 오류코드: {row[7]}")
'@ $db
```

### 4.2 미색인(미완료) 파일 목록 및 누락 원인 파악
어떤 파일이 아직 색인되지 못했는지(예: `parse_failed`가 발생한 원인 파일) 정확한 상대 경로를 추출합니다.

```powershell
@'
import sqlite3, sys
from pathlib import Path
from knowledge_mcp.config import Settings
from knowledge_mcp.documents import discover_sources

vault, project = map(Path, sys.argv[1:3])
settings = Settings.from_paths(vault_root=vault, project_root=project)

# 실제 Vault에서 수집된 소스 파일 목록
sources = {p.relative_to(vault).as_posix() for p in discover_sources(settings)}

db = sqlite3.connect(f'file:{settings.runtime_dir / "state.sqlite3"}?mode=ro', uri=True)
collections = ('obsidian_knowledge_bge_m3_ko_v1',)

for collection in collections:
    completed = {row[0] for row in db.execute(
        'SELECT path FROM collection_files WHERE collection_name = ?', (collection,)
    )}
    uncompleted = sources - completed
    print(f"[{collection}]")
    print(f"  - 발견된 총 대상 파일: {len(sources)}개")
    print(f"  - 색인 완료 파일: {len(sources & completed)}개")
    print(f"  - 미완료 파일: {len(uncompleted)}개")
    
    if uncompleted:
        print("  - 미완료 파일 목록:")
        for path in sorted(uncompleted):
            print(f"      * {path}")
'@ | & $python - $vault $project
```

---

## 5. 트러블슈팅 및 장애 복구 가이드

### 5.1 `parse_failed`가 반복되는 경우
1. 위 4.2 스크립트로 미완료 파일 경로를 확인합니다.
2. 해당 파일이 PDF인 경우:
   - PDF가 암호로 보호되어 있는지, 혹은 텍스트가 완전히 깨져 있는지 확인합니다.
   - 필요 시 해당 파일을 `.knowledgeignore`에 등록하여 색인 대상에서 제외할 수 있습니다.
3. 해당 파일이 Markdown인 경우:
   - 파일 시작 부분의 `---` YAML frontmatter가 올바른 문법으로 닫혀 있는지 확인합니다.
4. 문제를 해결한 후 다시 `knowledge-mcp index`를 실행하면 실패했던 파일만 재시도합니다.

### 5.2 Qdrant 데이터 디렉터리 및 SQLite 직접 삭제 금지
- `.knowledge/qdrant` 폴더나 `state.sqlite3`를 수동으로 삭제하면, Qdrant 내부의 세대 ID(generation)와 SQLite의 파일 해시가 불일치하여 색인 불일치가 발생할 수 있습니다.
- 처음부터 다시 색인하려면 반드시 CLI 명령인 `knowledge-mcp rebuild`를 사용하세요.

### 5.3 `.knowledge/index.lock` 파일이 남아있는 경우
- `.knowledge/index.lock` 파일 자체는 운영체제 수준의 파일 잠금(file lock) 핸들을 잡기 위한 앵커 파일입니다.
- 프로세스가 비정상 종료되더라도 운영체제가 프로세스 종료 시 파일 핸들을 자동으로 반환하므로, **파일이 디스크에 남아있다고 해서 잠겨있는 것이 아닙니다**.
- 실행 중인 다른 `knowledge-mcp` 프로세스가 없는 것이 확실하다면 안심하고 새 작업을 시작할 수 있습니다.
