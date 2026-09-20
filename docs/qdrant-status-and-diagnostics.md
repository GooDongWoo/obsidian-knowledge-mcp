# Qdrant 상태 진단 및 색인 모니터링 가이드

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

Vault의 `.knowledge/state.sqlite3`는 파일 수준의 동기화 상태와 실행 이력을 관리합니다.

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
$db = Join-Path $vault '.knowledge\state.sqlite3'

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
