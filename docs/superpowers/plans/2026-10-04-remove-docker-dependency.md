# Docker 의존성 및 레거시 롤백 코드 완전 제거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `obsidian-knowledge-mcp` 프로젝트에서 Windows 네이티브 Qdrant 전환 완료에 따라 더 이상 사용되지 않는 `docker-compose.yml`, Docker 서브프로세스 롤백 코드, 레거시 스토리지 환경변수 설정, 관련 단위 테스트 및 문서를 안전하게 완전 제거한다.

**Architecture:** Qdrant 기동 및 관리를 오직 localhost의 기존 healthy 서버 재사용과 로컬 네이티브 바이너리 실행(`_start_native`) 단일 경로로 단순화한다. Docker Compose 실행 서브프로세스, 레거시 `qdrant_docker_storage` 설정, `qdrant_backend` 분기를 정리하고, 순수 네이티브 단일 아키텍처로 통일한다.

**Tech Stack:** Python 3.11+, FastMCP 4.0.10, Qdrant 1.19.1 (Native Windows 바이너리), pytest.

**Spec:** 사용자 요구사항 (도커 롤백 불필요 결정 및 관련 부분 전면 삭제)

## Global Constraints

- Vault 원문 및 개인 문서는 변경하거나 외부로 유출하지 않는다.
- Qdrant 포트(`127.0.0.1:6333`, gRPC `6334`)와 네이티브 저장소 관리 계약(`.knowledge-native-owner.json`)은 변경하지 않는다.
- 기존 네이티브 Qdrant의 자동 기동(`_start_native`), 건강 상태 체크(`_healthy`), 프로세스 정리 로직은 온전히 유지한다.
- 모든 단위 테스트 및 기존 검색/동기화 기능이 통과해야 한다.

---

### Task 1: `docker-compose.yml` 및 로컬 환경변수 파일 정리

**Files:**
- Delete: `docker-compose.yml`
- Modify: `.env:4,8`
- Modify: `.env.example:11-20`

**Interfaces:**
- Consumes: 없음
- Produces: Docker Compose 파일이 없는 순수 네이티브 환경 설정 템플릿

- [ ] **Step 1: `docker-compose.yml` 파일 삭제**

Run in PowerShell:
```powershell
Remove-Item -Path "docker-compose.yml" -Force
```

- [ ] **Step 2: `.env.example`에서 레거시 Docker 롤백 설정 주석 및 backend 설정 제거**

`.env.example`의 다음 블록:
```ini
# Qdrant: reuse a healthy localhost service first; otherwise start native by default.
KNOWLEDGE_QDRANT_BACKEND=native
# Absolute executable path is recommended, particularly for unattended startup.
# KNOWLEDGE_QDRANT_EXECUTABLE="C:/Tools/qdrant-1.19.1/qdrant.exe"
# Native default: %USERPROFILE%/.knowledge-qdrant/<project-path-hash>
# Use a short absolute path on Windows. Only empty or native-owned storage is accepted.
# KNOWLEDGE_QDRANT_NATIVE_STORAGE="C:/Users/YourName/qdrant-native"
# Existing Docker storage setting is preserved for explicit backend=docker rollback.
# KNOWLEDGE_QDRANT_STORAGE="C:/Path/To/obsidian-knowledge-mcp/.knowledge/qdrant"
```
을 아래와 같이 네이티브 전용 설정으로 교체:
```ini
# Qdrant: reuse a healthy localhost service first; otherwise start local native binary.
# Absolute executable path is recommended, particularly for unattended startup.
# KNOWLEDGE_QDRANT_EXECUTABLE="C:/Tools/qdrant-1.19.1/qdrant.exe"
# Native storage default: %USERPROFILE%/.knowledge-qdrant/<project-path-hash>
# Use a short absolute path on Windows. Only empty or native-owned storage is accepted.
# KNOWLEDGE_QDRANT_NATIVE_STORAGE="C:/Users/YourName/qdrant-native"
```

- [ ] **Step 3: 로컬 `.env`에서 레거시 `KNOWLEDGE_QDRANT_STORAGE` 및 `KNOWLEDGE_QDRANT_BACKEND` 라인 정리**

`.env`에서 다음 라인 제거:
```ini
KNOWLEDGE_QDRANT_STORAGE="C:/Users/dongwoo/vs_proj/obsidian-knowledge-mcp/.knowledge/qdrant"
KNOWLEDGE_QDRANT_BACKEND="native"
```

- [ ] **Step 4: 파일 삭제 및 변경 사항 확인**

Run in PowerShell:
```powershell
git status
```
Expected: `docker-compose.yml` deleted, `.env.example` modified.

---

### Task 2: `src/knowledge_mcp/config.py`의 Docker 설정 및 분기 정리

**Files:**
- Modify: `src/knowledge_mcp/config.py:66-70, 76-84, 115-118, 158-164`
- Test: `tests/test_qdrant_storage.py`

**Interfaces:**
- Consumes: 없음
- Produces: `Settings` 클래스에서 `qdrant_backend`, `qdrant_docker_storage`, `docker_qdrant_storage_dir` 제거

- [ ] **Step 1: `src/knowledge_mcp/config.py`에서 Docker 관련 필드 및 유효성 검사 제거**

`Settings` 클래스에서:
1. `qdrant_backend: str = "native"` 및 `qdrant_docker_storage: Path | None = None` 필드 삭제.
2. `__post_init__`에서:
   - `if self.qdrant_backend not in ("native", "docker"):` 검증 삭제.
   - `for name in ("qdrant_native_storage", "qdrant_docker_storage"):` 루프를 `if self.qdrant_native_storage is not None:`로 단일화.
3. `@property def docker_qdrant_storage_dir` 프로퍼티 전체 삭제.
4. `from_env()` 메서드에서:
   - `qdrant_backend` 및 `qdrant_docker_storage` 인자 전달 제거.

- [ ] **Step 2: 문법 및 import 에러 확인**

Run in PowerShell:
```powershell
.\.venv\Scripts\python.exe -c "from knowledge_mcp.config import Settings; print('Config OK')"
```
Expected: `Config OK` 출력.

---

### Task 3: `src/knowledge_mcp/qdrant_process.py`의 Docker 서브프로세스 롤백 코드 제거

**Files:**
- Modify: `src/knowledge_mcp/qdrant_process.py:32, 93-105, 116`
- Test: `tests/test_qdrant_storage.py`

**Interfaces:**
- Consumes: Task 2의 간소화된 `Settings`
- Produces: `ensure_qdrant()`에서 Docker Compose 호출 없이 항상 `_start_native(settings)`만 수행하는 단일 런처

- [ ] **Step 1: `ensure_qdrant`에서 Docker 분기 제거 및 단일 네이티브 런처로 통일**

`qdrant_process.py`의 `ensure_qdrant` 내부:
```python
        process, log_path = None, None
        if settings.qdrant_backend == "docker":
            storage = settings.docker_qdrant_storage_dir
            storage.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["KNOWLEDGE_QDRANT_STORAGE"] = storage.as_posix()
            subprocess.run(
                ["docker", "compose", "-f", str(settings.project_root / "docker-compose.yml"), "up", "-d"],
                cwd=settings.project_root, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        else:
            process, log_path = _start_native(settings)
```
을 다음과 같이 단순화:
```python
        process, log_path = _start_native(settings)
```

- [ ] **Step 2: 오류 메시지 및 예외 문자열에서 Docker 관련 문구 정리**

1. line 116:
```python
raise RuntimeError(f"Qdrant did not become healthy; inspect log {log_path or 'docker compose logs qdrant'}. Use a short absolute native storage path on Windows.")
```
을
```python
raise RuntimeError(f"Qdrant did not become healthy; inspect log {log_path}. Use a short absolute native storage path on Windows.")
```
으로 수정.

2. line 32:
```python
raise RuntimeError("Unknown nonempty Qdrant storage: use new empty native storage and restore a snapshot; never adopt Docker data")
```
을
```python
raise RuntimeError("Unknown nonempty Qdrant storage: use new empty native storage and restore a snapshot; never adopt unmanaged data")
```
으로 수정.

- [ ] **Step 3: 문법 및 import 에러 확인**

Run in PowerShell:
```powershell
.\.venv\Scripts\python.exe -c "from knowledge_mcp.qdrant_process import ensure_qdrant; print('Process OK')"
```
Expected: `Process OK` 출력.

---

### Task 4: `tests/test_qdrant_storage.py`의 Docker 관련 테스트 정리 및 검증

**Files:**
- Modify: `tests/test_qdrant_storage.py`

**Interfaces:**
- Consumes: Task 2, Task 3의 네이티브 단일 아키텍처
- Produces: Docker에 의존하지 않는 100% 통과 단위 테스트 스위트

- [ ] **Step 1: 불필요한 Docker 테스트 함수 삭제**

1. `test_native_default_is_short_separate_and_project_specific`:
   - `assert first.docker_qdrant_storage_dir == tmp_path / "a" / ".knowledge" / "qdrant"` 라인 제거.
2. `test_explicit_docker_rollback_uses_old_storage` 함수 전체 삭제.
3. `test_compose_defaults_to_separate_docker_storage` 함수 전체 삭제.
4. `test_custom_legacy_docker_storage_does_not_become_native` 함수 전체 삭제.

- [ ] **Step 2: `test_settings_read_native_and_docker_paths`를 네이티브 전용으로 수정**

함수명을 `test_settings_read_native_paths`로 변경하고, `KNOWLEDGE_QDRANT_STORAGE` 및 `KNOWLEDGE_QDRANT_BACKEND` 관련 코드 제거:
```python
def test_settings_read_native_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path / "v"))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_NATIVE_STORAGE", str(tmp_path / "native"))
    monkeypatch.setenv("KNOWLEDGE_QDRANT_EXECUTABLE", str(tmp_path / "qdrant.exe"))
    config = Settings.from_env("test")
    assert config.qdrant_storage_dir == tmp_path / "native"
    assert config.qdrant_executable == str(tmp_path / "qdrant.exe")
```

- [ ] **Step 3: `test_qdrant_storage.py` 테스트 실행 및 확인**

Run in PowerShell:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qdrant_storage.py -v
```
Expected: All tests PASS.

---

### Task 5: 운영 문서 및 진단 가이드 업데이트

**Files:**
- Modify: `docs/user-guide-ko.md:31, 280-305`
- Modify: `docs/qdrant-status-and-diagnostics.md:86-104`

**Interfaces:**
- Consumes: Task 1~4의 순수 네이티브 아키텍처
- Produces: 사용자 가이드 및 진단 문서에서 Docker 롤백 관련 혼란스러운 레거시 설명 완전 제거

- [ ] **Step 1: `docs/user-guide-ko.md` 수정**

1. 1장 아키텍처 개요 (line 31 부근):
   - `"로컬 Qdrant: 기본적으로 Windows 네이티브 1.19.1을 사용합니다. 정상 실행 중인 로컬 서비스도 재사용하며 Docker 구성은 롤백용으로 보존합니다."`
   - 를 `"로컬 Qdrant: Windows 네이티브 1.19.1 바이너리를 사용합니다. 정상 실행 중인 로컬 서비스(127.0.0.1:6333)도 자동으로 감지하여 재사용합니다."`로 수정.
2. 6장 제목 및 본문 (line 280-305):
   - `## 6. 네이티브 Qdrant와 Docker 롤백` -> `## 6. 네이티브 Qdrant 운영 및 스토리지`로 제목 변경.
   - Docker Compose 수동 기동 명령어 및 `docker ps` 관련 롤백 설명 삭제.

- [ ] **Step 2: `docs/qdrant-status-and-diagnostics.md` 수정**

1. "검증 후 전환 및 롤백" 섹션에서 `docker compose -f ... up -d` 롤백 안내 블록(lines 96-103) 정리.

---

### Task 6: 전체 회귀 테스트 및 잔존 Docker 참조 검증

**Files:**
- Verification only

- [ ] **Step 1: 소스 코드 및 테스트에서 잔존 Docker 참조 검색**

Run in PowerShell:
```powershell
git grep -i "docker" src/ tests/
```
Expected: 기능 코드 및 활성 테스트에 Docker 참조 0건.

- [ ] **Step 2: 전체 단위 테스트 스위트 회귀 검증**

Run in PowerShell:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 전체 테스트 PASS.

- [ ] **Step 3: git diff 최종 확인**

Run in PowerShell:
```powershell
git status
```
Expected: 깨끗하고 명확한 변경 사항 확인.
