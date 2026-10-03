# Qdrant Hardening 단위 8(Linux 호환성) 및 최종 병합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 단위 7까지 완료된 Qdrant 하드닝 작업을 이어받아, 표준 라이브러리(`fcntl`/`msvcrt`) 기반의 고성능 크로스 플랫폼 파일 락과 POSIX 프로세스 수명주기를 완성하고, Windows 및 WSL Linux 전 범위 통합 검증을 거쳐 `main` 브랜치에 안전하게 병합한다.

**Architecture:** 외부 의존성 추가 없이 `sys.platform`을 통해 Windows CRT(`msvcrt`)와 POSIX VFS(`fcntl.flock`)의 비차단(non-blocking) 커널 락을 분기하여 비동기 이벤트 루프와 프로세스 크래시 복구 보증을 유지한다. 서브프로세스 기동(`start_new_session` vs `CREATE_NO_WINDOW`) 및 종료(`SIGTERM` vs `taskkill`)를 플랫폼별로 일관되게 정규화하고, Linux 환경에서는 불필요한 GPU wheel 없이 CPU 전용 프로필로 설치·동작하도록 `pyproject.toml`을 정리한다.

**Tech Stack:** Python 3.11+, FastMCP 4.0.10, MCP SDK 2.2.0, Qdrant 1.19.1 (Windows Native & Linux musl), PyTorch (CPU on Linux, CUDA 12 on Windows), pytest, WSL2 Ubuntu 22.04.

**Spec:** `docs/superpowers/specs/2026-09-30-qdrant-daemon-hardening-design.md`, `docs/superpowers/plans/2026-09-30-qdrant-daemon-hardening.md` (단위 8)

---

## Global Constraints

- **Vault 및 질의 원문 격리:** Vault 문서 내용과 검색어 원문을 로그, 테스트 출력, 보고서에 일절 노출하지 않는다.
- **포트 및 프로토콜 계약 보존:** Qdrant `127.0.0.1:6333` (테스트 시 `16333` 또는 `17333`), MCP 데몬 `127.0.0.1:8765` 및 FastMCP 4 wire 프로토콜 계약을 엄격히 준수한다.
- **Zero-Dependency 락킹:** 서드파티 파일 락 라이브러리(filelock 등)를 추가하지 않고, 파이썬 표준 라이브러리(`msvcrt`, `fcntl`)만을 사용해 성능과 커널 자동 해제 계약을 유지한다.
- **비차단 비동기 폴링 보존:** `async_index_lock`은 스레드 풀 생성 오버헤드 없이 non-blocking 시스템 콜(`LK_NBLCK` / `LOCK_NB`)과 `asyncio.sleep` 협력 폴링 구조를 유지한다.
- **격리된 작업 공간:** 모든 코드 수정 및 테스트는 worktree(`C:/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp`)에서 진행하며, 검증 통과 전까지 `main` 브랜치를 직접 수정하지 않는다.

---

### Task 1: 단위 7 실측 결과 문서 커밋 (Housekeeping)

**Files:**
- Modify: `docs/2026-10-01-qdrant-hardening-results.md`

**Interfaces:**
- Consumes: 단위 7에서 생성된 1,607개 벡터 복제, 한국어 CUDA 리랭킹, Docker vs Native 벤치마크 실측 수치.
- Produces: 클린한 git working tree 상태 (HEAD: Task 7 결과 반영 완료).

- [ ] **Step 1: 작업 디렉터리의 단위 7 결과 diff 확인**
```powershell
git diff docs/2026-10-01-qdrant-hardening-results.md
```
Expected: 단위 7 API clone(1607 points), benchmark 테이블 등 실측 수치 포함 확인.

- [ ] **Step 2: 결과 문서 스테이징 및 커밋**
```powershell
git add docs/2026-10-01-qdrant-hardening-results.md
git commit -m "docs: record unit 7 native Qdrant migration and benchmark results"
```

- [ ] **Step 3: git status가 클린한지 확인**
```powershell
git status
```
Expected: `working tree clean`

---

### Task 2: 표준 라이브러리 크로스 플랫폼 파일 락 구현 (`state.py` & `tests/test_state.py`)

**Files:**
- Modify: `src/knowledge_mcp/state.py`
- Modify: `tests/test_state.py`

**Interfaces:**
- Consumes: `index_lock(runtime_dir, lock_name="index.lock")`, `async_index_lock(runtime_dir)`
- Produces: 
  - `_lock_nonblocking(fd: int) -> None`: Windows `msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)`, POSIX `fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`
  - `_unlock(fd: int) -> None`: Windows `msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)`, POSIX `fcntl.flock(fd, fcntl.LOCK_UN)`
  - Linux에서 `import knowledge_mcp.state` 및 `import knowledge_mcp.cli` 즉시 성공.

- [ ] **Step 1: Linux에서 `msvcrt` 임포트 실패를 확인하는 재현 검증 (RED)**
```powershell
wsl -e sh -c "PYTHONPATH=/mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/src /home/dongwoo/.cache/knowledge-mcp-hardening-venv/bin/python -c 'import knowledge_mcp.cli'"
```
Expected: FAIL with `ModuleNotFoundError: No module named 'msvcrt'`

- [ ] **Step 2: `tests/test_state.py`에 플랫폼 중립 락 테스트 작성**
`tests/test_state.py` 최상단의 `import msvcrt`를 제거하고, 플랫폼별 락 래퍼(`_lock_nonblocking`, `_unlock`)를 테스트하도록 수정.
```python
def test_index_lock_cross_platform_primitives(tmp_path, monkeypatch):
    from knowledge_mcp import state
    called = []
    monkeypatch.setattr(state, "_lock_nonblocking", lambda fd: called.append("lock"))
    monkeypatch.setattr(state, "_unlock", lambda fd: called.append("unlock"))
    with state.index_lock(tmp_path):
        pass
    assert called == ["lock", "unlock"]
```

- [ ] **Step 3: `src/knowledge_mcp/state.py`의 락 구현을 `msvcrt` vs `fcntl`로 분기**
`src/knowledge_mcp/state.py`의 최상단 `import msvcrt`를 조건부로 변경하고 래퍼 함수 도입:
```python
import sys

if sys.platform == "win32":
    import msvcrt
    def _lock_nonblocking(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    def _lock_nonblocking(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
```
`index_lock`과 `async_index_lock` 내부에서 `_lock_nonblocking(lock_file.fileno())` 및 `_unlock(lock_file.fileno())`을 호출하도록 교체.

- [ ] **Step 4: Windows 테스트 통과 확인 (GREEN)**
```powershell
$env:PYTHONPATH = "C:\Users\dongwoo\.codex\worktrees\qdrant-hardening\obsidian-knowledge-mcp\src"
C:\Users\dongwoo\vs_proj\obsidian-knowledge-mcp\.venv\Scripts\python.exe -m pytest tests/test_state.py -k "lock" -v
```
Expected: PASS

- [ ] **Step 5: Linux WSL import probe 통과 확인 (GREEN)**
```powershell
wsl -e sh -c "PYTHONPATH=/mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/src /home/dongwoo/.cache/knowledge-mcp-hardening-venv/bin/python -c 'import knowledge_mcp.cli; print(\"IMPORT SUCCESS\")'"
```
Expected: `IMPORT SUCCESS`

- [ ] **Step 6: 커밋**
```powershell
git add src/knowledge_mcp/state.py tests/test_state.py
git commit -m "feat(compat): implement cross-platform advisory file locking with fcntl and msvcrt"
```

---

### Task 3: 프로세스 수명주기 및 시그널 POSIX 이식성 (`daemon.py`, `tests/test_daemon.py`, `tests/test_mcp_wire.py`)

**Files:**
- Modify: `src/knowledge_mcp/daemon.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_mcp_wire.py`

**Interfaces:**
- Consumes: `start_daemon_process`, `stop_daemon_process`
- Produces: POSIX 환경에서 `start_new_session=True`로 백그라운드 분리, `os.kill(pid, signal.SIGTERM)` / `proc.terminate()`로 프로세스 종료.

- [ ] **Step 1: `src/knowledge_mcp/daemon.py`의 `Popen` 호출에 POSIX 세션 플래그 정규화**
```python
    kwargs = {
        "cwd": str(settings.project_root),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "env": env,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
```

- [ ] **Step 2: `tests/test_daemon.py` 및 `tests/test_mcp_wire.py`의 플랫폼 종속 단언(assertion) 정규화**
`test_startup_timeout_terminates_owned_child_before_unlock` 등에서 `sys.platform == "win32"`일 때는 `taskkill` 단언, POSIX일 때는 `proc.terminate()` 단언을 수행하도록 분기 또는 `monkeypatch` 적용.

- [ ] **Step 3: Windows 테스트 구동 확인**
```powershell
$env:PYTHONPATH = "C:\Users\dongwoo\.codex\worktrees\qdrant-hardening\obsidian-knowledge-mcp\src"
C:\Users\dongwoo\vs_proj\obsidian-knowledge-mcp\.venv\Scripts\python.exe -m pytest tests/test_daemon.py tests/test_mcp_wire.py -k "daemon or proxy" -q
```
Expected: ALL PASS

- [ ] **Step 4: Linux WSL 환경에서 daemon 단위 테스트 구동 확인**
```powershell
wsl -e sh -c "PYTHONPATH=/mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/src /home/dongwoo/.cache/knowledge-mcp-hardening-venv/bin/python -m pytest /mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/tests/test_daemon.py -q"
```
Expected: ALL PASS

- [ ] **Step 5: 커밋**
```powershell
git add src/knowledge_mcp/daemon.py tests/test_daemon.py tests/test_mcp_wire.py
git commit -m "fix(compat): normalize POSIX daemon sessions and termination signals"
```

---

### Task 4: Linux CPU 전용 패키징 및 의존성 분기 (`pyproject.toml`)

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/user-guide-ko.md`

**Interfaces:**
- Consumes: Hatchling build system, PEP 508 environment markers.
- Produces: 
  - Linux 설치 시 NVIDIA CUDA wheel 의존성 배제 및 CPU torch/FastEmbed 동작.
  - Windows 기존 CUDA 12 가속 설치 사양 완벽 보존.

- [ ] **Step 1: `pyproject.toml`의 플랫폼별 의존성 마커 검토 및 수정**
```toml
dependencies = [
  "fastmcp==4.0.10",
  "mcp==2.2.0",
  "anyio>=4.9,<5",
  "sentence-transformers>=3,<6",
  "qdrant-client>=1.14,<2",
  "pypdf>=5,<7",
  "pathspec>=0.12,<1",
  "PyYAML>=6,<7",
  # Platform-specific inference dependencies
  "fastembed-gpu==0.8.0; sys_platform == 'win32'",
  "fastembed==0.8.0; sys_platform != 'win32'",
  "onnxruntime-gpu==1.26.0; sys_platform == 'win32'",
  "onnxruntime==1.26.0; sys_platform != 'win32'",
  "nvidia-cuda-runtime-cu12==12.8.90; sys_platform == 'win32'",
  "nvidia-cufft-cu12==11.4.1.4; sys_platform == 'win32'",
  "nvidia-cudnn-cu12==9.7.0.66; sys_platform == 'win32'",
]
```

- [ ] **Step 2: WSL Linux 환경에서 패키지 설치 및 임베딩 provider 초기화 smoke 테스트**
```powershell
wsl -e sh -c "PYTHONPATH=/mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/src /home/dongwoo/.cache/knowledge-mcp-hardening-venv/bin/python -c 'from knowledge_mcp.config import Settings; from knowledge_mcp.embeddings import LocalSentenceTransformerProvider; print(\"PROVIDER IMPORT OK\")'"
```
Expected: `PROVIDER IMPORT OK`

- [ ] **Step 3: 문서 갱신 (README & 한국어 가이드)**
Linux CPU 환경 설치 명령어 안내 추가 (CPU용 PyTorch 선설치 후 패키지 설치 가이드).

- [ ] **Step 4: 커밋**
```powershell
git add pyproject.toml README.md docs/user-guide-ko.md
git commit -m "feat(packaging): configure platform-aware dependency markers for Linux CPU support"
```

---

### Task 5: GitHub Actions CI 워크플로 매트릭스 구성 (`.github/workflows/ci.yml`)

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: GitHub Actions runner (`windows-latest`, `ubuntu-latest`).
- Produces: 자동화된 멀티 플랫폼 빌드 및 pytest 매트릭스 검증.

- [ ] **Step 1: `.github/workflows/ci.yml` 작성**
Windows 및 Ubuntu 환경에서 각각:
1. Python 3.11 및 3.12 설정
2. 의존성 설치 (Ubuntu는 CPU torch, Windows는 CUDA 또는 기본)
3. pytest 단위 테스트 실행
4. Qdrant 1.19.1 바이너리 테스트 스텝 포함

- [ ] **Step 2: CI 파일 구문 및 스키마 검증**

- [ ] **Step 3: 커밋**
```powershell
git add .github/workflows/ci.yml
git commit -m "ci: add multi-platform GitHub Actions workflow for Windows and Linux"
```

---

### Task 6: 전체 통합 게이트 검증 및 `main` 브랜치 병합 (Final Gate & Merge)

**Files:**
- Run: 전체 테스트 스위트
- Target: `main` 브랜치 병합

**Interfaces:**
- Consumes: `codex/qdrant-hardening` 브랜치의 모든 단위 커밋 (Unit 0 ~ Unit 8).
- Produces: `main` 브랜치로 무중단/테스트 검증된 Fast-forward 또는 머지 완료.

- [ ] **Step 1: Windows 전체 테스트 스위트 구동**
```powershell
$env:PYTHONPATH = "C:\Users\dongwoo\.codex\worktrees\qdrant-hardening\obsidian-knowledge-mcp\src"
C:\Users\dongwoo\vs_proj\obsidian-knowledge-mcp\.venv\Scripts\python.exe -m pytest tests/ -q
```
Expected: 모든 단위/통합 테스트 100% PASS (건너뛴 테스트 사유 확인).

- [ ] **Step 2: WSL Linux 전체 테스트 구동**
```powershell
wsl -e sh -c "PYTHONPATH=/mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/src /home/dongwoo/.cache/knowledge-mcp-hardening-venv/bin/python -m pytest /mnt/c/Users/dongwoo/.codex/worktrees/qdrant-hardening/obsidian-knowledge-mcp/tests/ -q"
```
Expected: Linux 환경에서도 100% PASS.

- [ ] **Step 3: SDD 진행 원장(`progress.md`) 최종 완료 업데이트**
`progress.md`에 Unit 8 완료 및 최종 게이트 통과 내역 기록 후 커밋.
```powershell
git add .superpowers/sdd/2026-09-30-qdrant-daemon-hardening/progress.md
git commit -m "docs(sdd): mark unit 8 and final multi-platform gates complete"
```

- [ ] **Step 4: `main` 브랜치로 병합**
메인 프로젝트 디렉터리(`C:\Users\dongwoo\vs_proj\obsidian-knowledge-mcp`)에서:
```powershell
git checkout main
git merge codex/qdrant-hardening --ff-only
```
(Fast-forward 병합 수행)

- [ ] **Step 5: Worktree 정리**
```powershell
git worktree remove "C:\Users\dongwoo\.codex\worktrees\qdrant-hardening\obsidian-knowledge-mcp"
```

- [ ] **Step 6: 메인 브랜치에서 최종 상태 확인**
```powershell
git status; git log -n 10 --oneline
```
Expected: `main` 브랜치가 최신 `codex/qdrant-hardening` 커밋을 가리키고 깨끗한 상태.
