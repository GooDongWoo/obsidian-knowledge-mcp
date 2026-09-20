# 문서 메타데이터 및 설정 가이드

이 문서는 `obsidian-knowledge-mcp`에서 문서 메타데이터(Frontmatter, Sidecar YAML), 보안 등급(Security Level), 무시 패턴(`.knowledgeignore`), 그리고 문서 유형 분류(`.knowledge-types.yaml`)를 설정하는 방법을 안내합니다.

---

## 1. 문서 메타데이터 설정 방법

문서의 생성일, 문서 유형, 보안 등급을 지정하면 `qdrant-find` 도구에서 정밀한 필터링(`document_type`, `created_from/to`, `include_private`)이 가능합니다.

### 1.1 Markdown 문서 (.md)
Markdown 파일의 최상단에 `---`로 감싸진 YAML Frontmatter를 작성합니다.

```markdown
---
security: private
document_type: diary
created_at: 2026-01-15
tags:
  - personal
  - reflection
---

# 오늘의 회고

본문 내용이 여기에 들어갑니다...
```

### 1.2 TXT 및 PDF 문서 (Sidecar YAML)
텍스트 파일이나 PDF 파일은 본문 내에 메타데이터를 직접 삽입하기 어려우므로, 대상 파일과 같은 디렉터리에 **동일한 파일명 뒤에 `.meta.yaml`을 붙인 Sidecar 파일**을 생성합니다.

- **원본 파일**: `2026-resume.pdf`
- **Sidecar 파일**: `2026-resume.pdf.meta.yaml`

```yaml
security: public
document_type: experience
created_at: 2026-02-01
modified_at: 2026-03-10
```

- **원본 파일**: `quick-memo.txt`
- **Sidecar 파일**: `quick-memo.txt.meta.yaml`

```yaml
security: private
document_type: note
```

---

## 2. 보안 등급 및 프라이버시 모델

### 2.1 `public` vs `private`
- **`public` (기본값)**: `security: private`가 명시되지 않은 모든 문서는 기본적으로 `public`으로 취급됩니다.
- **`private`**: 문서 Frontmatter 또는 Sidecar YAML에 명시적으로 `security: private`를 적은 문서만 `private`으로 분류됩니다.

### 2.2 검색 필터 동작 (`include_private`)
- **`include_private=false` (기본값)**:
  - AI 에이전트(Codex, Claude Code, Antigravity)가 일상적인 코딩이나 질의를 수행할 때 개인 일기, 민감한 금융/계정 메모 등 `private`으로 지정된 청크가 컨텍스트로 유출되지 않도록 차단합니다.
- **`include_private=true`**:
  - 사용자가 명시적으로 "내 비공개 문서도 포함해서 찾아줘"라고 요청한 경우에만 필터를 해제하여 검색 결과에 포함시킵니다.

> [!WARNING]
> 이 기능은 프롬프트 컨텍스트 격리를 위한 **클라이언트 검색 필터**이며, 암호화 기반의 접근 제어나 인증(Authentication) 시스템이 아닙니다.

---

## 3. 색인 제외 규칙 (`.knowledgeignore`)

Vault 내부에서 색인하지 않을 디렉터리나 파일 패턴을 지정합니다.

- **위치**: 프로젝트 루트 폴더의 `.knowledgeignore` (Vault 내부가 아님)
- **문법**: `.gitignore`와 동일한 pathspec 문법 지원

```text
# 템플릿 및 임시 폴더 제외
Templates/
.trash/
.obsidian/

# 특정 확장자나 대용량 바이너리 제외
*.canvas
*.excalidraw.md
archive/**/*.pdf

# 특정 개인 폴더 제외
Private/Confidential/
```

---

## 4. 문서 유형 자동 분류 (`.knowledge-types.yaml`)

개별 파일마다 Frontmatter에 `document_type`을 일일이 적지 않아도, 특정 디렉터리 경로에 있는 파일에 기본 유형을 자동으로 부여할 수 있습니다.

- **위치**: 프로젝트 루트 폴더의 `.knowledge-types.yaml`

```yaml
rules:
  - path_prefix: "Diary/"
    document_type: "diary"
  - path_prefix: "Research/Papers/"
    document_type: "paper"
  - path_prefix: "Career/Awards/"
    document_type: "award"
  - path_prefix: "Notes/Meetings/"
    document_type: "meeting"
```

- 파일 자체의 Frontmatter나 Sidecar YAML에 `document_type`이 지정되어 있으면 해당 값이 우선 적용됩니다.

---

## 5. 파일 크기 상한 정책

- 시스템 안정성 및 VRAM/메모리 고갈 방지를 위해 **30MB**를 초과하는 파일은 색인 대상에서 자동으로 제외됩니다.
- 대용량 파일은 인덱싱 시 `skipped_large_file`로 기록되며, 에러가 아닌 정상 스킵 처리됩니다.
