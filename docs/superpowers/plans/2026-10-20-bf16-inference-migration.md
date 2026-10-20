# BF16 Inference Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 이 문서는 계획이며 실제 구현·운영 재색인 실행 지시는 아니다.

**Goal:** CUDA에서는 임베딩·리랭커 모두 bf16을 기본 사용하고 CPU에서는 fp32를 사용하도록 변경한다. 전체 재색인을 기본 전환 절차로 수행하고, 검색 품질·메모리·지연·장애 처리의 영향을 검증한다.

**Architecture:** 하나의 장치 기반 정책을 두 PyTorch 모델에 공통 적용한다. 기본 `auto`는 CUDA → bf16, CPU → fp32이며, fp32 기준선·긴급 복귀를 위한 공통 override만 둔다. 문서 벡터는 전체 재생성하고 반환·저장 형식은 기존 float 계약을 유지한다.

**Tech Stack:** Python 3.11+, Sentence Transformers `>=3,<6`, PyTorch, NumPy, FastEmbed/ONNX, Qdrant, pytest, FastMCP.

**Spec:** 최초 요청은 bf16 전환의 영향성 검사 계획이다. 2026-10-20 추가 요구: 전체 재색인을 허용하며 현재 환경에서 CUDA 사용 가능 시 임베딩·리랭커 모두 bf16, CPU에서는 bf16을 사용하지 않는다. 이 개정은 이전의 fp32 기본값·모델별 opt-in·기존 인덱스 재사용 우선 정책을 대체한다.

## Global Constraints

- 이번 작업은 계획 수정이다. 코드·운영 `.env`·Vault·인덱스 변경, 데몬 재시작, 실제 색인, commit/push는 실행하지 않는다. 전체 재색인은 후속 구현·적용 단계에 포함한다.
- `DENSE_MODELS`, `MODEL_COLLECTIONS`, `model_settings()`, provider factory, `EmbeddingProvider`, `Tokenizer`, 모델별 컬렉션, 요청의 `embedding_model`, `--client`를 유지한다.
- 모델 ID·revision, tokenizer, prompt, pooling, truncation, chunking `400/60`, normalization, batch 기본값 `8`, CPU thread 기본값 `4`, CUDA allocator budget은 동시에 변경하지 않는다.
- 대상은 현재 PyTorch embedding·reranker다. 유지 중인 FastEmbed/ONNX·BM25의 실행 정밀도는 변경하지 않는다. 전역 `torch.set_default_dtype()`, FP16/INT8 도입, Qdrant 저장 dtype 변경, FlashAttention 추가, 의존성 일괄 업그레이드는 범위 밖이다.
- 운영 corpus 전체 재색인은 허용된 전환 방식이다. 시험은 임시 Vault·별도 Qdrant·별도 SQLite에서 수행한다. 적용 시 기존 컬렉션과 manifest를 남기고 새 버전 컬렉션에 전체 색인해 복귀 가능성을 보존한다.
- 기존 미추적 책임 리팩토링 계획·설계 문서를 수정하거나 구현된 것으로 취급하지 않는다. 그 리팩토링 전체를 선행 조건으로 만들지 않는다.
- 단위 테스트 통과, kwargs 전달, GPU 지원 조회만으로 실제 bf16 추론·품질·성능 검증이 완료되었다고 하지 않는다.

---

## 1. 확인한 코드·환경과 기대 효과

최초 조사 기준: 2026-10-20, `main`, HEAD `dd1f55e1bad017f331c0a7151a705159fbe296b9`. 이 개정에서도 관련 indexer·dependency·daemon·검색 진입 경로를 재확인했다. GPU·버전 표는 최초 조사 스냅샷이며 구현 전에 다시 조회한다.

| 경로 | 현재 동작 | 검증할 영향 |
|---|---|---|
| `config.py::Settings` | 정밀도 설정 없음 | 자동 정책 기본값, 공통 fp32 override, 기존 생성자 호환성 |
| `embeddings.py::LocalSentenceTransformerProvider` | lazy `SentenceTransformer(model_name)`, dtype/device 명시 없음 | CUDA bf16·CPU fp32, tokenizer 접근 시 로딩, 반환 형식 |
| `embeddings.py::_encode` | 공유 lock, batch 8, normalization, NumPy 반환 | 수치 오차·정규화·배치별 CPU 전송·메모리 |
| `reranker.py::_initialize` | CUDA 우선, 초기화 실패 시 CPU 한 번 재시도 | CUDA 모델 bf16, CPU 재생성 fp32, 오류 의미 보존 |
| `retrieval.py::MultiModelStore` | RRF 후보 최대 40 → rerank → 출처당 2 → limit | 점수·순위 변화와 RRF 복귀 |
| `qdrant_store.py` | 1024차원 cosine dense·BM25, 엄격한 스키마 검사 | 저장 형식·payload·필터 유지 |
| `indexer.py::_indexing_inputs` | 모델·parser/chunker·BM25 포함, dtype 없음 | 실효 embedding dtype을 generation에 반영 |
| `cli.py::_dependencies` | tokenizer 접근으로 embedding 로딩, 모델별 manifest 생성 | 로딩 완료 후 실효 dtype을 indexer에 전달 |
| `daemon.py::bootstrap` | 시작 시 sync, 이후 reranker warmup | 재시작이 쓰기를 유발. 전환 중 검색·자동 시작 경로 관리 |
| `server.py::find/status` | indexing 중 검색 거부, 오류 상태의 기존 store 검색은 가능 | 재색인 실패 후 부분 인덱스가 서비스되지 않도록 전환 절차 검증 |

최초 로컬 설치: torch `2.11.0+cu128`, sentence-transformers `5.7.0`, transformers `5.17.0`, numpy `2.5.3`, fastembed-gpu `0.8.0`, onnxruntime-gpu `1.26.0`, qdrant-client `1.19.1`.

최초 GPU 조회: RTX 4070 Ti, VRAM 12282 MiB, compute capability 8.9, driver 610.88. CUDA available 및 `is_bf16_supported(including_emulation=False)` 모두 True였다. 실제 모델 추론·성능은 아직 측정하지 않았다.

**효과의 해석:** 목표는 fp32에 가까운 검색 품질을 유지하며 가중치 메모리를 줄이고 지연 개선 가능성을 확인하는 것이다. bf16이 fp32보다 수치 정밀도나 검색 정확도를 높인다고 전제하지 않는다. fp16 대비 범위/안정성상의 이점과 fp32 대비 품질 손실을 구분한다. 전체 VRAM·E2E latency 개선율은 workload와 kernel에 따라 실측한다. [공식 inference 문서](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html)

## 2. 확정할 기본 정책

두 모델을 따로 opt-in하는 설정 대신 공통 설정 하나를 추가한다. 기본값이면 CUDA에서 두 모델 모두 bf16이다.

```python
# Settings의 마지막 필드에 추가해 기존 positional 인자의 의미를 보존한다.
inference_precision: str = "auto"
```

```dotenv
# 기본 동작: CUDA -> bfloat16, CPU -> float32
KNOWLEDGE_INFERENCE_PRECISION=auto
# 비교 실험/긴급 복귀 시에만 float32로 지정한다.
```

허용값은 `auto`, `float32`다. 빈 값·None·임의 dtype은 거부한다. `float32`는 두 모델 모두에 적용되는 진단/복귀 override다. 정상 운용 기본값을 fp32로 남기거나 모델 하나만 bf16인 상태를 최종 완료로 삼지 않는다.

| 환경·정책 | embedding | reranker | 처리 |
|---|---|---|---|
| CUDA 사용 가능, auto | CUDA bf16 | CUDA bf16 | 기본 목표. 별도 bf16 opt-in 불필요 |
| CUDA 사용 불가, CPU | CPU fp32 | CPU fp32 | CPU bf16 금지 |
| float32 override | 선택 장치 fp32 | 선택 장치 fp32 | 기준선·복귀 전용 |
| CUDA 있으나 bf16 지원 불가/확인 불가 | CUDA fp32로 조용히 바꾸지 않음 | 기존 초기화 실패 정책에 따른 CPU fp32 가능 | 지원 불일치 기록, 현재 환경의 출시 gate 실패 |
| reranker CUDA 초기화 실패 | embedding 기존 경로 유지 | CPU fp32 한 번 재시도 | degraded 상태로 보고 |
| embedding 로딩/추론 실패 | 작업 실패 | 해당 없음 | 새 무한 재시도·묵시적 성공 처리 없음 |
| reranker 추론 실패/잘못된 점수 | 기존 후보 유지 | RRF 복귀 | 기존 세 가지 오류 코드·metadata 유지 |

현재 환경은 CUDA/CPU를 명시적으로 고른다. 기타 accelerator를 새로 지원하지 않는다. CUDA 유무 판정은 torch 기준이며 ONNX provider discovery로 대신하지 않는다. bf16 지원 조회는 CUDA 가용성과 별도 검증이다. CPU fallback은 정상적인 장애 처리지만 “두 모델 CUDA bf16 완료”로 보고하지 않는다.

bf16 가중치·연산 경로여도 일부 커널·누산·정규화는 fp32일 수 있다. 모든 tensor를 bf16으로 강제하지 않는다. token IDs·mask 타입은 유지하며 반환 embedding/score는 float32 NumPy·Python float를 유지한다. 출력 cast가 bf16의 수치 손실을 복원하지는 않는다.

## 3. 전체 재색인을 기본으로 하는 데이터 전환

**기본 순서: fp32 기준선 → CUDA bf16 구현/회귀 → 새 컬렉션 전체 bf16 색인 → 품질·자원 확인 → 서비스 전환.** 기존 fp32 인덱스 재사용이나 10/50/90% 혼재 평가는 필수 gate에서 제거한다. 혼재 벡터를 정상 운용의 목표로 삼지 않는다.

- 기존 `KNOWLEDGE_COLLECTION`·모델별 manifest 경로를 활용해 새 컬렉션 `obsidian_knowledge_bge_m3_ko_bf16_v2`를 선택한다. 사용자 지정 컬렉션이면 원래 이름과 새 이름을 함께 기록한다. 모델 등록표·요청 라우팅은 유지한다.
- 새 컬렉션을 비어 있는 상태에서 전체 색인한다. 기존 컬렉션을 삭제하는 CLI rebuild는 기본 경로가 아니다. 새 컬렉션 최초 일반 sync도 모든 문서를 임베딩하므로 전체 재색인이다.
- `KnowledgeIndexer`에 optional keyword-only `embedding_precision: str | None = None`을 추가한다. None인 기존 호출은 기존 signature를 유지하고, 실제 ST 구성은 로딩 완료된 `provider.effective_dtype`을 반드시 넘긴다.
- `_indexing_inputs()`는 값이 있을 때 `embedding_precision`을 포함한다. requested auto 대신 실제 bfloat16/float32를 기록한다. source-state indexing_version에도 같은 입력이 반영되어 stat/hash가 같은 파일도 dtype 변경 시 전체 재임베딩 대상이 된다. SQLite 테이블 변경은 필요 없다.
- 같은 실효 dtype 재시작은 재색인하지 않는다. CUDA ↔ CPU 또는 override로 embedding dtype이 달라지면 전체 재색인한다. corpus가 작다는 사용자 판단에 따라 이를 수용하고 소요 시간을 측정한다.
- partial/error/cancel이면 전환하지 않는다. 동일 signature의 다음 sync는 완료된 파일을 유지하고 누락/실패 파일을 재개한다. timeout/cancel 뒤 자동 rebuild·자동 재시도는 하지 않는다.
- CPU fp32 query로 bf16 인덱스를 읽는 상황은 fallback 호환성으로 확인한다. CPU에서 문서를 쓰기 시작할 때는 전체 fp32 재색인을 완료하고 서비스한다.
- fp32 복귀는 원래 collection과 공통 override를 함께 복원한다. bf16 문서 벡터를 cast하면 fp32 원본으로 되돌아간다고 하지 않는다.

`--force-full-hash`만으로 dtype 변경에 따른 벡터 전체 재생성이 보장되지 않는다. 새 컬렉션과 실효 dtype signature로 전체 재색인을 검증한다.

## 4. 변경할 파일과 책임

| 단위 | 생성/수정 대상 | 독립 산출물 |
|---|---|---|
| 0 | 필요 시 `tests/conftest.py`, `tests/test_env_and_ignores.py`, `pyproject.toml`; 새 `tests/fixtures/precision_eval.jsonl`, `scripts/validate_inference_precision.py` | 환경 격리·고정 평가셋·fp32 기준선 |
| 1 | `config.py`, 새 `inference_precision.py`, 새 `tests/test_inference_precision.py` | auto 정책·공통 fp32 override |
| 2 | `embeddings.py`, `tests/test_embedding_providers.py`, `tests/test_embeddings.py` | CUDA bf16/CPU fp32 embedding·float 반환 |
| 3 | `reranker.py`, `tests/test_retrieval.py` | 동일 자동 정책·CPU/RRF fallback |
| 4 | `indexer.py`, `cli.py`, `server.py`; `tests/test_indexer.py`, `test_cli_paths.py`, `test_server.py`, `test_model_lifecycle.py`, `test_resource_limits.py`, `test_daemon.py` | 실효 dtype signature·상태·전체 재색인 회귀 |
| 5 | validator, `tests/test_qdrant_integration.py`, 로컬 측정 결과 | 전체 bf16 인덱스의 품질·자원·복귀 증거 |
| 6 | `.env.example`, `README.md`, `docs/user-guide-ko.md`, 새 `docs/2026-10-20-bf16-inference-results.md` | 적용/재색인/복귀 절차·결과 |

운영 파일의 기준 디렉터리는 `src/knowledge_mcp/`다. 경로가 선행 작업으로 바뀌면 실제 위치를 갱신한다. 범용 provider 프레임워크나 상주 모델 두 복사본을 만들지 않는다.

## 5. 논리 단위별 실행 계획

### 단위 0 — 격리와 기준선

**Interfaces:** Settings.from_paths와 기존 provider/SearchRequest/Qdrant client 주입을 재사용한다. validator는 `--phase baseline|compare|qdrant`, `--output PATH`, Qdrant phase의 `--qdrant-url URL`을 제공한다.

- [ ] HEAD·미추적 파일·설치 버전·GPU 지원을 다시 기록한다. 현재 무지정 dtype과 명시적 fp32가 같은지 parameter dtype·결과로 확인한다.
- [ ] pytest에서 부모 KNOWLEDGE 환경과 운영 .env 유입을 차단한다. 기존 환경 격리 fixture가 구현되었으면 재사용한다. 없으면 아래 fixture를 추가한다. 환경 로딩 자체의 테스트는 env_loading marker와 임시 .env를 사용한다.

```python
import os
import pytest
from knowledge_mcp import config

@pytest.fixture(autouse=True)
def isolated_knowledge_env(monkeypatch, request):
    for key in tuple(os.environ):
        if key.startswith("KNOWLEDGE_") and key != "KNOWLEDGE_TEST_QDRANT_URL":
            monkeypatch.delenv(key, raising=False)
    if request.node.get_closest_marker("env_loading") is None:
        monkeypatch.setattr(config, "_load_env_file", lambda *_: None)
```

- [ ] env_loading/real_models marker를 등록한다. 실제 모델 테스트는 `KNOWLEDGE_RUN_REAL_MODELS=1`을 collection 시 확인해 opt-in하고 일반 테스트는 모델 다운로드/GPU 로딩 없이 실행한다.
- [ ] 기존 실패를 기준선으로 분리한다. 일부 config 테스트의 URL 기대값 6333과 현재 기본값 26333 불일치는 재현 후 별도 보고한다. bf16 신규 실패처럼 취급하거나 숨기지 않는다.
- [ ] 문서 최소 200개·질의 100개와 관련성 0/1/2 qrels를 고정한다. 한국어 의미30, 기술어 혼용20, 정확 식별자15, 부정/hard negative15, 날짜·유형·private 필터10, 긴 입력10개로 나눈다. 결과 확인 전에 정답을 정한다.

```json
{"kind":"document","id":"d01","text":"CUDA 초기화에 실패한 리랭커는 CPU에서 다시 로딩한다.","source_path":"runtime/cuda.md","security_level":"public"}
{"kind":"query","id":"q01","text":"CUDA 로딩 실패 시 리랭커의 다음 동작은?","relevance":{"d01":2},"critical":true}
```

- [ ] validator는 고정 model revision을 캐시에서 local_files_only로 로딩하고 입력 hash·버전·장치·실효 dtype을 기록한다. baseline/compare는 DB 없이, qdrant phase는 소유 서버에서 실행한다. 각 조합은 별도 자식 process로 순차 측정한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_embedding_providers.py tests/test_retrieval.py tests/test_model_lifecycle.py tests/test_resource_limits.py
.\.venv\Scripts\python.exe scripts/validate_inference_precision.py --phase baseline --output .knowledge/bf16-validation/baseline.json
```

**Gate 0:** fp32 기준선·재현 identity 확보, 운영 데이터 접촉 없음, 기존 실패 구분. validator는 후속 구현할 파일이다.

### 단위 1 — 장치 기반 자동 precision

**Interfaces:** 작은 공통 모듈에 아래 두 함수만 둔다. torch import는 사용하는 함수 내부로 유지한다.

```python
def resolve_torch_precision(policy: str, device: str) -> str:
    if policy not in ("auto", "float32"):
        raise ValueError("inference_precision must be auto or float32")
    return "bfloat16" if policy == "auto" and device.split(":", 1)[0] == "cuda" else "float32"

def model_dtype_kwargs(model_class: type, dtype: str) -> dict[str, dict[str, str]]:
    import inspect
    parameters = inspect.signature(model_class).parameters
    for name in ("model_kwargs", "automodel_args"):
        if name in parameters:
            return {name: {"torch_dtype": dtype}}
    raise ValueError("model constructor has no supported dtype argument")
```

- [ ] Settings 마지막에 공통 field·환경 로더·검증을 추가한다. 테스트를 먼저 작성해 기본 auto, CPU fp32, CUDA bf16, override fp32, 잘못된 값, positional/replace/from_paths/from_env를 확인한다.
- [ ] 신규 field/helper 부재로 실패함을 확인한 뒤 최소 구현한다. torch.cuda.is_available 기준으로 CUDA/CPU를 고르고 선택 장치를 resolver에 전달한다.
- [ ] CUDA bf16 loader preflight는 선택 GPU context에서 native 지원을 확인한다. 현재 API는 including_emulation=False, 구 API는 native capability 확인을 함께 사용한다. 입증 실패는 명시적 초기화 오류이며 CUDA fp32로 조용히 바꾸지 않는다.

```python
def test_default_policy_is_device_based():
    from knowledge_mcp.inference_precision import resolve_torch_precision
    assert resolve_torch_precision("auto", "cuda:0") == "bfloat16"
    assert resolve_torch_precision("auto", "cpu") == "float32"
    assert resolve_torch_precision("float32", "cuda:0") == "float32"
```

- [ ] CPU에서 CUDA bf16 probe를 호출하지 않는지 확인한다. 생성자 helper는 ST3 automodel_args와 현재 model_kwargs fake signature로 검증한다. TypeError를 삼켜 재로딩하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_inference_precision.py tests/test_config.py tests/test_env_and_ignores.py tests/test_cli_paths.py
```

**Gate 1:** 별도 설정 없이 CUDA 두 모델 bf16·CPU fp32 정책. CLI/proxy import에 ML 의존성 유입 없음.

### 단위 2 — embedding 적용

**Interfaces:** async query/document 반환·vector name·1024차원·tokenizer 계약 유지. provider에 precision_policy/effective_dtype/device를 보유하고 로딩 전 effective/device는 None이다.

- [ ] CUDA 가용성·bf16 지원을 stub한 로딩 테스트부터 작성한다. constructor에 device와 dtype kwargs를 전달한다. lazy model lock·worker 경계와 tokenizer 최초 접근에도 같은 정책을 적용한다.
- [ ] normalize_embeddings=True·batch·NumPy 반환을 유지하고 출구의 np.asarray(vectors, dtype=np.float32)로 형식을 고정한다. encode precision에 bfloat16을 넘기거나 전체 문서를 GPU Tensor로 모으지 않는다.
- [ ] 빈 문서는 로딩 없이 [] 반환. 정상 shape·순서·float 목록, NaN/Inf/0벡터 쓰기 차단을 검증한다. 정상 fake zero vectors는 단위 벡터로 바꾸고 zero-vector negative test는 남긴다.
- [ ] 작은 bf16 Tensor 변환 테스트와 기존 FastEmbed/ONNX session 회귀를 실행한다. FastEmbed에 torch dtype을 전달하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_inference_precision.py tests/test_embedding_providers.py tests/test_embeddings.py tests/test_model_lifecycle.py tests/test_resource_limits.py
```

**Gate 2:** CUDA query/document bf16·CPU fp32, float 반환·토큰 경계·정규화·자원 제한 유지.

### 단위 3 — reranker 적용

**Interfaces:** 기존 rerank API·RerankError·세 가지 SearchResult 오류 코드를 유지하고 상태에 policy·실효 dtype을 추가한다.

- [ ] CUDA bf16·CPU fp32·override·CUDA 초기화 실패 후 CPU fp32 테스트부터 추가한다. CPU 재생성 시 dtype을 다시 계산한다.
- [ ] initialize에 동일 helper를 적용하고 CPU 재시도 한 번·초기화 오류 cache·RRF fallback을 유지한다. CUDA fp32로 숨겨지는 경로를 만들지 않는다.
- [ ] 구버전 NumPy 경계 차이를 피하도록 최대40개 점수는 worker 내에서 Tensor로 받아 float32 변환한다. activation/softmax는 바꾸지 않는다.

```python
scores = self._model.predict(
    [(query, candidate.document) for candidate in candidates],
    batch_size=self.batch_size,
    convert_to_tensor=True,
)
return scores.detach().float().cpu().numpy()
```

- [ ] fake predict를 Tensor 계약에 맞추고 NaN/Inf/길이 불일치 검증을 보존한다. 빈 후보·rerank=False 무추론, 동점 정렬 안정성, 오류 시 원래 RRF 순서·점수·metadata를 확인한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_retrieval.py tests/test_model_lifecycle.py tests/test_resource_limits.py
```

**Gate 3:** 두 모델 CUDA bf16 기본 적용·CPU fallback fp32. fallback을 완전한 CUDA bf16 성공으로 표시하지 않는다.

### 단위 4 — 재색인 signature·상태·lifecycle

**Interfaces:** indexer에 embedding_precision optional keyword 추가. dependencies는 tokenizer 로딩 후 실제 dtype을 전달한다. status는 precision_policy/effective_dtype·기존 device/fallback을 제공한다.

- [ ] indexing inputs에 실제 embedding dtype을 추가한다. 같은 dtype 재시작은 generation 유지, dtype 변경만으로 generation/indexing_version 변화 테스트를 작성한다.

```python
def test_embedding_precision_changes_generation(tmp_path):
    from knowledge_mcp.config import Settings
    from knowledge_mcp.indexer import KnowledgeIndexer
    vault, project = tmp_path / "vault", tmp_path / "project"
    vault.mkdir()
    project.mkdir()
    source = vault / "same.md"
    source.write_text("동일한 문서", encoding="utf-8")
    settings = Settings.from_paths(vault_root=vault, project_root=project)
    fp32 = KnowledgeIndexer(settings, object(), object(), embedding_precision="float32")
    bf16 = KnowledgeIndexer(settings, object(), object(), embedding_precision="bfloat16")
    assert fp32._fingerprint(source) != bf16._fingerprint(source)
    assert fp32._source_state(source)["indexing_version"] != bf16._source_state(source)["indexing_version"]
```

- [ ] 임시 Qdrant/manifest에서 fp32→bf16 sync 시 모든 문서 재임베딩, 다음 bf16 sync는 unchanged인지 확인한다. reranker 상태는 embedding signature에 넣지 않는다. metadata/BM25/필터는 유지한다.
- [ ] cached/live status가 실제 dtype을 반영하는지 검증한다. 로딩 전 dtype을 추측하거나 query·본문·예외 payload를 기록하지 않는다.
- [ ] 동시 warmup/tokenizer/첫 요청에서 한 번 생성, shared lock, cancel 시 native worker 종료까지 소유권 유지, batch/thread/allocator 회귀를 실행한다.
- [ ] 재색인 중 검색 거부를 확인한다. 현재 error 상태만으로 부분 인덱스 검색이 차단되지는 않으므로 전환 중 소비 클라이언트를 중지하고 전체 성공 후 연결한다. partial/error/cancel이면 candidate를 서비스하지 않는 복귀 절차를 검증한다. 일반 오류 정책 전체를 리팩토링하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_indexer.py tests/test_cli_paths.py tests/test_server.py tests/test_daemon.py tests/test_model_lifecycle.py tests/test_resource_limits.py tests/test_proxy.py tests/test_mcp_wire.py
```

**Gate 4:** dtype 변경은 전체 재색인·같은 dtype은 증분 유지. 모델 라우팅·MCP·privacy·취소·락 보존, 부분 재색인 결과 전환 금지.

### 단위 5 — 전체 bf16 인덱스 실측

**Interfaces:** validator JSON에 입력 hash·revision·버전·실효 dtype·색인 완료도·품질·메모리·latency를 저장한다. 의도적인 dtype/collection 차이를 제외한 identity가 다르면 비교를 거부한다.

- [ ] 새 소유 Qdrant1.19.1을 36333/36334·별도 storage로 실행하고 버전·PID·소유권을 확인한다. 운영26333/6333은 사용하지 않는다. 기존 integration test client 주입을 재사용하고 운영 endpoint 제한은 완화하지 않는다.
- [ ] 동일 corpus로 fp32 기준 collection과 전체 bf16 candidate를 만든다. source/point/manifest·실효 dtype·누락/failed0 확인. dtype별 generation/point ID는 달라질 수 있으므로 source_path·chunk_index·본문 hash로 정렬하고 ID 문자열 동일성을 요구하지 않는다.
- [ ] 필수 비교는 전체 fp32 기준선과 전체 bf16 문서/query/reranker다. CPU-only fp32·공통 override 복귀·CPU query→bf16 인덱스 읽기도 확인한다. 모델별 전환은 원인 분석용이며 최종 정상 구성은 아니다.
- [ ] rerank=False·동일 후보 reranker·실제 dense20+BM25 20→RRF40→rerank→출처당2→limit8을 구분해 분석한다. critical 누락·근접 점수 역전·필터를 확인한다.
- [ ] 모델별/동시 상주 steady allocated·peak allocated/reserved·RSS·가능한 process/total VRAM 기록. 별도 process 순차 측정, 앞뒤 CUDA synchronize, warmup5회, 질의100개5회 반복, corpus 처리량을 사용한다.
- [ ] batch8/thread4·allocator/TF32/attention 유지. batch1·많은 chunk 파일·긴 query/pair 경계 확인. 입력 길이를 줄여 이득을 만들지 않는다. dtype별 kernel dispatch 변화는 기록한다.
- [ ] 현재 Windows CUDA·CPU 실행은 필수 gate. Linux CPU·ST3/4/5 대표 조합은 별도 환경에서 회귀한다. Linux CUDA 배포 시 실제 GPU gate도 수행한다. 미실행 환경은 표시하고 현재 venv를 다운그레이드하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/validate_inference_precision.py --phase compare --output .knowledge/bf16-validation/compare.json
$env:KNOWLEDGE_TEST_QDRANT_URL = 'http://127.0.0.1:36333'
.\.venv\Scripts\python.exe scripts/validate_inference_precision.py --phase qdrant --qdrant-url http://127.0.0.1:36333 --output .knowledge/bf16-validation/qdrant.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_qdrant_integration.py
```

서버 소유 확인 후 URL을 지정한다. integration test 기본 URL이 운영 서비스일 수 있으므로 변수를 생략하지 않는다.

**Gate 5의 수치 기준 — 측정값이 아닌 제안 기준:**

| 항목 | 통과/판정 기준 |
|---|---|
| 정책 | auto에서 두 CUDA 모델 주요 가중치·대표 forward bf16, CPU fp32 |
| 전체 색인 | expected source/point/manifest 일치, 실패·누락·미완료0, 전체 문서 새 dtype 생성 |
| 건전성 | 1024차원, NaN/Inf/0벡터0, norm 오차 최대0.01 |
| 수치 진단 | fp32/bf16 cosine 평균>=0.999·최소>=0.99, 실패 시 원인 조사 |
| 품질 | Recall@20·nDCG@8·MRR@8 각각 하락<=0.01, critical 누락 증가0 |
| 불확실성 | paired bootstrap2000회·고정seed, 품질 차이95% 하한>=-0.02 |
| 순위 진단 | top8 overlap·top1 변경률·tie·score 오차 보고, fp32 순위를 정답으로 대신하지 않음 |
| 메모리 | 동시 상주 steady allocated VRAM>=25% 감소 목표, total/peak/RSS 별도 보고 |
| 성능 | warm p95 증가<=10%, 처리량 감소<=5%, 실제 감소율 보고 |
| 안정성 | OOM0·의도하지 않은 CPU/RRF fallback0·정상 CUDA fp32 경로0 |
| 누수 | warmup 후500회 allocated 증가<=max(64MiB, 시작대비5%), 단조 증가 없음 |
| 저장·필터 | private/date/type 위반0, 동일 source/chunk의 본문·metadata·BM25 의미·schema 유지 |

합성셋만 쓰면 운영 corpus 품질은 미검증이라고 기록한다. bf16이 fp32보다 정확도를 개선해야 한다는 조건은 두지 않는다. 메모리 목표 미달·지연 악화는 원인을 보고하고 성능 기대 충족 여부를 따로 판정한다. 품질·안정성 실패는 전환 보류다.

**Gate 5:** 두 모델 CUDA bf16+전체 bf16 인덱스의 품질·완전성·안정성 확인. GPU skip·리랭커 단독 통과는 전체 완료가 아니다. 혼재 호환성 증명을 기본 전환 조건으로 두지 않는다.

### 단위 6 — 전체 재색인·전환·복귀

**Interfaces:** 정상 설정은 auto. 복귀는 공통 float32와 이전 collection을 함께 복원한다. 기존 collection/manifest는 유지한다.

- [ ] env.example·README·가이드에 CUDA bf16/CPU fp32 기본값·override·실효 dtype 변경 시 전체 재색인·startup sync·restart를 설명한다. 저장 dtype 변경·모든 커널 bf16을 약속하지 않는다.
- [ ] 결과 문서에 버전/revision/hash·색인 시간·dtype·품질·자원·fallback·기존 실패·미실행 환경을 기록하고 원문은 commit하지 않는다.
- [ ] 격리된 offline 전체 회귀와 소유 Qdrant integration을 수행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m 'not real_models' --ignore=tests/test_qdrant_integration.py
git diff --check
```

- [ ] 리허설 순서: 소비 클라이언트/auto-start·writer 중지 → 기존 collection/runtime 설정 기록 → Qdrant snapshot·SQLite 일관성 백업 → auto+새 collection 선택 → 전체 색인 → failed/partial0·dtype·품질 확인 → 클라이언트 재연결. 색인 process와 daemon이 동시에 모델을 로딩하지 않게 순차 실행한다.
- [ ] CLI index로 새 collection을 미리 만들었으면 process 종료 후 daemon을 시작한다. startup sync는 동일 dtype signature로 unchanged여야 한다. 이미 실행 중인 daemon start가 새 설정을 적용하지 않는 점도 확인한다.
- [ ] 중단/실패 시 클라이언트는 연결하지 않는다. 복귀는 writer/daemon 종료 → 이전 collection+float32 복원 → startup sync 성공·dtype·검색 확인 → 클라이언트 재연결이다. 보존한 기존 collection은 이전 baseline 코드로도 읽을 수 있어야 한다.
- [ ] CPU 전환 때 bf16 corpus를 fp32로 재색인하는 동안에도 소비 경로를 중지하고 전체 성공 후 재개한다. CPU query 호환성과 CPU 문서 쓰기 운영을 구분한다.
- [ ] 초기 관측은 query100건 이상·업무일2일 중 더 긴 조건을 제안한다. OOM·새 fallback·critical 누락 시 복귀한다. 자동화는 별도 요청이 있을 때만 생성하고 기존 collection/백업 삭제는 완료 조건에 넣지 않는다.

**Gate 6:** 기본 CUDA 두 모델 bf16·전체 색인 전환 또는 CPU 두 모델 fp32를 확인한다. PASS/PARTIAL/HOLD·성능 목표 충족·미검증 범위를 구분한다. 계획 수정만으로 운영 전환 완료라고 하지 않는다.

## 6. 최종 보고 항목

| 구분 | 기록 |
|---|---|
| 정책 | auto 기본값·실제 device/dtype·공통 override |
| 데이터 | 전체 색인 시간·source/point·manifest·dtype signature·재시작 unchanged |
| 품질 | Recall/nDCG/MRR·critical 누락·CI·worst cases |
| 자원 | steady/peak allocated/reserved·RSS·total VRAM·cold/warm latency |
| 회귀 | 모델 라우팅·MCP·FastEmbed/ONNX·필터·lock·cancel·fallback |
| 복귀 | 이전 collection/manifest 보존·fp32 복귀·실패 시 서비스 차단 절차 |
| 결론 | CUDA 두 모델 bf16 전환 가능 / CPU fp32 확인 / 추가 검증 필요 / 보류 |

## 7. 근거와 이번 수정의 검증 범위

- 현재 저장소의 config/embeddings/reranker/indexer/cli/daemon/server/retrieval/qdrant_store 및 관련 테스트. 환경 수치는 최초 조사 스냅샷이며 구현 시 재확인한다.
- [Sentence Transformers inference efficiency](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html), [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html): bf16 모델 로딩과 출력 precision 구분.
- [PyTorch bf16 support](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html): CUDA 가용성과 native bf16 지원 구분.
- [ST3 embedding](https://github.com/UKPLab/sentence-transformers/blob/v3.0.0/sentence_transformers/SentenceTransformer.py), [ST3 CrossEncoder](https://github.com/UKPLab/sentence-transformers/blob/v3.0.0/sentence_transformers/cross_encoder/CrossEncoder.py): 하한 생성자/변환 경계. 실제 버전별 실행 검증을 대체하지 않는다.

이번 개정은 사용자 요구에 따른 정책·재색인·signature·검증·복귀 계획 수정이다. 실제 모델 추론·pytest 회귀·성능 평가·운영 재색인은 실행하지 않았다.
