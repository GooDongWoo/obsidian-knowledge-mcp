# BF16 추론 검증 결과와 전환·복귀 절차 — 2026-10-21

격리된 Windows 합성 corpus 실험의 최종 **joined gate는 PASS(16/16)**다. 운영 BF16 전환·운영 corpus 품질·장기 관측은 **HOLD/미실행**이다. 기존 운영 FP32 서비스는 원래 코드로 복원되었다. 이 문서는 실행 지침이며 운영 설정 변경이나 재색인 완료 기록이 아니다.

**가장 중요한 trade-off:** 같은 production 경로의 FP32→BF16 비교에서 steady CUDA allocated는 49.7%, warm E2E p95는 70.5% 감소했다. 그러나 호스트 end RSS는 **2,259,804,160→3,625,791,488 bytes(약 2.10→3.38 GiB)로 증가**했고 lifetime peak RSS도 증가했다. VRAM 감소를 전체 RAM 감소로 해석하면 안 된다. 품질 평균 차이는 양수지만 95% 신뢰구간에 0이 포함된다. Critical source 누락은 **1→1**이고 adjacent exact score tie는 **291→1114**다. 상대 회귀 gate 통과이며 정확도 개선이나 절대 critical 누락 0을 입증하지 않는다.

| 판정 범위 | 상태 | 의미 |
|---|---|---|
| 현재 Windows 합성 CUDA 비교·전체 색인·E2E | PASS | 최종 결합 16 gates와 정해진 품질·성능 목표 충족 |
| CPU/플랫폼 검증 범위 | PARTIAL | 실제 FP32 CPU smoke·두 질의는 통과; 전체 CPU workload·Linux·CPU-only wheel 미검증 |
| 운영 BF16 전체 재색인·전환·장기 관측 | HOLD | 실행 전; 합성 PASS로 대신할 수 없음 |

## 1. 적용되는 동작

| 설정/상황 | 임베딩 | 리랭커 | 운영 의미 |
|---|---|---|---|
| `KNOWLEDGE_INFERENCE_PRECISION=auto`, native BF16 CUDA | bfloat16 | bfloat16 | 기본 정책 |
| `auto`, CPU | float32 | float32 | CPU 문서 쓰기는 FP32 전체 재색인 필요 |
| `float32`, CUDA/CPU | float32 | float32 | 공통 override |
| CUDA auto에서 native BF16 확인 불가/실패 | 초기화 실패 | 기존 CPU 생성 재시도 1회, FP32 | 자동 임베딩 CPU 복귀로 간주하지 말고 원인 확인 또는 공통 float32 선택 |

허용값은 대소문자를 구분하는 `auto|float32`다. 지원 조회 API가 허용하면 `including_emulation=False`로 native 지원을 확인한다. 구 API에서는 CUDA capability major≥8과 지원 조회를 함께 요구한다. 리랭커 CPU 실행 성공은 `rerank_applied=true`이며 RRF 복귀와 다르다. 초기화/추론/점수 검증 실패 시 기존 RRF 순서로 복귀하는 계약은 유지한다.

모델 ID/revision·tokenizer·prompt·pooling·truncation·normalization·chunk 400/60·batch 8·CPU threads 4·allocator budget은 함께 바꾸지 않았다. 모델/provider routing, 모델별 collection, 요청 `embedding_model`, `--client` 계약도 유지한다. FastEmbed/ONNX·BM25 및 Qdrant 저장 dtype 변경은 범위 밖이다. BF16 파라미터·대표 forward·일부 실제 BF16 CUDA kernel을 확인했지만 모든 연산이 BF16이라는 보장은 아니다.

실효 임베딩 dtype은 generation/indexing signature의 `embedding_precision`에 들어간다. dtype 변경 시 다음 일반 sync도 전체 문서를 다시 처리한다. precision 키가 없던 구 manifest→새 `float32`도 signature가 달라 1회 전체 FP32 sync가 필요하다. 새 코드에서 동일 dtype·동일 소스의 반복 sync만 unchanged gate를 통과했다. 리랭커 dtype만의 변경은 문서 임베딩 identity를 바꾸지 않는다.

`knowledge-index-status`의 `models` 각 항목에는 `embedding_precision_policy`, `embedding_effective_dtype`, `embedding_device`가 있다(단일 store는 최상위). `reranker`에는 `precision_policy`, `effective_dtype`, `device`, `loaded`, `last_fallback`이 있다. 초기화 전에는 일부 값이 null이다. 이 live 필드는 모델 로드나 Qdrant 재조회 없이 갱신된다. `daemon start`는 실행 중인 프로세스에 새 설정을 적용하지 않으므로 명시적으로 stop/start해야 한다.

현재 native Qdrant 기본 endpoint는 HTTP `127.0.0.1:26333`, gRPC `26334`다. `http://127.0.0.1:6333` 입력은 26333으로 정규화하는 legacy alias다. 아래 36333/36334는 검증 전용으로 production CLI endpoint를 변경하는 예가 아니다.

## 2. 실험 identity와 증거 경계

| 항목 | 고정값/실행값 |
|---|---|
| 환경 | Windows, RTX 4070 Ti, CUDA 12.8, Python 3.13.5 |
| 설치 버전 | torch 2.11.0+cu128; sentence-transformers 5.7.0; transformers 5.17.0; qdrant-client/server 1.19.1; fastembed-gpu 0.8.0; onnxruntime-gpu 1.26.0 |
| 임베딩 | `dragonkue/BGE-m3-ko` revision `7074d66aa46562342193ca4feb3d89bf9dad71b4` |
| 리랭커 | `dragonkue/bge-reranker-v2-m3-ko` revision `2aca5884ecac490192af9ebd86836d9073d826cd` |
| Dataset | [동결 합성 fixture](../tests/fixtures/inference_precision.jsonl), 200 documents / 100 queries / 50 correlated scenarios |
| Dataset SHA256 | `76ed0a935968daeeba703de722ac636ba8df49672b77f0755145d6d688694bfe` |
| Compare artifact SHA256 | `4ed42c8d3f4cdee82e78ded168f3e7e4d8e73a4d18c87114981010c301303992` |
| Qdrant artifact SHA256 | `0e74553aed72bff3d13fbc4dffbc59453883347f1dcec378b492b289b0498877` |
| Final joined artifact SHA256 | `9d3bdcfefd4d4d942836629417ff38b2c22efa748c6973f6ea794ebda7276b84` |
| Compare/Qdrant 당시 HEAD | `4df60fc32b0736a4042c4825ab78576bace25197` + 미커밋 validator 변경 |
| Compare/Qdrant harness SHA256 | `004f507fc8010e582501a99717643091e53e95667e168e68a8724270e54494dc` / `ff15e6bc7e9c03d37f1742cb7e299e1160e67c353ecfe602386b73fe9e84d513` |
| Final offline analysis SHA256 | `497e6ef3c7d2e877c10dc856ec5b2e6853b106dc1209ad41eba131d828dfbe67` (commit `0ec810d`) |

원문 질의/출력·로컬 모델 경로·PID·storage journal·리뷰 자료는 ignored `.knowledge/bf16-validation`에만 남긴다. 이 문서에는 합성 집계만 기록했다. 최종 근거는 `joined-gate-task5-fix1.json`이며 원래 compare/Qdrant artifact를 SHA로 결합한다. 측정 사이의 대표 kernel 진단 추가와 analysis 수정 때문에 harness hash는 다르다. 입력·모델·패키지·실행 계약과 worker runtime은 joined gate에서 일치 검증했다. 일부 CPU worker의 source hash는 진단 수정 후 디스크 상태를 읽은 값이므로 불변 loaded-code snapshot 증거가 아니다.

물리 메모리 visible 값은 32,629,664 KiB(약 31.1 GiB)였다. 운영 복원 후 free 14,386,732 KiB는 한 시점의 snapshot이며 peak memory 여유 증거가 아니다. Windows/WDDM per-process VRAM의 `N/A`는 0이 아니다.

## 3. 서로 구분해야 하는 세 비교

1. **구 loader default vs 구 explicit FP32:** document/query embedding 최대 절대 오차 0, 400 judged reranker score 오차 0. 기존 default가 FP32였다는 독립 parity 증거다.
2. **새 production FP32 override vs auto BF16:** 동일 경로·두 모델 동시 resident·순차 fresh worker의 dtype 비교다. 파라미터 전체와 대표 Linear forward는 각각 CUDA FP32/BF16이었다. DB 없는 비교에서 steady allocated −49.8749%, combined p95 −68.0232%, embedding corpus throughput +46.0916%였다. 구 standalone→새 production override p95는 embedding 29.388735→34.339565 ms(+16.846%), reranker 341.966835→350.534885 ms(+2.506%)였다. 경로·resident 조건이 달라 이를 guard/NumPy bridge 비용으로 단정할 수 없다.
3. **실제 full index/E2E:** production parser·frontmatter·tokenizer chunker·indexer·Qdrant·SQLite 및 dense20+BM2520/RRF40→reranker→source cap2/limit8을 통과한 아래 결과다. 한 arm당 full index 1회, warmup 5회 뒤 100 queries×5의 warm E2E 측정이다.

| Full-index/E2E 측정 | FP32 | Auto BF16 |
|---|---:|---:|
| Steady CUDA allocated bytes | 4,571,011,584 | 2,299,990,528 |
| Warm query window peak allocated bytes | 4,944,632,832 | 2,492,468,736 |
| Peak reserved bytes | 5,515,509,760 | 2,663,383,040 |
| **End host RSS bytes** | **2,259,804,160** | **3,625,791,488** |
| **Lifetime peak host RSS bytes** | **3,831,336,960** | **4,469,309,440** |
| Ambient 포함 total GPU used MiB | 9,178 | 6,351 |
| Warm E2E p95 ms | 1,046.563020 | 309.164565 |
| Full index seconds | 180.159536 | 150.505367 |
| Full index points/second | 1.254443730 | 1.501607586 |

Steady allocated −49.6831%는 ≥25% 절감 gate, p95 −70.4591%는 ≤10% 회귀 gate, 전체 색인 throughput +19.7031%는 ≤5% 감소 gate를 통과했다. 전체 색인 시간에는 실제 parser/Git native process 시작·embedding·BM25·Qdrant·SQLite가 포함되고 모델 로딩과 expected-chunk prepass는 제외된다. Windows Git 시작 비용이 파일당 약 0.9초여서 dtype만의 시간 절감으로 귀속할 수 없다. BF16 worker는 shared-candidate 비교용 baseline reference도 보유했다. RSS 원인을 격리한 실험이 아니므로 인과를 단정하지 않는다. 두 모델 resident 값이며 standalone reranker BF16 RSS/peak는 미측정이다. CUDA peak는 reset한 warm query 구간 값이다.

두 arm 모두 added200, changed/deleted/skipped/failed0, 오류 없음, source/manifest200, points226이었다. 동일 dtype 반복 sync는 unchanged200이었다. body/hash/metadata/schema 및 정렬한 sparse BM25 의미는 일치했고 filter 위반0, 예상 밖 CPU/RRF fallback0, OOM0이었다. 정규화 1024차원 vector는 finite/zero0, aligned chunk cosine mean 0.999900470/min 0.999319204였다. 501 allocated samples/arm의 warmup 이후 성장0이었다.

| Document-level 품질 | FP32 | BF16 | Delta의 paired scenario bootstrap 95% CI |
|---|---:|---:|---|
| Candidate Recall@20 | 0.875 | 0.875 | [0, 0] |
| Rerank=false nDCG@8 | 0.778715477 | 0.779704163 | [−0.008848876, 0.012074786] |
| Same-candidate nDCG@8 | 0.807714406 | 0.825440872 | [−0.003049386, 0.047757061] |
| E2E served-result recall | 0.795 | 0.805 | [−0.01, 0.03] |
| E2E nDCG@8 | 0.807714406 | 0.824211450 | [−0.002262870, 0.041675131] |
| E2E MRR@8 | 0.883511905 | 0.904083333 | [−0.001666667, 0.050817857] |

각 분석의 Recall/MRR/nDCG delta≥−0.01, paired 95% lower≥−0.02 gate를 통과했다(seed20261020, 2,000 resamples, 50 scenarios). E2E recall은 최종 8 chunks에서 보이는 문서 recall이며 candidate Recall@20과 다르다. 문서당 qrel로 source dedupe하며 query당 4 judgments 외 196 문서는 unjudged를 0 gain으로 처리한 불완전 평가다. Critical source 누락1→1, top8 document overlap0.947142857, top1 변경률0.06이다. Same-candidate score 평균/최대 절대 오차0.000281300/0.033335119, adjacent exact tie291→1114, gap≤0.001은 양쪽2354였다. 문서 qrel은 반환 chunk의 answer-span 포함을 보장하지 않는다.

CPU smoke와 BF16 full collection에 대한 ordinary/filtered 2개 CPU 질의는 실제 FP32 CPU 파라미터·forward·rerank를 확인했다. **Windows CUDA wheel을 설치한 호스트에서 constructor의 `torch.cuda.is_available`을 false로 강제하여 수행한 실제 CPU 실행**이다. Linux/CPU-only wheel 검증이 아니다. CPU 전체 workload/throughput, 다른 ST3/4 조합, Linux CUDA, 8192-token 전체 truncation 경계는 미실행이다. 대표 CUDA profiling은 timed/peak 구간 밖에서 BF16 GEMM/efficient attention을 확인했고 전체 shape의 kernel 검증은 아니다.

첫 Qdrant 시도는 synthetic Vault가 상위 repo Git 이력을 상속해 parser가 이를 탐색하면서 collection 생성 전 중단되었다. `qdrant-task5-aborted.json`은 superseded HOLD로 보존했다. 이후 fixture에 빈 독립 Git repo를 만들었으며 dataset body/frontmatter/labels/hash는 바꾸지 않았다. 최종 결과는 재실행 artifact다. 기존 critical 누락, 구 loader 대비 p95 진단 회귀, host RSS 증가와 score tie 증가를 숨기지 않는다.

## 4. 새 소유 서버로 재현하기

아래는 **실제 ML 실행용** 절차다. 운영 client/auto-start/writer를 멈추고 모델 worker와 daemon이 겹치지 않는 전용 측정 시간을 확보한 뒤 별도로 실행한다. 기존 production `.env`/Vault를 읽는 CLI 대신 validator가 만든 임시 Vault·별도 SQLite·별도 collection을 사용한다. 설치된 환경을 그대로 사용하며 model revision snapshot은 이미 cache에 있어야 한다. 현재 고정 validator는 Windows의 검증 전용 `http://127.0.0.1:36333`만 허용한다.

깨끗한 PowerShell에서 프로젝트 루트로 이동하고 Qdrant 1.19.1 실행 파일 경로를 수정한다. 이미 점유한 테스트 서버를 임의로 재사용하거나 종료하지 않는다. 다음은 새 짧은 storage와 **SDD scratch 밖의 persistent ownership 파일**을 만든다. PID/storage metadata는 로컬 파일이며 commit하지 않는다.

```powershell
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = "$PWD/src"
$python = Join-Path $PWD '.venv/Scripts/python.exe'
$qdrantExe = (Resolve-Path 'C:/Tools/qdrant-1.19.1/qdrant.exe').Path
if (Get-NetTCPConnection -LocalPort 36333,36334 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Test ports occupied; inspect ownership before proceeding'
}
if (Get-ChildItem Env:QDRANT__*) { throw 'Use a clean shell without QDRANT__ overrides' }
$runId = [guid]::NewGuid().ToString('N').Substring(0, 10)
$storage = Join-Path $env:USERPROFILE ".qb-$runId"
New-Item -ItemType Directory -Path $storage | Out-Null
$artifactDir = Join-Path $PWD ".knowledge/bf16-validation/$runId"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$ownership = Join-Path $artifactDir 'qdrant-owned.json'
$configPath = Join-Path $storage 'config.yaml'
$dataPath = (Join-Path $storage 'data').Replace('\', '/')
$snapshotPath = (Join-Path $storage 'snapshots').Replace('\', '/')
@"
storage:
  storage_path: '$dataPath'
  snapshots_path: '$snapshotPath'
service:
  host: 127.0.0.1
  http_port: 36333
  grpc_port: 36334
telemetry_disabled: true
log_level: WARN
"@ | Set-Content -LiteralPath $configPath -Encoding utf8
$ownedProcess = Start-Process -FilePath $qdrantExe -ArgumentList @('--config-path', "`"$configPath`"") -WorkingDirectory $storage -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $storage 'stdout.log') -RedirectStandardError (Join-Path $storage 'stderr.log')
@{pid=$ownedProcess.Id; url='http://127.0.0.1:36333'; storage=$storage; executable=$qdrantExe; purpose='bf16 isolated validation'} | ConvertTo-Json | Set-Content -LiteralPath $ownership -Encoding utf8
# Bounded startup polling for the owned child; then verify its provenance.
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($ownedProcess.HasExited) { throw 'Owned Qdrant exited; inspect its logs' }
    try {
        $null = Invoke-RestMethod http://127.0.0.1:36333/healthz -TimeoutSec 1
        $ready = $true
        break
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'Owned Qdrant not ready; inspect its logs' }
& $python -c 'import runpy,sys; m=runpy.run_path("scripts/validate_inference_precision.py"); print(m["verify_owned_qdrant"]("http://127.0.0.1:36333", sys.argv[1]))' $ownership
if ($LASTEXITCODE) { throw 'Owned server verification failed; inspect its logs' }
```

최종 verdict에는 다음 **세 단계**가 모두 필요하다. compare와 qdrant는 내부 worker를 순차 실행한다. 각 명령 종료 후 output status도 읽는다. 프로세스 exit0만으로 gate PASS를 판정하지 않는다.

```powershell
$env:KNOWLEDGE_RUN_REAL_MODELS = '1'
& $python scripts/validate_inference_precision.py --phase compare --output "$artifactDir/compare.json"
if ($LASTEXITCODE) { throw 'Compare failed; preserve HOLD evidence' }
& $python scripts/validate_inference_precision.py --phase qdrant --qdrant-url http://127.0.0.1:36333 --ownership $ownership --output "$artifactDir/qdrant.json"
if ($LASTEXITCODE) { throw 'Qdrant phase failed; preserve HOLD evidence' }
# Qdrant alone: HOLD_pending_join when its own checks pass, otherwise HOLD.
$env:KNOWLEDGE_RUN_REAL_MODELS = $null
& $python scripts/validate_inference_precision.py --phase finalize --compare-artifact "$artifactDir/compare.json" --qdrant-artifact "$artifactDir/qdrant.json" --output "$artifactDir/joined-gate.json"
if ($LASTEXITCODE) { throw 'Offline finalize failed' }
$joined = Get-Content -LiteralPath "$artifactDir/joined-gate.json" -Raw | ConvertFrom-Json
if ($joined.status -ne 'PASS') { throw ($joined.reasons -join '; ') }
```

`finalize`는 saved JSON/stdlib만 읽고 모델·서비스·ML opt-in이 필요 없다. 필수 7 compare workers, 두 full-index arm과 CPU-read, 입력 identity/runtime 및 개별 gate가 하나라도 빠지거나 다르면 HOLD다. 정확한 input SHA도 결과에 기록한다. explicit `--ownership`을 계속 넘기므로 SDD workspace 정리 뒤에도 재현이 scratch의 기본 metadata에 의존하지 않는다. 이 파일이 있어도 새 실행에서는 현재 PID/listener/storage를 다시 검증해야 한다. storage·collection·snapshot 삭제는 완료 조건이 아니며 자동 cleanup/monitoring을 만들지 않는다.

일반 회귀는 ML opt-in을 해제하고 아래 두 suite를 **순차** 실행한다. integration 전에 같은 소유 검증을 다시 수행한다. integration은 fake embedding을 사용하는 실제 Qdrant 검사이며 모델 품질 검증을 대체하지 않는다.

```powershell
$env:KNOWLEDGE_RUN_REAL_MODELS = $null
$env:KNOWLEDGE_TEST_QDRANT_URL = 'http://127.0.0.1:36333'
& $python -m pytest -q -m 'not real_models' --ignore=tests/test_qdrant_integration.py
if ($LASTEXITCODE) { throw 'Offline suite failed' }
& $python -c 'import runpy,sys; m=runpy.run_path("scripts/validate_inference_precision.py"); print(m["verify_owned_qdrant"]("http://127.0.0.1:36333", sys.argv[1]))' $ownership
if ($LASTEXITCODE) { throw 'Owned server verification failed' }
& $python -m pytest -q tests/test_qdrant_integration.py
if ($LASTEXITCODE) { throw 'Owned integration failed' }
git diff --check
if ($LASTEXITCODE) { throw 'Whitespace check failed' }
```

## 5. 운영 전체 색인·전환 리허설

이 절차는 별도 운영 적용 단계에서 수행한다. 기존 index를 `rebuild`로 지우지 말고 새 versioned collection을 선택한다. 소비 client/proxy와 auto-start를 중지하여 성공 전 연결되지 않도록 한다.

1. **읽기 preflight:** 설정된 Vault가 존재하고 읽을 수 있는지, 기대한 문서가 보이는지, 기존 collection metadata의 `vault_id`가 해당 canonical Vault 경로와 맞는지 확인한다. `vault_id`는 `uuid5(NAMESPACE_URL, str(Path(vault).resolve()).casefold())`다. 새 collection은 처음에는 없어야 한다. 누락/접근 실패 Vault를 정상 empty로 취급하지 않는다. discovery 결과 `[]`만으로 삭제를 허용하지 않는다. 기존 discovery 삭제 위험은 이번 정밀도 변경의 코드 수정 범위 밖이다.

```powershell
$project = (Resolve-Path 'C:/Path/To/obsidian-knowledge-mcp').Path
$python = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $project 'src'
$env:KNOWLEDGE_PROJECT_ROOT = $project
$env:KNOWLEDGE_VAULT_ROOT = 'C:/Path/To/Your/Obsidian'
$vault = (Resolve-Path -LiteralPath $env:KNOWLEDGE_VAULT_ROOT -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $vault -PathType Container)) { throw 'Vault missing' }
Get-ChildItem -LiteralPath $vault -Force -ErrorAction Stop | Select-Object -First 5
# Also open a known expected note locally to confirm actual read access/content.
$qdrantUrl = 'http://127.0.0.1:26333'
$oldCollection = 'YOUR_VERIFIED_OLD_COLLECTION'
$expectedVaultId = & $python -c 'from pathlib import Path; from uuid import uuid5,NAMESPACE_URL; import sys; print(uuid5(NAMESPACE_URL,str(Path(sys.argv[1]).resolve()).casefold()))' $vault
if ($LASTEXITCODE) { throw 'Vault identity calculation failed' }
$oldInfo = Invoke-RestMethod "$qdrantUrl/collections/$oldCollection"
if ($oldInfo.result.config.metadata.vault_id -ne $expectedVaultId.Trim()) { throw 'Collection/Vault identity mismatch' }
```

2. **중지·기록:** 소비 경로/auto-start/writer를 중지하고 daemon stop을 실행한다. `daemon status`뿐 아니라 `/health`의 PID, command/parent, port와 writer lock을 교차 확인한다. pidfile이 없다는 이유만으로 idle을 추정하거나 임의 Python process를 종료하지 않는다. 이전 checkout/HEAD·환경·runtime·collection·Vault 경로·설정과 dtype을 로컬 journal에 기록한다.
3. **일관성 백업:** writer가 없는 상태에서 Qdrant collection snapshot과 SQLite backup을 만들고 복원 가능한지 검증한다. 기존 collection·manifest·code/config를 그대로 보존한다. 상세 snapshot 절차는 [Qdrant 운영 가이드](qdrant-status-and-diagnostics.md#native-qdrant-migration)를 따른다. 아래 변수는 현재 확인한 값으로 채운다.

```powershell
$env:BF16_STATE_SOURCE = 'C:/Path/To/Project/.knowledge/state.sqlite3'
$env:BF16_STATE_BACKUP = 'C:/Path/To/Backup/state-before-bf16.sqlite3'
Invoke-RestMethod -Method Post -Uri "$qdrantUrl/collections/$oldCollection/snapshots"
# Save the returned snapshot name/path in the local journal and verify it.
# Target directory must exist; choose a new backup filename, never overwrite.
if (Test-Path -LiteralPath $env:BF16_STATE_BACKUP) { throw 'Backup already exists' }
& $python -c 'import os,sqlite3; from pathlib import Path; p=Path(os.environ["BF16_STATE_SOURCE"]); assert p.is_file(); s=sqlite3.connect(p.as_uri()+"?mode=ro", uri=True); d=sqlite3.connect(os.environ["BF16_STATE_BACKUP"]); s.backup(d); assert d.execute("PRAGMA integrity_check").fetchone()[0]=="ok"; d.close(); s.close()'
if ($LASTEXITCODE) { throw 'SQLite backup failed' }
```

4. **새 collection 전체 색인:** 승인된 새 코드/설정에서 `auto`와 새로운 collection을 함께 선택한다. 예시는 Windows이며 `$python`은 선택한 checkout의 Python, `KNOWLEDGE_PROJECT_ROOT`와 `PYTHONPATH`도 그 checkout을 가리켜야 한다. Vault는 1단계에서 확인한 동일 경로다. 새 collection이 이미 있으면 원인을 확인하고 다른 새 이름을 선택한다.

```powershell
$env:KNOWLEDGE_INFERENCE_PRECISION = 'auto'
$env:KNOWLEDGE_COLLECTION = 'obsidian_knowledge_bge_m3_ko_bf16_v2'
& $python -m knowledge_mcp.cli index --client codex
if ($LASTEXITCODE) { throw 'Index incomplete; keep consumers disconnected' }
# CLI process has exited: only now start the daemon; never overlap model loads.
& $python -m knowledge_mcp.cli daemon start --client codex
Invoke-RestMethod http://127.0.0.1:8765/health
```

5. **재연결 gate:** CLI summary의 failed0/오류 없음과 예상 source/point 수를 확인한다. HTTP200만 보지 말고 `/health` ready, `knowledge-index-status` completed(부분 실패0), 두 모델 CUDA+bfloat16, 예상 밖 fallback0, 실제 검색/필터/critical 질의를 검사한다. startup sync는 동일 signature·소스에서 unchanged여야 한다. 파일이 그 사이 바뀌었다면 정상 변경을 따로 설명한다. 모든 gate 성공 후 client를 재연결한다. 중단/timeout/partial/failure이면 연결하지 않고 상태를 읽어 판단하며 자동 sync/rebuild 재시도하지 않는다.
6. **관측:** query100건 이상·업무일2일 중 더 긴 조건을 초기 관측 기간으로 제안한다. OOM, 새 CPU/RRF fallback, critical 누락 발생 시 복귀한다. 장기 관측 자동화는 별도 요청이 있을 때만 생성한다. 구 collection/backup 삭제는 완료 조건에 넣지 않는다.

CPU 전환도 같은 소비 중지·백업·새 collection 절차를 따른다. BF16 corpus에 대한 CPU reader 호환성은 제한된 두 질의 확인이다. CPU writer가 FP32로 전체 문서를 다시 색인하는 동안 소비 경로를 열지 않는다. CUDA 강제 비활성화 방법은 배포 환경에 맞춰 별도로 검증하며 CPU 두 모델 float32와 전체 완료를 확인한 뒤 재개한다.

## 6. 실패 시 복귀와 최종 상태

writer·client·auto-start·새 daemon을 중지한다. 빠르게 보존 baseline을 재사용하려면 **이전 코드/환경/설정 + 이전 collection + 일관된 이전 SQLite manifest**를 함께 복원한다. 복원 전 현재 상태도 보존한다. 이전 코드도 해당 collection을 읽을 수 있어야 한다. 새 코드로 복귀하려면 공통 `float32`와 이전 collection을 선택하되, 구 manifest의 precision 키 부재로 전체 FP32 sync가 발생할 수 있으므로 소비 중지 상태로 전부 성공할 때까지 기다린다. 보존 baseline을 다시 쓰지 않으려면 새 FP32 collection을 사용한다. `float32` 설정만 바꿨다고 즉시 복귀 완료로 판정하지 않는다.

```powershell
# After stopping the new daemon, choose ONE rollback route described above.
$project = (Resolve-Path 'C:/Path/To/ChosenRollbackCheckout').Path
$python = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $project 'src'
$env:KNOWLEDGE_PROJECT_ROOT = $project
$env:KNOWLEDGE_INFERENCE_PRECISION = 'float32'
$env:KNOWLEDGE_COLLECTION = $oldCollection
# Restore the chosen code/environment/runtime consistently before this command.
& $python -m knowledge_mcp.cli daemon start --client codex
Invoke-RestMethod http://127.0.0.1:8765/health
```

Vault 존재/읽기 및 `vault_id` preflight를 다시 수행하고 startup sync 성공, 임베딩·리랭커 FP32, 실제 검색을 확인한 뒤 client를 재연결한다. 기존 collection/snapshot을 지우지 않는다.

실제 검증 뒤에는 모든 model worker가 종료된 후 controller가 원래 main FP32 서비스를 복원했다. 정상 startup sync run217은 added2/changed1/deleted0/error null, points3196→3234였다. 이는 중지 중 누적된 정상 Vault 변경의 동기화이며 운영 store가 전혀 변하지 않았다는 주장이 아니다. 운영 BF16 설정/collection 전환은 수행하지 않았다. 합성 joined gate PASS와 별개로 운영 전체 corpus·전체 CPU/Linux·장기 관측은 계속 미검증이다.

## 7. 문서화 단계의 회귀 검증

최종 리뷰의 일반 테스트 `.env` 로더 별칭 격리 수정과 두 discovery 회귀 사례를 반영한 뒤, 실제 ML opt-in을 해제한 최종 offline 전체 회귀는 **384 passed in 164.69s**, 별도 소유 Qdrant integration은 **18 passed in 21.41s**였다. 두 suite는 순차 실행했고 integration은 검증 전용 36333 서버만 사용했다. 저장된 실험의 offline finalize도 16/16 PASS였다. 이번 단계에서 실제 모델 실행·운영 재색인·서비스 전환을 추가로 수행하지 않았다.

최초 전체 실행은 legacy stdio proxy 복구 사례에서 내부 25초 제한 이후에도 종료되지 않아 소유 pytest tree만 검증 후 중단했다. 해당 사례 단독 실행은 1 passed in 5.66s, 이후 전체 실행 두 번과 이번 격리 수정 후 전체 실행에서는 지연이 재현되지 않았다. 원인이 규명되거나 proxy가 수정된 것은 아니다. 첫 완료 회귀의 유일한 실패는 baseline부터 현재 native 기본 포트 26333/26334와 테스트 기대값 6333/6334가 달랐던 기존 불일치였다(381 passed/1 failed). 해당 기대값 두 줄만 정정한 focused RED 1 failed/9 passed→GREEN 10 passed를 거친 뒤 당시 382개 전체 통과를 확인했다. 운영 endpoint 정책이나 legacy 6333 입력 alias는 바꾸지 않았다. 원시 로그와 중단된 실행 증거는 로컬 ignored 경로에 보존했다.
