# Open WebUI 토큰 카운팅 & 청킹 전략 분석 보고서

## 1. 핵심 결론

> **Open WebUI는 LLM의 컨텍스트 윈도우 한도를 자동으로 감지하거나 관리하지 않는다.**
> 토큰 카운팅(tiktoken)은 **청킹 단계에서만** 사용되며, LLM에 보내는 최종 프롬프트의 토큰 수를 제한하는 로직은 없다.

| 질문 | 답변 |
|------|------|
| tiktoken으로 토큰을 세는가? | ✅ 청킹 시 `token` 모드에서만 사용 |
| LLM의 컨텍스트 한도를 자동 파악하는가? | ❌ 하지 않음 |
| 토큰 한도에 따라 청크 수를 조정하는가? | ❌ TOP_K는 고정값 |
| 프롬프트 크기를 토큰 기준으로 제한하는가? | ❌ 제한 없음 |

---

## 2. tiktoken 사용 범위

tiktoken은 **오직 청킹 단계**에서만 사용됩니다.

### 2.1 사용되는 곳

**파일**: `routers/retrieval.py`

```python
import tiktoken  # L29

# 1. TokenTextSplitter에서 청크 크기 측정 (L1413-1418)
text_splitter = TokenTextSplitter(
    encoding_name='cl100k_base',    # GPT-4 계열 인코딩
    chunk_size=1000,                # 1000 토큰
    chunk_overlap=100,              # 100 토큰 겹침
)

# 2. merge_docs_to_target_size에서 청크 크기 측정 (L1277-1278)
encoding = tiktoken.get_encoding('cl100k_base')
measure_chunk_size = lambda text: len(encoding.encode(text))
```

### 2.2 사용되지 않는 곳

- ❌ LLM 요청 전 프롬프트 토큰 수 계산
- ❌ RAG 컨텍스트 크기 제한
- ❌ 대화 히스토리 길이 관리
- ❌ 모델별 컨텍스트 윈도우 체크

---

## 3. LLM 컨텍스트 윈도우 한도 관리 — 하지 않음

### 3.1 모델 메타데이터에 컨텍스트 길이 정보 없음

```python
# models/models.py — ModelMeta 클래스
class ModelMeta(BaseModel):
    profile_image_url: Optional[str] = '/static/favicon.png'
    description: Optional[str] = None
    capabilities: Optional[dict] = None  # context_length 필드 없음
```

### 3.2 요청 전 토큰 수 검증 없음

```
[사용자 메시지] + [RAG 컨텍스트] + [시스템 프롬프트] + [도구 스펙]
                            │
                            ▼
                    총 토큰 수 검증? → ❌ 없음
                            │
                            ▼
                    그대로 LLM API에 전송
                            │
                            ▼
                    LLM이 컨텍스트 초과 에러 반환 (가능)
```

### 3.3 관련 설정

```python
# payload.py — 사용자가 수동으로 설정 가능한 파라미터
'max_tokens': int           # OpenAI 형식 — 응답 최대 토큰 수 (입력 제한 아님)
'num_ctx': int              # Ollama 형식 — 컨텍스트 윈도우 크기
'num_predict': int          # Ollama 형식 — 응답 최대 토큰 수
```

이 파라미터들은 사용자가 UI/API에서 수동으로 설정하는 것이며, 시스템이 자동으로 계산하지 않습니다.

---

## 4. TOP_K와 토큰 한도의 관계 — 관계 없음

### 4.1 TOP_K는 고정값

```python
# config.py L2720
RAG_TOP_K = PersistentConfig('RAG_TOP_K', 'rag.top_k',
    int(os.environ.get('RAG_TOP_K', '3')))  # 기본값 3, 고정
```

TOP_K는 **모델의 컨텍스트 윈도우와 무관하게** 항상 동일한 수의 청크를 가져옵니다.

### 4.2 실제 주입되는 컨텍스트 크기 계산

```
최악의 경우 (기본 설정):
  CHUNK_SIZE = 1000자 ≈ 250 토큰
  TOP_K = 3
  쿼리 확장으로 3개 쿼리 × 3 청크 = 최대 9개 청크 (중복 제거 후)

  주입되는 RAG 컨텍스트 ≈ 9 × 250 = 약 2,250 토큰
  + <source> 태그 오버헤드 ≈ 300 토큰
  ─────────────────────────────
  총 RAG 컨텍스트 ≈ 약 2,500 토큰

  → 대부분의 LLM (4K~128K 컨텍스트)에서 문제 없음
```

```
위험한 경우:
  CHUNK_SIZE = 4000
  TOP_K = 20
  Full Context 모드 (문서 전체 주입)

  → 대용량 문서 시 컨텍스트 초과 가능
  → 시스템이 경고하거나 잘라주지 않음
```

### 4.3 동적 TOP_K 조정이 없는 이유 (추정)

1. **기본값(TOP_K=3, CHUNK_SIZE=1000)이 보수적**이라 대부분의 모델에서 초과하지 않음
2. Open WebUI는 **범용 UI**이므로, 모델별 세밀한 토큰 관리보다 **단순한 설정**을 우선
3. 컨텍스트 초과 시 **LLM API가 에러를 반환**하므로, 사용자가 설정을 조정하면 됨

---

## 5. 청킹(Chunking) 전략 상세

### 5.1 청킹 파이프라인

문서 업로드 시 다음 순서로 청킹이 수행됩니다:

```
원본 문서
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: 마크다운 헤더 분할 (선택적)                       │
│  ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER = true (기본값)      │
│                                                          │
│  # 서론            → 청크 A (서론 섹션)                    │
│  ## 배경            → 청크 B (배경 섹션)                   │
│  ## 방법론          → 청크 C (방법론 섹션)                  │
│  ### 실험 설계      → 청크 D (실험 설계 하위 섹션)           │
│  # 결론            → 청크 E (결론 섹션)                    │
│                                                          │
│  메타데이터에 headings 정보 보존                            │
│  (예: {"Header 1": "서론", "Header 2": "배경"})           │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1.5: 소형 청크 병합 (선택적)                         │
│  CHUNK_MIN_SIZE_TARGET > 0 일 때만 활성화                  │
│                                                          │
│  짧은 섹션(예: "# 서론\n짧은 문장")을                       │
│  다음 섹션과 병합하여 최소 크기 확보                          │
│  - 같은 파일/소스의 청크만 병합                              │
│  - 최대 크기(CHUNK_SIZE) 초과하지 않음                      │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: 문자/토큰 기반 2차 분할                          │
│                                                          │
│  ┌─ TEXT_SPLITTER = '' 또는 'character' (기본값) ──────┐  │
│  │  RecursiveCharacterTextSplitter                     │  │
│  │  - CHUNK_SIZE = 1000 (문자 수)                      │  │
│  │  - CHUNK_OVERLAP = 100 (문자 수)                    │  │
│  │  - 구분자: ["\n\n", "\n", " ", ""] 순서로 시도       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ TEXT_SPLITTER = 'token' ──────────────────────────┐  │
│  │  TokenTextSplitter (tiktoken)                      │  │
│  │  - CHUNK_SIZE = 1000 (토큰 수)                      │  │
│  │  - CHUNK_OVERLAP = 100 (토큰 수)                    │  │
│  │  - TIKTOKEN_ENCODING_NAME = 'cl100k_base'          │  │
│  └────────────────────────────────────────────────────┘  │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: 임베딩 생성 + 벡터 DB 저장                       │
│                                                          │
│  각 청크 → 임베딩 벡터 생성 (qwen3-embedding 등)           │
│         → ChromaDB의 file-{id} 컬렉션에 저장              │
│         → 메타데이터: file_id, name, hash, headings 등    │
└──────────────────────────────────────────────────────────┘
```

### 5.2 RecursiveCharacterTextSplitter (기본값)

**문자 수 기반** 분할. LangChain의 구현을 사용합니다.

```
구분자 우선순위:
1. "\n\n" (단락 경계) → 가장 먼저 시도
2. "\n"   (줄바꿈)   → 단락으로 안 나뉘면
3. " "    (공백)     → 줄바꿈으로 안 나뉘면
4. ""     (문자)     → 최후의 수단

CHUNK_SIZE = 1000자 기준으로:
"이것은 첫 번째 단락입니다.\n\n이것은 두 번째 단락입니다.\n\n..."
→ "\n\n"으로 나누되, 1000자를 넘지 않도록 조합
→ 100자 겹침(CHUNK_OVERLAP)으로 문맥 유지
```

**장점**: 자연스러운 경계(단락, 줄)에서 분할
**단점**: 문자 수 ≠ 토큰 수 (한국어는 1문자 ≈ 1~2토큰)

### 5.3 TokenTextSplitter (선택적)

**토큰 수 기반** 분할. tiktoken으로 정확한 토큰 수를 측정합니다.

```python
# 활성화 방법
TEXT_SPLITTER=token
TIKTOKEN_ENCODING_NAME=cl100k_base  # GPT-4/Claude 호환 인코딩

# 동작
text = "양자 컴퓨팅은 기존 컴퓨터와는..."
tokens = tiktoken.encode(text)  # [12345, 67890, ...]
# len(tokens) 기준으로 1000 토큰마다 분할
```

**장점**: LLM 토큰 기준으로 정확한 크기 제어
**단점**: 인코딩마다 토큰화 방식이 달라 모델별 차이 존재

### 5.4 MarkdownHeaderTextSplitter (Stage 1)

마크다운 헤더(`#`, `##`, `###` 등)를 기준으로 문서 구조를 보존하며 분할합니다.

```python
# 지원하는 헤더 레벨 (retrieval.py L1376-1383)
headers_to_split_on = [
    ('#',      'Header 1'),
    ('##',     'Header 2'),
    ('###',    'Header 3'),
    ('####',   'Header 4'),
    ('#####',  'Header 5'),
    ('######', 'Header 6'),
]
```

```
# 프로젝트 개요              ┌─ 청크: "프로젝트 개요\n내용..."
프로젝트 개요 내용...         │  metadata: {Header 1: "프로젝트 개요"}
                             │
## 일정                      ├─ 청크: "일정\n3분기 시작..."
3분기에 시작하여...           │  metadata: {Header 1: "프로젝트 개요",
                             │             Header 2: "일정"}
### 마일스톤                  │
7월, 9월, 11월 마일스톤...    ├─ 청크: "마일스톤\n7월..."
                             │  metadata: {Header 1: "프로젝트 개요",
## 예산                      │             Header 2: "일정",
총 예산 10억원...             │             Header 3: "마일스톤"}
                             │
                             └─ 청크: "예산\n총 예산..."
                                metadata: {Header 1: "프로젝트 개요",
                                           Header 2: "예산"}
```

이 헤더 메타데이터는 **하이브리드 검색의 Enriched Text**에서 활용됩니다.

### 5.5 소형 청크 병합 (merge_docs_to_target_size)

**파일**: `routers/retrieval.py` L1258-1319

`CHUNK_MIN_SIZE_TARGET` > 0이면, 마크다운 헤더 분할로 생긴 작은 청크를 병합합니다.

```
병합 전 (마크다운 분할 결과):
  청크1: "# 서론\n짧은 서론."        (15자)  ← 너무 작음
  청크2: "## 배경\n배경 설명..."     (800자)
  청크3: "## 결론\n한 줄 결론."      (20자)  ← 너무 작음

병합 후 (CHUNK_MIN_SIZE_TARGET=200, CHUNK_SIZE=1000):
  청크1+2: "# 서론\n짧은 서론.\n## 배경\n배경 설명..."  (815자) ← 병합됨
  청크3: "## 결론\n한 줄 결론."  (20자) ← 다음 청크가 없어 그대로

병합 조건:
  ✅ 같은 파일 (file_id 동일)
  ✅ 같은 소스 (source 동일)
  ✅ 현재 청크 < CHUNK_MIN_SIZE_TARGET
  ✅ 병합 결과 ≤ CHUNK_SIZE
```

---

## 6. Enriched Text — 하이브리드 검색용 메타데이터 강화

**파일**: `retrieval/utils.py` L179-210

하이브리드 검색 시 BM25 스코어링을 개선하기 위해, 청크 텍스트에 메타데이터를 추가합니다.

```
원본 청크:
  "3분기에 프로젝트를 시작하여 4분기에 완료 예정입니다."

Enriched 청크 (BM25 검색용):
  "3분기에 프로젝트를 시작하여 4분기에 완료 예정입니다.
   Filename: project_plan.pdf project plan pdf project plan pdf
   Title: 2024년 프로젝트 계획
   Section: 프로젝트 개요 > 일정
   Source: project_plan.pdf"
```

파일명이 **2번 반복**되는 이유는 BM25에서 파일명 매칭에 가중치를 주기 위함입니다.

---

## 7. 문자 수 vs 토큰 수 — 실제 차이

### 7.1 언어별 차이

```
영어 (CHUNK_SIZE=1000):
  문자 기반: ~1000자 ≈ ~250 토큰 (1자 ≈ 0.25토큰)
  토큰 기반: ~1000토큰 ≈ ~4000자

한국어 (CHUNK_SIZE=1000):
  문자 기반: ~1000자 ≈ ~500~700 토큰 (1자 ≈ 0.5~0.7토큰)
  토큰 기반: ~1000토큰 ≈ ~1400~2000자
```

### 7.2 실질적 영향

```
기본 설정 (character, CHUNK_SIZE=1000, TOP_K=3):

영어 문서:
  3 청크 × 250 토큰 = ~750 토큰 (RAG 컨텍스트)
  → 4K 모델에서도 여유로움

한국어 문서:
  3 청크 × 600 토큰 = ~1,800 토큰 (RAG 컨텍스트)
  → 4K 모델에서도 가능하나 영어보다 큰 비율 차지
```

### 7.3 한국어 사용 시 권장 설정

```bash
# 토큰 기반 청킹으로 변경하면 더 정확한 크기 제어 가능
TEXT_SPLITTER=token
TIKTOKEN_ENCODING_NAME=cl100k_base
CHUNK_SIZE=500        # 토큰 단위로 500
CHUNK_OVERLAP=50      # 토큰 단위로 50
```

---

## 8. 컨텍스트 초과 시 동작

### 8.1 현재 동작 — 보호 장치 없음

```
사용자 설정:
  TOP_K=50, CHUNK_SIZE=2000, Full Context 모드

LLM 컨텍스트 윈도우: 4,096 토큰

실제 프롬프트:
  시스템 프롬프트:     ~500 토큰
  RAG 컨텍스트:       ~25,000 토큰  ← 이미 초과
  사용자 메시지:       ~50 토큰
  ────────────────────
  총:                ~25,550 토큰  >> 4,096 한도

→ LLM API가 에러 반환 (400 Bad Request / context_length_exceeded)
→ Open WebUI는 이 에러를 사용자에게 그대로 전달
```

### 8.2 truncate_content — 제한적 보호

**파일**: `utils/task.py` L123-140

RAG 템플릿에서 `middletruncate` 필터를 사용하면 컨텍스트를 잘라낼 수 있습니다:

```
# 커스텀 RAG 템플릿에서 사용 가능
{{CONTEXT|middletruncate:8000}}
```

하지만 이는 **문자 수 기준**이며 기본 템플릿에는 적용되어 있지 않습니다.

### 8.3 안전한 설정 가이드

| 모델 컨텍스트 | 권장 TOP_K | 권장 CHUNK_SIZE | 예상 RAG 토큰 |
|-------------|-----------|----------------|-------------|
| 4K (GPT-3.5) | 3 | 500 | ~375 |
| 8K | 5 | 1000 | ~1,250 |
| 32K | 10 | 1000 | ~2,500 |
| 128K (GPT-4) | 20 | 1000 | ~5,000 |

---

## 9. 전체 설정 파라미터 표

### 9.1 청킹 관련

| 환경변수 | 기본값 | 단위 | 역할 |
|---------|--------|------|------|
| `TEXT_SPLITTER` | `""` (character) | - | `""` 또는 `"character"`: 문자 기반, `"token"`: 토큰 기반 |
| `CHUNK_SIZE` | `1000` | 문자 또는 토큰 | 청크 최대 크기 |
| `CHUNK_OVERLAP` | `100` | 문자 또는 토큰 | 청크 간 겹침 |
| `CHUNK_MIN_SIZE_TARGET` | `0` | 문자 또는 토큰 | 소형 청크 병합 목표 (0=비활성) |
| `ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` | `true` | - | 마크다운 헤더 기반 1차 분할 |
| `TIKTOKEN_ENCODING_NAME` | `cl100k_base` | - | 토큰 인코딩 (token 모드에서만) |

### 9.2 검색 관련

| 환경변수 | 기본값 | 역할 |
|---------|--------|------|
| `RAG_TOP_K` | `3` | 검색 시 가져올 청크 수 |
| `RAG_TOP_K_RERANKER` | `3` | 리랭킹 후 반환 청크 수 |
| `RAG_RELEVANCE_THRESHOLD` | `0.0` | 최소 관련성 점수 |
| `RAG_FULL_CONTEXT` | `false` | 전체 문서 주입 모드 |
| `BYPASS_EMBEDDING_AND_RETRIEVAL` | `false` | 벡터 검색 건너뛰기 |

### 9.3 토큰/컨텍스트 관련

| 설정 | 위치 | 역할 |
|------|------|------|
| `max_tokens` | API 요청 파라미터 | 응답 최대 토큰 수 (입력 제한 아님) |
| `num_ctx` | Ollama 파라미터 | 컨텍스트 윈도우 크기 |
| `num_predict` | Ollama 파라미터 | 응답 최대 토큰 수 |

---

## 10. 주요 파일 참조

| 파일 | 역할 | 핵심 라인 |
|------|------|----------|
| `config.py` | 청킹/토큰 설정 정의 | L2885-2918 |
| `routers/retrieval.py` | 청킹 파이프라인 (save_docs_to_vector_db) | L1371-1422 |
| `routers/retrieval.py` | 소형 청크 병합 (merge_docs_to_target_size) | L1258-1319 |
| `routers/retrieval.py` | tiktoken 사용 (TokenTextSplitter) | L1413-1418 |
| `retrieval/utils.py` | Enriched Text 생성 | L179-210 |
| `utils/task.py` | truncate_content (middletruncate) | L123-140 |
| `utils/payload.py` | max_tokens/num_ctx 매핑 | L108, L140-141 |
| `retrieval/loaders/main.py` | 파일 타입별 로더 선택 | L229-489 |

---

## 11. 결론

Open WebUI의 토큰/청킹 관리는 **"정적 설정 + 사용자 책임"** 방식입니다:

1. **토큰 카운팅**: tiktoken은 `token` 모드 청킹에서만 사용하며, LLM 요청 시에는 토큰을 세지 않음
2. **컨텍스트 한도**: 모델의 컨텍스트 윈도우를 자동 감지하지 않으며, 초과 시 LLM API 에러가 그대로 전달됨
3. **청크 수**: TOP_K는 고정값(기본 3)이며, 모델 컨텍스트에 따라 동적 조정되지 않음
4. **청킹 전략**: 마크다운 헤더 분할 → 문자/토큰 기반 2차 분할 → 소형 청크 병합의 3단계 파이프라인
5. **안전장치**: 기본 설정(TOP_K=3, CHUNK_SIZE=1000)이 보수적이라 대부분의 모델에서 안전하지만, 사용자가 설정을 높이면 초과 위험 있음

이는 Open WebUI가 **다양한 LLM 백엔드를 지원하는 범용 UI**이기 때문입니다. 모델마다 컨텍스트 윈도우가 다르고, 같은 모델도 배포 방식에 따라 다를 수 있어, **사용자가 자신의 환경에 맞게 설정하도록** 위임한 것입니다.
