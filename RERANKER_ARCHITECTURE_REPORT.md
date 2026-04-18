# Open WebUI 리랭커(Reranker) 아키텍처 분석 보고서

## 1. 리랭커란?

리랭커(Reranker)는 1차 검색(벡터 유사도/BM25)으로 가져온 문서들을 **쿼리와 함께 다시 정밀 평가**하여 순위를 재조정하는 2단계 검색 모델입니다.

```
[1단계: Bi-Encoder (빠름)]              [2단계: Cross-Encoder (정확)]
쿼리 → 임베딩 ─┐                       ┌─ (쿼리, 문서1) → 점수: 0.92
               ├→ 유사도 TOP_K 추출 ──→ ├─ (쿼리, 문서2) → 점수: 0.45
문서들 → 임베딩 ┘                       ├─ (쿼리, 문서3) → 점수: 0.88
                                        └─ 재정렬: 문서1 > 문서3 > 문서2
```

**Bi-Encoder** (1단계): 쿼리와 문서를 **독립적으로** 임베딩 → 코사인 유사도 비교 (빠르지만 부정확)
**Cross-Encoder** (2단계): 쿼리-문서 쌍을 **함께** 모델에 입력 → 관련성 점수 직접 예측 (느리지만 정확)

---

## 2. Open WebUI 리랭킹 전체 파이프라인

```
사용자 질의: "프로젝트 일정이 어떻게 되나요?"
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: 1차 검색 — BM25 + 벡터 검색 병행 (Hybrid Search)     │
│                                                              │
│  ┌─────────────┐        ┌──────────────────┐                │
│  │ BM25 검색    │        │ 벡터 유사도 검색   │                │
│  │ (키워드 매칭) │        │ (의미 유사도)      │                │
│  │              │        │                  │                │
│  │ 결과:        │        │ 결과:             │                │
│  │ 문서A (0.8)  │        │ 문서C (0.91)      │                │
│  │ 문서B (0.6)  │        │ 문서A (0.87)      │                │
│  │ 문서D (0.5)  │        │ 문서E (0.82)      │                │
│  └──────┬──────┘        └────────┬─────────┘                │
│         │                        │                          │
│         └──────────┬─────────────┘                          │
│                    ▼                                        │
│  ┌─────────────────────────────┐                            │
│  │ EnsembleRetriever (RRF)     │  ← Reciprocal Rank Fusion │
│  │                             │                            │
│  │ BM25 가중치: 0.5            │  ← HYBRID_BM25_WEIGHT      │
│  │ 벡터 가중치: 0.5            │                            │
│  │                             │                            │
│  │ 융합 결과 (중복 제거):       │                            │
│  │   문서A, 문서B, 문서C,      │                            │
│  │   문서D, 문서E              │                            │
│  └──────────┬──────────────────┘                            │
└─────────────┼────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 리랭킹 — Cross-Encoder로 정밀 재평가                 │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │ RerankCompressor                                 │        │
│  │                                                  │        │
│  │ 입력: (쿼리, 문서) 쌍 → Cross-Encoder 모델       │        │
│  │                                                  │        │
│  │ ("프로젝트 일정이 어떻게 되나요?", 문서A) → 0.92  │        │
│  │ ("프로젝트 일정이 어떻게 되나요?", 문서B) → 0.31  │        │
│  │ ("프로젝트 일정이 어떻게 되나요?", 문서C) → 0.88  │        │
│  │ ("프로젝트 일정이 어떻게 되나요?", 문서D) → 0.15  │        │
│  │ ("프로젝트 일정이 어떻게 되나요?", 문서E) → 0.76  │        │
│  └──────────┬──────────────────────────────────────┘        │
│             ▼                                               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ 필터링 & 정렬                                     │        │
│  │                                                  │        │
│  │ 1. 관련성 임계값 적용 (RELEVANCE_THRESHOLD=0.0)  │        │
│  │    → 문서D (0.15) 제거 (임계값 초과 시)           │        │
│  │                                                  │        │
│  │ 2. 점수 내림차순 정렬                             │        │
│  │    → 문서A(0.92) > 문서C(0.88) > 문서E(0.76)     │        │
│  │                                                  │        │
│  │ 3. TOP_K_RERANKER=3 적용                         │        │
│  │    → 문서A, 문서C, 문서E 반환                     │        │
│  └──────────┬──────────────────────────────────────┘        │
└─────────────┼────────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: 프롬프트 주입 → LLM 호출                             │
│                                                              │
│  <context>                                                   │
│  <source id="1">문서A 내용...</source>                       │
│  <source id="2">문서C 내용...</source>                       │
│  <source id="3">문서E 내용...</source>                       │
│  </context>                                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 지원하는 리랭킹 백엔드 (3종)

### 3.1 Sentence Transformers CrossEncoder (기본값)

**로컬 실행**. sentence-transformers 라이브러리의 CrossEncoder 모델을 직접 로드하여 추론합니다.

**파일**: `routers/retrieval.py` L195-230

```python
rf = sentence_transformers.CrossEncoder(
    get_model_path(reranking_model, auto_update),
    device=DEVICE_TYPE,                    # cuda / cpu / mps
    trust_remote_code=True,
    activation_fn=torch.nn.Sigmoid(),      # 선택적 시그모이드 활성화
)
```

| 항목 | 내용 |
|------|------|
| **실행 위치** | 로컬 (Open WebUI 서버) |
| **모델 예시** | `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base` |
| **입력** | (쿼리, 문서) 텍스트 쌍 |
| **출력** | 관련성 점수 (float) |
| **장점** | 외부 의존 없음, 빠른 추론 |
| **단점** | GPU 메모리 필요, 모델 다운로드 필요 |

### 3.2 ColBERT (Late Interaction)

**로컬 실행**. Jina ColBERT v2 모델의 Late Interaction 방식으로 정밀 스코어링합니다.

**파일**: `retrieval/models/colbert.py`

```python
class ColBERT(BaseReranker):
    def __init__(self, name, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt = Checkpoint(name, colbert_config=ColBERTConfig(model_name=name))

    def predict(self, sentences):
        query = sentences[0][0]
        docs = [i[1] for i in sentences]
        embedded_docs = self.ckpt.docFromText(docs, bsize=32)[0]
        embedded_queries = self.ckpt.queryFromText([query], bsize=32)
        scores = self.calculate_similarity_scores(embedded_query, embedded_docs)
        return scores

    def calculate_similarity_scores(self, query_embeddings, document_embeddings):
        # 토큰 레벨 MatMul → Max Pooling → Sum → Softmax
        computed_scores = torch.matmul(document_embeddings, query_embeddings.T)
        maximum_scores = torch.max(computed_scores, dim=1).values
        final_scores = maximum_scores.sum(dim=1)
        return torch.softmax(final_scores, dim=0).numpy()
```

| 항목 | 내용 |
|------|------|
| **실행 위치** | 로컬 (Open WebUI 서버) |
| **트리거 조건** | 모델명에 `jinaai/jina-colbert-v2` 포함 |
| **입력** | 쿼리 토큰 임베딩 × 문서 토큰 임베딩 |
| **출력** | Softmax 정규화된 점수 |
| **장점** | Cross-Encoder보다 빠름 (Late Interaction) |
| **단점** | GPU 필요, 대형 모델 |

**ColBERT Late Interaction 방식**:
```
쿼리: ["프로젝트", "일정"]  →  각 토큰별 임베딩
문서: ["3분기", "출시", "계획"]  →  각 토큰별 임베딩

쿼리 토큰 × 문서 토큰 = MatMul 행렬
→ 문서 토큰별 최대 유사도 (Max Pooling)
→ 합산 (Sum)
→ Softmax 정규화
= 최종 관련성 점수
```

### 3.3 External Reranker (외부 API)

**원격 API 호출**. Cohere, Jina, 자체 서버 등 외부 리랭킹 서비스를 호출합니다.

**파일**: `retrieval/models/external.py`

```python
class ExternalReranker(BaseReranker):
    def predict(self, sentences, user=None):
        query = sentences[0][0]
        docs = [i[1] for i in sentences]

        payload = {
            'model': self.model,
            'query': query,
            'documents': docs,
            'top_n': len(docs),
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        data = r.json()

        # 응답에서 점수 추출 (index 순 정렬)
        sorted_results = sorted(data['results'], key=lambda x: x['index'])
        return [result['relevance_score'] for result in sorted_results]
```

**API 요청/응답 형식**:
```json
// 요청
POST {RAG_EXTERNAL_RERANKER_URL}
{
  "model": "reranker-model",
  "query": "프로젝트 일정이 어떻게 되나요?",
  "documents": ["문서A 내용", "문서B 내용", "문서C 내용"],
  "top_n": 3
}

// 응답
{
  "results": [
    {"index": 0, "relevance_score": 0.92},
    {"index": 1, "relevance_score": 0.31},
    {"index": 2, "relevance_score": 0.88}
  ]
}
```

| 항목 | 내용 |
|------|------|
| **실행 위치** | 외부 서버 (HTTP API) |
| **트리거 조건** | `RAG_RERANKING_ENGINE=external` |
| **호환 서비스** | Cohere Rerank, Jina Reranker, 자체 서버 |
| **장점** | 로컬 GPU 불필요, 최신 모델 사용 가능 |
| **단점** | 네트워크 지연, API 비용, 외부 의존 |

---

## 4. 리랭커 없이 하이브리드 검색만 할 때 (Fallback)

리랭킹 모델이 설정되지 않은 경우, `RerankCompressor`는 **임베딩 코사인 유사도**로 대체합니다.

**파일**: `retrieval/utils.py` L1266-1274

```python
async def acompress_documents(self, documents, query, ...):
    if self.reranking_function is not None:
        # 리랭커 있으면: Cross-Encoder 점수
        scores = await asyncio.to_thread(self.reranking_function, query, documents)
    else:
        # 리랭커 없으면: 임베딩 코사인 유사도 (Fallback)
        from sentence_transformers import util
        query_embedding = await self.embedding_function(query, prefix)
        document_embedding = await self.embedding_function([doc.page_content for doc in documents], prefix)
        scores = util.cos_sim(query_embedding, document_embedding)[0]
```

| 방식 | 정확도 | 속도 | 비고 |
|------|--------|------|------|
| **Cross-Encoder** | 높음 | 느림 | 쿼리+문서를 함께 인코딩 |
| **ColBERT** | 높음 | 중간 | 토큰 레벨 Late Interaction |
| **External API** | 높음 | 네트워크 의존 | 외부 서비스 |
| **코사인 유사도 (Fallback)** | 낮음 | 빠름 | 독립 임베딩 비교 |

---

## 5. 리랭커 활성화 조건

리랭커가 동작하려면 **두 가지 조건**이 모두 충족되어야 합니다:

```python
# main.py L1109-1116
if app.state.config.ENABLE_RAG_HYBRID_SEARCH          # 조건 1: 하이브리드 검색 활성화
   and not app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:  # 조건 2: 임베딩 우회 비활성화
    app.state.rf = get_rf(...)     # 리랭킹 모델 로드
else:
    app.state.rf = None            # 리랭커 비활성화
```

**즉, 하이브리드 검색(`ENABLE_RAG_HYBRID_SEARCH=true`)이 켜져야만 리랭커가 로드됩니다.**

리랭커 없이 하이브리드 검색만 사용할 수도 있습니다 (`RAG_RERANKING_MODEL`을 비워두면 코사인 유사도 Fallback).

---

## 6. RRF (Reciprocal Rank Fusion) — BM25 + 벡터 결과 융합

리랭킹 전에, BM25와 벡터 검색 결과를 **EnsembleRetriever**가 RRF로 융합합니다.

**파일**: `retrieval/utils.py` L254-284

```
BM25 결과:    문서A(1위), 문서B(2위), 문서D(3위)
벡터 결과:    문서C(1위), 문서A(2위), 문서E(3위)

RRF 공식: score(d) = Σ  weight_i / (k + rank_i(d))

BM25 가중치 = 0.5 (HYBRID_BM25_WEIGHT)
벡터 가중치 = 0.5

문서A: 0.5/(60+1) + 0.5/(60+2) = 0.00820 + 0.00806 = 0.01626  ← 1위
문서C: 0.0       + 0.5/(60+1) = 0.00820                        ← 2위
문서B: 0.5/(60+2) + 0.0       = 0.00806                        ← 3위
문서E: 0.0       + 0.5/(60+3) = 0.00794                        ← 4위
문서D: 0.5/(60+3) + 0.0       = 0.00794                        ← 5위

→ 융합 순위: A > C > B > E > D
→ 이 결과가 리랭커에 전달됨
```

**가중치 제어** (`HYBRID_BM25_WEIGHT`):
- `0.0`: 벡터 검색만 사용 (BM25 무시)
- `0.5`: 균등 융합 (기본값)
- `1.0`: BM25만 사용 (벡터 무시)

---

## 7. RerankCompressor 핵심 코드 상세 분석

**파일**: `retrieval/utils.py` L1227-1299

```python
class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any     # 임베딩 함수 (Fallback용)
    top_n: int                  # 반환할 최대 문서 수 (TOP_K_RERANKER)
    reranking_function: Any     # 리랭킹 함수 (CrossEncoder/ColBERT/External)
    r_score: float              # 관련성 임계값 (RELEVANCE_THRESHOLD)

    async def acompress_documents(self, documents, query, callbacks=None):
        # ── Step 1: 점수 산출 ──
        if self.reranking_function is not None:
            scores = await asyncio.to_thread(
                self.reranking_function, query, documents
            )
            # Cross-Encoder: [(쿼리, 문서1), (쿼리, 문서2), ...] → [0.92, 0.31, ...]
        else:
            # Fallback: 코사인 유사도
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        # ── Step 2: 임계값 필터링 ──
        docs_with_scores = list(zip(documents, scores))
        if self.r_score:
            # r_score(0.0) 이상인 문서만 유지
            docs_with_scores = [(d, s) for d, s in docs_with_scores if s >= self.r_score]

        # ── Step 3: 정렬 + Top-N 절단 ──
        result = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        final_results = []
        for doc, doc_score in result[:self.top_n]:   # top_n = TOP_K_RERANKER
            doc.metadata['score'] = doc_score        # 메타데이터에 점수 기록
            final_results.append(doc)

        return final_results
```

---

## 8. 중복 제거 메커니즘

BM25와 벡터 검색 결과에 같은 문서가 있을 수 있으므로, **SHA-256 해시 기반 중복 제거**를 적용합니다.

**파일**: `retrieval/utils.py` L179-211

```python
CHUNK_HASH_KEY = '_content_hash'

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# EnsembleRetriever에서 id_key로 사용
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_search_retriever],
    weights=[0.5, 0.5],
    id_key=CHUNK_HASH_KEY,   # ← 이 키로 중복 판별
)
```

BM25가 Enriched Text (파일명+헤더 포함)를 사용해도, 원본 텍스트의 해시로 비교하므로 중복이 정확히 감지됩니다.

---

## 9. 런타임 모델 변경

UI의 Settings에서 리랭킹 모델을 변경하면 **즉시 반영**됩니다.

**파일**: `routers/retrieval.py` L905-978

```python
# 1. 기존 모델 언로드 + GPU 메모리 해제
request.app.state.rf = None
request.app.state.RERANKING_FUNCTION = None
gc.collect()
if DEVICE_TYPE == 'cuda':
    torch.cuda.empty_cache()

# 2. 새 모델 로드
request.app.state.rf = get_rf(
    engine=new_engine,
    reranking_model=new_model,
    external_reranker_url=new_url,
    ...
)

# 3. 래퍼 함수 재생성
request.app.state.RERANKING_FUNCTION = get_reranking_function(
    new_engine, new_model, request.app.state.rf
)
```

---

## 10. 환경변수 설정 가이드

### 10.1 로컬 CrossEncoder 사용

```bash
# .env
ENABLE_RAG_HYBRID_SEARCH=true
RAG_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# RAG_RERANKING_ENGINE 비워두기 (기본값 = 로컬)
RAG_TOP_K_RERANKER=3
RAG_RELEVANCE_THRESHOLD=0.0
RAG_HYBRID_BM25_WEIGHT=0.5
```

### 10.2 ColBERT 사용

```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_RERANKING_MODEL=jinaai/jina-colbert-v2
```

### 10.3 외부 API 사용 (Cohere, Jina 등)

```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_RERANKING_ENGINE=external
RAG_RERANKING_MODEL=rerank-v3.5
RAG_EXTERNAL_RERANKER_URL=https://api.cohere.ai/v1/rerank
RAG_EXTERNAL_RERANKER_API_KEY=your-api-key
RAG_EXTERNAL_RERANKER_TIMEOUT=30
```

### 10.4 하이브리드 검색만 (리랭커 없이)

```bash
ENABLE_RAG_HYBRID_SEARCH=true
# RAG_RERANKING_MODEL 비워두기
# → 코사인 유사도 Fallback으로 동작
```

---

## 11. 설정 파라미터 전체 표

| 환경변수 | 기본값 | 역할 |
|---------|--------|------|
| `ENABLE_RAG_HYBRID_SEARCH` | `false` | 하이브리드 검색 + 리랭킹 활성화 (필수) |
| `RAG_RERANKING_ENGINE` | `""` (로컬) | `""`: 로컬 CrossEncoder, `"external"`: 외부 API |
| `RAG_RERANKING_MODEL` | `""` | 리랭킹 모델명 (비어있으면 리랭커 비활성) |
| `RAG_RERANKING_MODEL_AUTO_UPDATE` | `true` | HuggingFace에서 모델 자동 업데이트 |
| `RAG_RERANKING_MODEL_TRUST_REMOTE_CODE` | `true` | 원격 코드 신뢰 여부 |
| `RAG_EXTERNAL_RERANKER_URL` | `""` | 외부 리랭커 API 엔드포인트 |
| `RAG_EXTERNAL_RERANKER_API_KEY` | `""` | 외부 리랭커 API 키 |
| `RAG_EXTERNAL_RERANKER_TIMEOUT` | `""` | 외부 리랭커 타임아웃 (초) |
| `RAG_TOP_K` | `3` | 1차 검색 결과 수 |
| `RAG_TOP_K_RERANKER` | `3` | 리랭킹 후 반환 결과 수 |
| `RAG_RELEVANCE_THRESHOLD` | `0.0` | 최소 관련성 점수 (미달 시 제거) |
| `RAG_HYBRID_BM25_WEIGHT` | `0.5` | BM25 가중치 (0=벡터만, 1=BM25만) |

---

## 12. 주요 파일 참조

| 파일 | 역할 | 핵심 라인 |
|------|------|----------|
| `config.py` | 리랭킹 환경변수 정의 | L2843-2882 |
| `main.py` | 리랭킹 모델 초기화 및 app.state 등록 | L1102-1162 |
| `routers/retrieval.py` | `get_rf()` — 모델 로드 팩토리 | L159-234 |
| `routers/retrieval.py` | 런타임 모델 변경 핸들러 | L905-978 |
| `retrieval/utils.py` | `RerankCompressor` — 핵심 리랭킹 로직 | L1227-1299 |
| `retrieval/utils.py` | `query_doc_with_hybrid_search()` — 전체 파이프라인 | L213-323 |
| `retrieval/utils.py` | `get_reranking_function()` — 래퍼 생성 | L913-923 |
| `retrieval/utils.py` | `VectorSearchRetriever` — 벡터 검색 | L100-144 |
| `retrieval/models/colbert.py` | ColBERT 구현 | 전체 |
| `retrieval/models/external.py` | 외부 API 리랭커 구현 | 전체 |
| `retrieval/models/base_reranker.py` | BaseReranker 추상 클래스 | 전체 |

---

## 13. 결론

Open WebUI의 리랭킹은 **2단계 검색 아키텍처**입니다:

1. **1단계 (Recall)**: BM25 키워드 검색 + 벡터 유사도 검색을 **RRF(Reciprocal Rank Fusion)**로 융합하여 후보 문서 풀을 넓힘
2. **2단계 (Precision)**: Cross-Encoder / ColBERT / 외부 API로 쿼리-문서 쌍을 정밀 평가하여 **진짜 관련 있는 문서만 선별**

리랭커를 사용하면 RAG 품질이 크게 향상되지만, **`ENABLE_RAG_HYBRID_SEARCH=true`가 필수 전제조건**이며, 리랭킹 모델(`RAG_RERANKING_MODEL`)까지 설정해야 완전한 2단계 검색이 활성화됩니다.
