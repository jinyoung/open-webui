# Open WebUI 쿼리 확장 전략 및 HyDE 미사용 분석 보고서

## 1. 핵심 결론

> **Open WebUI는 HyDE(Hypothetical Document Embeddings)를 사용하지 않는다.**
> 대신, **LLM 기반 Multi-Query Generation (쿼리 확장)** 전략을 사용한다.

코드베이스 전체에서 `HyDE`, `hypothetical`, `fake_document` 등의 키워드가 **0건**입니다.

---

## 2. HyDE vs 쿼리 확장 — 개념 비교

### 2.1 HyDE (Hypothetical Document Embeddings)

2022년 Gao et al. 논문에서 제안된 기법으로, **가상의 답변을 먼저 생성**하여 검색에 활용합니다.

```
사용자: "양자 컴퓨팅의 장점은?"
       │
       ▼
LLM에게: "이 질문에 대한 답변을 작성해 (검색 없이)"
       │
       ▼
가상 답변: "양자 컴퓨팅은 큐비트를 활용하여 병렬 연산이 가능하며,
           기존 컴퓨터로는 수천 년 걸리는 소인수 분해를
           수 분 내에 처리할 수 있다. 또한 양자 얽힘을 통해..."
       │
       ▼
이 가상 답변을 임베딩 → 벡터 검색
       │
       ▼
가상 답변과 유사한 실제 문서들이 검색됨
```

**원리**: 짧은 질문보다 긴 답변이 실제 문서와 **임베딩 공간에서 더 가까움**
**해결하는 문제**: 질문(query)과 문서(document) 사이의 의미적 간극 (query-document mismatch)

### 2.2 Open WebUI의 쿼리 확장 (Multi-Query Generation)

사용자 메시지로부터 **여러 개의 검색 쿼리**를 LLM이 생성하여 다각도로 검색합니다.

```
사용자: "양자 컴퓨팅의 장점은?"
       │
       ▼
LLM에게: "이 대화에서 검색에 적합한 쿼리 1~3개를 JSON으로 생성해"
       │
       ▼
생성된 쿼리들:
  ["양자 컴퓨팅 장점", "양자 컴퓨터 기존 컴퓨터 비교", "큐비트 병렬 연산"]
       │
       ▼
각 쿼리를 개별 임베딩 → 각각 벡터 검색 → 결과 병합
```

**원리**: 하나의 질문을 **여러 관점의 쿼리로 분해**하여 재현율(recall) 향상
**해결하는 문제**: 단일 쿼리로는 관련 문서를 모두 찾지 못하는 문제

### 2.3 비교표

| 항목 | HyDE | Open WebUI 쿼리 확장 |
|------|------|---------------------|
| **LLM에게 요청** | "가상 답변을 생성해" | "검색 쿼리를 생성해" |
| **LLM 출력** | 긴 텍스트 (가상 답변) | 짧은 키워드 쿼리 1~3개 |
| **임베딩 대상** | 가상 답변 1개 | 쿼리 N개 |
| **검색 횟수** | 1회 | N회 (쿼리 수만큼) |
| **해결하는 문제** | 쿼리-문서 의미 간극 | 단일 쿼리의 낮은 재현율 |
| **위험 요소** | 가상 답변이 틀리면 엉뚱한 문서 검색 | 쿼리가 부적절하면 노이즈 증가 |
| **LLM 비용** | 긴 답변 생성 (토큰 많음) | 짧은 JSON 생성 (토큰 적음) |
| **예측 가능성** | 낮음 (답변 품질에 의존) | 높음 (구조화된 JSON 출력) |

---

## 3. 쿼리 확장 구현 상세

### 3.1 함수 정의

**파일**: `backend/open_webui/routers/tasks.py` L428-503

```python
@router.post('/queries/completions')
async def generate_queries(request: Request, form_data: dict, user=...):
    type = form_data.get('type')  # 'web_search' 또는 'retrieval'

    # 1. 프롬프트 템플릿 선택
    if request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE.strip() != '':
        template = request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE

    # 2. 대화 히스토리를 포함한 프롬프트 조립
    content = query_generation_template(template, form_data['messages'], user)

    # 3. Task Model(보조 LLM)에게 쿼리 생성 요청
    payload = {
        'model': task_model_id,
        'messages': [{'role': 'user', 'content': content}],
        'stream': False,
        'metadata': {'task': 'query_generation'},
    }

    return await generate_chat_completion(request, form_data=payload, user=user)
```

### 3.2 프롬프트 템플릿

**파일**: `backend/open_webui/config.py` L1845-1867

```
### Task:
Analyze the chat history to determine the necessity of generating search
queries, in the given language. By default, **prioritize generating 1-3 broad
and relevant search queries** unless it is absolutely certain that no additional
information is required.

### Guidelines:
- Respond **EXCLUSIVELY** with a JSON object.
- When generating search queries, respond in the format:
  { "queries": ["query1", "query2"] }
- If no search is needed, return: { "queries": [] }
- Err on the side of suggesting search queries if there is **any chance**
  they might provide useful information.
- Today's date is: {{CURRENT_DATE}}.

### Output:
Strictly return in JSON format:
{
  "queries": ["query1", "query2"]
}

### Chat History:
<chat_history>
{{MESSAGES:END:6}}
</chat_history>
```

**핵심 특징**:
- 대화 히스토리 최근 6개 메시지를 컨텍스트로 제공
- 검색이 불필요하면 빈 배열 반환 가능 (`{"queries": []}`)
- 1~3개의 넓고 관련성 있는 쿼리 생성 유도
- JSON 형식 강제

### 3.3 호출 지점 (2곳)

#### 호출 1: RAG 파일 검색 시

**파일**: `backend/open_webui/utils/middleware.py` L1870-1923

```python
async def chat_completion_files_handler(request, body, extra_params, user):
    files = body.get('metadata', {}).get('files', None)

    # 모든 파일이 Full Context 모드가 아닐 때만 쿼리 확장 실행
    all_full_context = all(item.get('context') == 'full' for item in files)

    queries = []
    if not all_full_context:
        # Task Model에게 검색 쿼리 생성 요청
        queries_response = await generate_queries(
            request,
            {'model': body['model'], 'messages': body['messages'], 'type': 'retrieval'},
            user,
        )
        queries = queries_response.get('queries', [])

    # 쿼리 생성 실패 시 원본 메시지를 그대로 사용
    if len(queries) == 0:
        queries = [get_last_user_message(body['messages'])]

    # 생성된 쿼리들로 벡터 DB 검색
    sources = await get_sources_from_items(queries=queries, ...)
```

#### 호출 2: 웹 검색 시

**파일**: `backend/open_webui/utils/middleware.py` L1428-1478

```python
async def chat_web_search_handler(request, form_data, extra_params, user):
    # Task Model에게 웹 검색 쿼리 생성 요청
    res = await generate_queries(
        request,
        {'model': form_data['model'], 'messages': messages, 'type': 'web_search'},
        user,
    )
    queries = json.loads(response).get('queries', [])

    # 실패 시 원본 메시지 사용
    if len(queries) == 0 or queries[0].strip() == '':
        queries = [user_message]
```

### 3.4 활성화/비활성화 설정

| 환경변수 | 기본값 | 역할 |
|---------|--------|------|
| `ENABLE_RETRIEVAL_QUERY_GENERATION` | `true` | RAG 파일 검색 시 쿼리 확장 |
| `ENABLE_SEARCH_QUERY_GENERATION` | `true` | 웹 검색 시 쿼리 확장 |
| `QUERY_GENERATION_PROMPT_TEMPLATE` | `""` (기본 템플릿 사용) | 커스텀 프롬프트 |

---

## 4. 쿼리 확장이 검색에 미치는 영향

### 4.1 쿼리 확장 없이 (비활성 시)

```
사용자: "우리 회사 복리후생 제도 알려줘"
       │
       ▼
단일 쿼리 임베딩: "우리 회사 복리후생 제도 알려줘"
       │
       ▼
벡터 DB 검색: TOP_K=3
       │
결과: "복리후생" 키워드와 의미적으로 가까운 청크 3개만 반환
      → 건강검진, 자녀학자금 등 다른 복리후생 항목은 놓칠 수 있음
```

### 4.2 쿼리 확장 사용 시 (기본 동작)

```
사용자: "우리 회사 복리후생 제도 알려줘"
       │
       ▼
Task Model이 쿼리 생성:
  ["회사 복리후생 제도", "직원 건강검진 지원", "자녀학자금 휴가 정책"]
       │
       ▼
쿼리 1: "회사 복리후생 제도"     → 청크 A, B, C
쿼리 2: "직원 건강검진 지원"     → 청크 D, E, F
쿼리 3: "자녀학자금 휴가 정책"   → 청크 G, A, H
       │
       ▼
병합 + 정렬: A, B, C, D, E, F, G, H (중복 제거)
       │
       ▼
TOP_K=3 적용 → 가장 관련도 높은 3개 반환
```

### 4.3 효과

| 지표 | 쿼리 확장 없음 | 쿼리 확장 사용 |
|------|--------------|--------------|
| **검색 쿼리 수** | 1개 | 1~3개 |
| **후보 문서 풀** | 좁음 | 넓음 |
| **재현율 (Recall)** | 낮음 | 높음 |
| **LLM 추가 비용** | 없음 | Task Model 1회 호출 |
| **지연 시간** | 없음 | +0.5~2초 (쿼리 생성) |

---

## 5. 전체 검색 파이프라인에서의 위치

```
사용자 질의
    │
    ▼
┌──────────────────────────────────────────┐
│ ① 쿼리 확장 (Query Expansion)            │  ← 이 보고서의 주제
│    generate_queries()                     │
│    "복리후생 알려줘"                       │
│     → ["복리후생 제도", "건강검진", "휴가"] │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ ② 1차 검색 (Retrieval)                   │
│    ├─ 벡터 유사도 검색 (Bi-Encoder)       │
│    └─ BM25 키워드 검색 (하이브리드 시)     │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ ③ 결과 융합 (Fusion)                     │
│    EnsembleRetriever + RRF              │
│    (하이브리드 검색 시에만)                │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ ④ 리랭킹 (Reranking)                    │
│    Cross-Encoder / ColBERT / External   │
│    (하이브리드 + 리랭킹 모델 설정 시)      │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ ⑤ 프롬프트 주입 + LLM 호출               │
│    RAG Template에 검색 결과 삽입          │
└──────────────────────────────────────────┘
```

**쿼리 확장은 파이프라인의 가장 첫 단계**로, 이후 모든 검색의 품질을 좌우합니다.

---

## 6. HyDE를 도입하려면?

Open WebUI에 HyDE를 추가하려면 Pipeline/Function으로 구현 가능합니다:

```python
# 개념적 구현 — Pipe Function으로 작성 가능
async def hyde_query_expansion(user_query, llm_call, embedding_function):
    # 1. 가상 답변 생성
    hypothetical_answer = await llm_call(
        f"다음 질문에 대해 상세히 답변해주세요 (검색 없이): {user_query}"
    )

    # 2. 가상 답변을 임베딩
    hyde_embedding = await embedding_function(hypothetical_answer)

    # 3. 가상 답변 임베딩으로 벡터 검색
    results = vector_db.search(vectors=[hyde_embedding], limit=TOP_K)

    return results
```

### HyDE 도입 시 트레이드오프

| 장점 | 단점 |
|------|------|
| 쿼리-문서 의미 간극 해소 | LLM이 틀린 답변 생성 시 엉뚱한 검색 |
| 짧은 질문도 풍부한 검색 가능 | 긴 텍스트 생성 비용 (토큰 소모) |
| 전문 용어 매칭 개선 | 지연 시간 증가 |

### 쿼리 확장 + HyDE 병행이 가장 효과적

```
사용자: "양자 컴퓨팅의 장점은?"
       │
       ├─ 쿼리 확장: ["양자 컴퓨팅 장점", "큐비트 병렬 연산"]  ← 재현율 ↑
       │
       └─ HyDE: "양자 컴퓨팅은 큐비트를 활용하여..."          ← 정밀도 ↑
              │
              ▼
         3가지 벡터를 모두 검색 → 결과 병합 → 리랭킹
```

---

## 7. 주요 파일 참조

| 파일 | 역할 | 핵심 라인 |
|------|------|----------|
| `routers/tasks.py` | `generate_queries()` 함수 정의 | L428-503 |
| `config.py` | `DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE` | L1845-1867 |
| `config.py` | `ENABLE_RETRIEVAL_QUERY_GENERATION` | L1832-1835 |
| `config.py` | `ENABLE_SEARCH_QUERY_GENERATION` | L1826-1829 |
| `config.py` | `QUERY_GENERATION_PROMPT_TEMPLATE` | L1839-1842 |
| `utils/middleware.py` | RAG 파일 검색 시 호출 | L1883 |
| `utils/middleware.py` | 웹 검색 시 호출 | L1446 |
| `utils/middleware.py` | 쿼리 생성 실패 시 Fallback | L1922-1923 |

---

## 8. 결론

Open WebUI는 **HyDE 대신 Multi-Query Generation**을 선택했으며, 이는 다음과 같은 이유로 합리적인 설계입니다:

1. **예측 가능성**: JSON 형식의 쿼리 목록은 파싱이 안정적이고, 가상 답변은 품질 변동이 큼
2. **비용 효율**: 짧은 쿼리 3개 생성 < 긴 가상 답변 1개 생성 (토큰 소모)
3. **범용성**: 웹 검색과 RAG 파일 검색 모두 동일한 쿼리 확장 로직 재사용
4. **안전성**: 가상 답변이 틀리면 검색 자체가 오염되지만, 쿼리 확장은 최악의 경우 원본 메시지로 Fallback

다만, 전문 도메인(의학, 법률 등)에서 질문과 문서 간 용어 차이가 큰 경우에는 HyDE가 더 효과적일 수 있으며, 이는 Pipeline/Function으로 추가 구현 가능합니다.
