# Open WebUI 대용량 문서 처리 흐름 분석 보고서

## 1. 핵심 결론

> **Open WebUI는 Map-Reduce, Refine, Stuff Chain 등의 다단계 요약 패턴을 사용하지 않는다.**
> 대신, **단순 RAG (Retrieve → Inject → Single LLM Call)** 방식을 사용한다.

큰 문서를 업로드하고 "이 문서를 요약해줘"라고 하면:

```
문서 → 청크 분할 → 벡터 DB 저장 → 유사도 검색 → TOP_K 청크 추출 → 프롬프트에 주입 → LLM 1회 호출
```

**즉, 문서 전체를 요약하는 것이 아니라, 질의와 가장 관련 있는 일부 청크만 가져와서 LLM에게 전달합니다.**

---

## 2. Map-Reduce vs Open WebUI 비교

### 2.1 학술적 문서 요약 패턴들

```
┌─────────────────────────────────────────────────────────────────┐
│ Stuff (단순 주입)                                                │
│   [전체 문서] → LLM 1회 호출 → 요약                              │
│   ⚠ 컨텍스트 윈도우 초과 시 불가                                  │
├─────────────────────────────────────────────────────────────────┤
│ Map-Reduce (분할 요약 + 병합)                                    │
│   [청크1] → LLM → 요약1 ┐                                      │
│   [청크2] → LLM → 요약2 ├→ [요약1+2+3] → LLM → 최종 요약       │
│   [청크3] → LLM → 요약3 ┘                                      │
│   ✅ 대용량 가능, 비용 높음                                      │
├─────────────────────────────────────────────────────────────────┤
│ Refine (점진적 개선)                                             │
│   [청크1] → LLM → 초안 → [초안+청크2] → LLM → 개선              │
│                          → [개선+청크3] → LLM → 최종             │
│   ✅ 순서 보존, 느림                                             │
├─────────────────────────────────────────────────────────────────┤
│ ★ Open WebUI의 RAG Injection (검색 기반 주입)                    │
│   [전체 문서] → 청크 분할 → 벡터 DB 저장                          │
│   [사용자 질의] → 임베딩 → TOP_K 검색 → 프롬프트 주입 → LLM 1회  │
│   ⚠ 질의와 무관한 내용은 누락됨                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Open WebUI가 Map-Reduce를 쓰지 않는 이유

- **범용 채팅 UI**이지 문서 요약 전용 도구가 아님
- LLM API 호출 최소화 (비용/속도)
- 대부분의 사용 시나리오는 "문서에서 특정 정보 찾기"이지 "전체 요약"이 아님
- LangChain의 Chain 패턴을 의존성으로 가져오지 않음 (RAG 유틸만 사용)

---

## 3. "이 문서를 요약해줘" 실행 시 전체 흐름

### 3.1 Phase 1: 문서 업로드 (사전 단계)

```
사용자가 PDF 첨부
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  프론트엔드: MessageInput.svelte                          │
│  uploadFileHandler() → POST /api/v1/files/               │
│  파일을 백엔드로 업로드                                     │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  백엔드: routers/retrieval.py                             │
│  1. 콘텐츠 추출 (PDF→텍스트)                               │
│  2. RecursiveCharacterTextSplitter로 청크 분할             │
│     - CHUNK_SIZE = 1000 (기본값)                          │
│     - CHUNK_OVERLAP = 100 (기본값)                        │
│  3. 각 청크에 대해 임베딩 벡터 생성                          │
│     - 모델: qwen3-embedding:latest (Ollama)              │
│  4. 벡터 DB (ChromaDB)에 저장                             │
│     - collection_name = "file-{file_id}"                 │
└──────────────────────────────────────────────────────────┘
```

**예시**: 50페이지 PDF → 약 100~200개 청크로 분할 → 벡터 DB에 저장

### 3.2 Phase 2: 메시지 전송

```
사용자: "이 문서를 요약해줘" + 첨부파일 참조
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  프론트엔드: Chat.svelte                                  │
│  POST /api/chat/completions                              │
│  {                                                       │
│    "model": "frentis-ai-model",                          │
│    "messages": [{"role": "user", "content": "이 문서를    │
│                  요약해줘"}],                              │
│    "files": [{"type": "file", "id": "file-abc123",       │
│               "collection_name": "file-abc123"}]         │
│  }                                                       │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  백엔드: main.py → process_chat_payload()                 │
│  metadata['files'] = files                               │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Phase 3: 검색 쿼리 생성

**파일**: `middleware.py` L1882-1923

```
사용자 메시지: "이 문서를 요약해줘"
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  generate_queries()                                       │
│  Task Model에게 검색 쿼리 생성 요청                         │
│                                                          │
│  입력: "이 문서를 요약해줘"                                 │
│  출력: ["문서 주요 내용", "핵심 결론", "문서 개요"]           │
│        (또는 원본 메시지를 그대로 사용)                       │
└──────────┬───────────────────────────────────────────────┘
           ▼
```

### 3.4 Phase 4: 벡터 검색 (TOP_K 청크 추출)

**파일**: `retrieval/utils.py` L926-1178

```
검색 쿼리: ["문서 주요 내용", "핵심 결론", "문서 개요"]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  get_sources_from_items()                                 │
│                                                          │
│  각 쿼리 → 임베딩 벡터 생성 (qwen3-embedding)             │
│         → 벡터 DB에서 유사도 검색                          │
│         → TOP_K=3 (기본값) 개 청크 반환                    │
│                                                          │
│  ┌─────────────────────────────────┐                     │
│  │ 벡터 DB (200개 청크 중)          │                     │
│  │                                 │                     │
│  │ 쿼리1 → 청크#42 (0.89)         │                     │
│  │         청크#43 (0.85)         │                     │
│  │         청크#1  (0.82)         │                     │
│  │                                 │                     │
│  │ 쿼리2 → 청크#98 (0.91)         │                     │
│  │         청크#42 (0.87) ← 중복  │                     │
│  │         청크#55 (0.80)         │                     │
│  │                                 │                     │
│  │ 쿼리3 → 청크#1  (0.88) ← 중복  │                     │
│  │         청크#2  (0.84)         │                     │
│  │         청크#99 (0.79)         │                     │
│  └─────────────────────────────────┘                     │
│                                                          │
│  결과: 약 5~9개 고유 청크 (중복 제거 후)                    │
│  = 전체 문서의 약 3~5%만 추출                              │
└──────────┬───────────────────────────────────────────────┘
           ▼
```

### 3.5 Phase 5: 프롬프트 컨텍스트 조립

**파일**: `middleware.py` L879-935, `task.py` L246

```
추출된 청크들
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  get_source_context() → 청크를 <source> 태그로 래핑       │
│                                                          │
│  <source id="1" name="report.pdf">                       │
│    ... 청크 #1 텍스트 (서론 부분) ...                      │
│  </source>                                               │
│  <source id="2" name="report.pdf">                       │
│    ... 청크 #42 텍스트 (핵심 결론 부분) ...                 │
│  </source>                                               │
│  <source id="3" name="report.pdf">                       │
│    ... 청크 #98 텍스트 (방법론 부분) ...                    │
│  </source>                                               │
│  ... (총 5~9개)                                           │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  rag_template() → RAG 템플릿에 주입                       │
│                                                          │
│  ### Task:                                               │
│  Respond to the user query using the provided context,   │
│  incorporating inline citations in the format [id]...    │
│                                                          │
│  <context>                                               │
│  <source id="1">...청크1...</source>                     │
│  <source id="2">...청크2...</source>                     │
│  ...                                                     │
│  </context>                                              │
└──────────┬───────────────────────────────────────────────┘
           ▼
```

### 3.6 Phase 6: LLM 호출 (1회)

```
최종 메시지 배열
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  messages = [                                            │
│    {                                                     │
│      "role": "system",                                   │
│      "content": "### Task:\n                             │
│        Respond to the user query using the provided      │
│        context...\n                                      │
│        <context>\n                                       │
│        <source id=\"1\">...서론 부분...</source>\n        │
│        <source id=\"2\">...결론 부분...</source>\n        │
│        <source id=\"3\">...방법론 부분...</source>\n      │
│        </context>"                                       │
│    },                                                    │
│    {                                                     │
│      "role": "user",                                     │
│      "content": "이 문서를 요약해줘"                       │
│    }                                                     │
│  ]                                                       │
│                                                          │
│  → POST http://ai-server.dream-flow.com:30000/v1/       │
│    chat/completions                                      │
│    model: frentis-ai-model                               │
│                                                          │
│  ★ LLM은 전체 문서가 아닌, 검색된 일부 청크만 보고 응답     │
└──────────────────────────────────────────────────────────┘
```

---

## 4. "요약"이 불완전한 이유

### 4.1 정보 손실 구조

```
50페이지 PDF 문서
│
├─ 청크 200개로 분할
│
├─ 벡터 검색으로 TOP_K=3~9개 추출
│   (= 전체의 약 2~5%)
│
└─ LLM은 이 2~5%만 보고 "요약" 생성
   → 나머지 95%는 LLM이 볼 수 없음
```

### 4.2 "이 문서를 요약해줘"의 문제점

| 문제 | 설명 |
|------|------|
| **부분 요약** | 문서 전체가 아닌, 검색된 일부 청크만 기반으로 요약 |
| **쿼리 편향** | "요약"이라는 쿼리와 유사한 청크만 추출 (서론/결론 편중) |
| **중간 내용 누락** | 본론, 세부 데이터, 표 등 유사도가 낮은 부분은 빠짐 |
| **구조 상실** | 문서의 논리적 흐름/순서가 보존되지 않음 |

---

## 5. Full Context 모드 — 대안

Open WebUI에는 **Full Context 모드**가 있어, 벡터 검색 없이 문서 전체를 LLM에 주입할 수 있습니다.

### 5.1 활성화 조건

**파일**: `retrieval/utils.py` L1028

```python
if item.get('context') == 'full' or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:
    # 전체 문서 내용을 직접 사용 (벡터 검색 건너뜀)
    query_result = {
        'documents': [[file_content]],  # 전체 텍스트
        'metadatas': [[metadata]]
    }
```

### 5.2 설정 방법

```bash
# .env에 추가
BYPASS_EMBEDDING_AND_RETRIEVAL=true   # 모든 파일에 대해 전체 컨텍스트 사용
# 또는
RAG_FULL_CONTEXT=true                  # RAG에서 전체 컨텍스트 모드
```

또는 파일 아이템에 `context: 'full'` 속성 설정 (프론트엔드에서)

### 5.3 Full Context 모드의 한계

```
문서 크기 vs 컨텍스트 윈도우

문서: 50페이지 ≈ 25,000 토큰
모델 컨텍스트: frentis-ai-model = ?

├─ 컨텍스트 내: 전체 주입 가능 → 완전한 요약 가능
└─ 컨텍스트 초과: 뒷부분 잘림 → 불완전한 요약
   (Open WebUI에는 잘림 경고나 자동 분할 로직 없음)
```

---

## 6. 두 모드 비교

| 항목 | RAG 모드 (기본) | Full Context 모드 |
|------|----------------|------------------|
| **문서 처리** | 청크 분할 → 벡터 검색 → TOP_K 추출 | 전체 텍스트 직접 주입 |
| **LLM 호출 횟수** | 1회 | 1회 |
| **정보 완전성** | 부분적 (2~5%) | 완전 (100%) |
| **컨텍스트 초과 위험** | 낮음 (TOP_K로 제한) | 높음 (대용량 문서 시) |
| **요약 품질** | 핵심만 추출, 편향 가능 | 전체 기반, 정확 |
| **비용/속도** | 빠름 (작은 프롬프트) | 느림 (큰 프롬프트) |
| **적합한 용도** | Q&A, 특정 정보 검색 | 전체 요약, 분석 |

---

## 7. 관련 설정 파라미터

### 7.1 검색 품질 관련

| 환경변수 | 기본값 | 역할 | 요약 시 권장 |
|---------|--------|------|------------|
| `TOP_K` | 3 | 쿼리당 가져올 청크 수 | 10~20으로 상향 |
| `CHUNK_SIZE` | 1000 | 청크 크기 (문자) | 유지 |
| `CHUNK_OVERLAP` | 100 | 청크 간 겹침 | 유지 |
| `RELEVANCE_THRESHOLD` | 0.0 | 최소 유사도 | 유지 |
| `ENABLE_RAG_HYBRID_SEARCH` | false | BM25 하이브리드 검색 | true 권장 |
| `RAG_FULL_CONTEXT` | false | 전체 컨텍스트 모드 | 요약 시 true |
| `BYPASS_EMBEDDING_AND_RETRIEVAL` | false | 임베딩/검색 건너뛰기 | 요약 시 true |

### 7.2 컨텍스트 주입 관련

| 환경변수 | 기본값 | 역할 |
|---------|--------|------|
| `RAG_SYSTEM_CONTEXT` | false | true=시스템 메시지, false=유저 메시지로 주입 |
| `RAG_TEMPLATE` | (긴 템플릿) | 컨텍스트 주입 템플릿 |

---

## 8. 만약 Map-Reduce 요약을 원한다면?

Open WebUI 자체에는 없지만, 다음 방법으로 구현 가능합니다:

### 8.1 방법 1: Pipeline/Function으로 커스텀 구현

Open WebUI의 **Pipe Function**으로 Map-Reduce 로직을 직접 구현:

```python
# 개념적 코드 (실제 Pipe Function으로 작성 가능)
async def summarize_large_document(file_content, model):
    # Map: 각 청크 요약
    chunks = split_into_chunks(file_content, chunk_size=4000)
    chunk_summaries = []
    for chunk in chunks:
        summary = await llm_call(f"다음 텍스트를 3문장으로 요약:\n{chunk}")
        chunk_summaries.append(summary)

    # Reduce: 요약들을 합쳐서 최종 요약
    combined = "\n".join(chunk_summaries)
    final_summary = await llm_call(f"다음 부분 요약들을 종합하여 최종 요약 작성:\n{combined}")
    return final_summary
```

### 8.2 방법 2: Full Context + 큰 컨텍스트 모델 사용

```bash
# .env 설정
BYPASS_EMBEDDING_AND_RETRIEVAL=true
```

frentis-ai-model의 컨텍스트 윈도우가 충분히 크다면 (128K 등), 전체 문서를 한 번에 주입하여 요약 가능.

### 8.3 방법 3: TOP_K를 크게 설정

```bash
# .env 설정
TOP_K=50  # 기본 3에서 50으로 상향
```

더 많은 청크를 가져와서 커버리지를 높임. 완전하지는 않지만 개선됨.

---

## 9. 주요 파일 참조

| 파일 | 역할 | 핵심 라인 |
|------|------|----------|
| `utils/middleware.py` | 파일 처리 및 RAG 주입 오케스트레이션 | L1870-1978 (files_handler), L2699 (호출부) |
| `retrieval/utils.py` | 벡터 검색 및 소스 조립 | L926-1178 (get_sources_from_items), L407-475 (query_collection) |
| `utils/task.py` | RAG 템플릿 적용 | L246-282 (rag_template) |
| `routers/retrieval.py` | 문서 업로드 시 청킹/임베딩 | L1258-1420 (text_splitter), L1529-1700 (save_docs) |
| `config.py` | TOP_K, CHUNK_SIZE 등 설정 | L2786-2950 |

---

## 10. 결론

Open WebUI에서 대용량 문서 + "이 문서를 요약해줘":

1. **Map-Reduce 아님** — 청크별 요약 → 병합 로직 없음
2. **Refine 아님** — 점진적 요약 개선 로직 없음
3. **단순 RAG** — 벡터 검색으로 TOP_K 청크만 추출 → LLM 1회 호출
4. **결과**: 문서의 **2~5%만 보고 생성한 "부분 요약"**이 됨
5. **대안**: `BYPASS_EMBEDDING_AND_RETRIEVAL=true`로 전체 문서 주입 가능 (컨텍스트 윈도우 한도 내)

이는 Open WebUI가 "문서 요약 도구"가 아닌 **"LLM 채팅 인터페이스 + RAG 보조"**로 설계되었기 때문입니다. 진정한 대용량 문서 요약이 필요하면 Pipeline/Function으로 Map-Reduce 로직을 별도 구현해야 합니다.
