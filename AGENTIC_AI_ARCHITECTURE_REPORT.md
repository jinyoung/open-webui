# Open WebUI 에이전틱 AI 아키텍처 분석 보고서

## 1. 핵심 결론

> **Open WebUI는 LangGraph, LangChain Agent, ReAct, Plan-and-Solve 등의 외부 에이전트 프레임워크를 사용하지 않는다.**
> 대신, **자체 구현한 Tool Call Loop (도구 호출 반복 루프)** 패턴으로 에이전틱 동작을 구현한다.

| 프레임워크/패턴 | 사용 여부 | 비고 |
|----------------|----------|------|
| **LangGraph** | ❌ 미사용 | import 0건, 의존성 0건 |
| **LangChain Agent (AgentExecutor)** | ❌ 미사용 | import 0건 |
| **ReAct (Thought→Action→Observation)** | ❌ 미사용 | 해당 패턴 없음 |
| **Plan-and-Solve** | ❌ 미사용 | 플래닝 단계 없음 |
| **LangChain (문서 처리만)** | ✅ 사용 | 로더, 스플리터, 리트리버만 |
| **자체 Tool Call Loop** | ✅ 사용 | middleware.py 핵심 루프 |

---

## 2. LangChain 사용 범위 — RAG 인프라에만 한정

Open WebUI는 LangChain을 **에이전트가 아닌 RAG 파이프라인의 유틸리티**로만 사용합니다.

### 2.1 의존성 목록 (pyproject.toml)

```
langchain==1.2.10              # 기본 패키지
langchain-community==0.4.1     # 문서 로더
langchain-classic==1.0.1       # 리트리버
langchain-text-splitters==1.1.1 # 텍스트 분할
```

> `langgraph`, `langchain-agents` 등의 에이전트 관련 패키지는 **전혀 포함되어 있지 않습니다**.

### 2.2 LangChain이 실제 사용되는 곳

| 용도 | 파일 | 사용 클래스 |
|------|------|------------|
| 문서 로딩 | `retrieval/loaders/main.py` | `PyPDFLoader`, `CSVLoader`, `Docx2txtLoader` 등 |
| 텍스트 분할 | `routers/retrieval.py` | `RecursiveCharacterTextSplitter`, `TokenTextSplitter` |
| 검색 리트리버 | `retrieval/utils.py` | `BM25Retriever`, `EnsembleRetriever`, 커스텀 `VectorSearchRetriever` |
| 함수 스펙 변환 | `utils/tools.py` | `convert_to_openai_function` (Pydantic → OpenAI 함수 스펙) |

LangChain의 `AgentExecutor`, `create_react_agent`, `StateGraph` 등 에이전트 관련 모듈은 **단 한 번도 import되지 않습니다**.

---

## 3. Open WebUI의 에이전틱 아키텍처 — 자체 구현

### 3.1 아키텍처 개요

Open WebUI는 **OpenAI Function Calling 프로토콜**을 기반으로 한 **단순 반복 루프(Iterative Tool Call Loop)** 패턴을 직접 구현합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Open WebUI Agent Loop                     │
│                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────────┐       │
│   │ 사용자    │     │ LLM      │     │ 도구 실행기   │       │
│   │ 메시지    │────→│ 호출     │────→│ (Tool Exec)  │       │
│   └──────────┘     └────┬─────┘     └──────┬───────┘       │
│                         │                   │               │
│                         │    tool_calls?    │               │
│                         │◄──────────────────│               │
│                         │                   │               │
│                    ┌────▼─────┐             │               │
│                    │ tool_call│    YES      │               │
│                    │ 있는가?  │────────────→│               │
│                    └────┬─────┘             │               │
│                         │ NO               │               │
│                         ▼                   │               │
│                    ┌──────────┐             │               │
│                    │ 최종 응답 │             │               │
│                    │ 반환     │             │               │
│                    └──────────┘             │               │
│                                             │               │
│   최대 반복: 30회 (CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES)     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 학술적 패턴과의 비교

| 패턴 | 구조 | Open WebUI |
|------|------|-----------|
| **ReAct** | Thought → Action → Observation 명시적 루프 | ❌ Thought/Observation 단계 없음 |
| **Plan-and-Solve** | Plan 생성 → 단계별 실행 | ❌ 사전 플래닝 단계 없음 |
| **LangGraph** | 상태 그래프 + 조건부 분기 | ❌ 그래프 구조 없음 |
| **Tool Call Loop** | LLM → tool_call → 실행 → 결과 주입 → 재호출 | ✅ **이것을 사용** |
| **Function Calling** | OpenAI 프로토콜 기반 도구 호출 | ✅ **기반 프로토콜** |

Open WebUI의 접근법은 **LLM 자체의 추론 능력에 의존**합니다. 별도의 Thought/Plan 단계를 코드로 강제하지 않고, LLM이 tool_calls를 반환하면 실행하고, 그 결과를 다시 LLM에게 전달하는 **최소한의 루프**입니다.

---

## 4. 두 가지 도구 호출 모드

Open WebUI는 도구 호출을 위한 **두 가지 모드**를 제공합니다.

### 4.1 Native Function Calling (네이티브 모드)

LLM이 직접 `tool_calls`를 생성하는 방식입니다.

```
사용자 메시지 + 도구 스펙 (tools=[...])
         │
         ▼
    ┌─────────┐
    │   LLM   │  ← 모델이 직접 tool_calls JSON을 생성
    └────┬────┘
         │  {"tool_calls": [{"function": {"name": "search_web", "arguments": "{...}"}}]}
         ▼
    ┌─────────────┐
    │ 도구 실행    │  ← Open WebUI가 실행
    └────┬────────┘
         │  결과를 메시지로 변환
         ▼
    ┌─────────┐
    │   LLM   │  ← 도구 결과 포함하여 재호출
    └────┬────┘
         │  (더 이상 tool_calls 없으면 종료)
         ▼
    최종 응답 반환
```

**적용 조건**: LLM이 Function Calling을 네이티브로 지원하는 경우 (GPT-4, Claude, etc.)

**코드 위치**: `middleware.py` L2678-2683
```python
# 네이티브 모드: 도구 스펙을 LLM 요청에 직접 포함
if metadata.get("function_calling") == "native":
    form_data["tools"] = [
        {"type": "function", "function": tool["spec"]}
        for tool in tools.values()
    ]
```

### 4.2 Default Function Calling (래퍼 모드)

LLM이 Function Calling을 지원하지 않을 때, **보조 LLM(Task Model)**이 도구 선택을 대신하는 방식입니다.

```
사용자 메시지
    │
    ▼
┌───────────────────┐
│ Task Model (보조)  │  ← 별도 LLM이 도구 스펙을 보고 어떤 도구를 호출할지 결정
│                   │
│ 프롬프트:          │
│ "다음 도구 중에서   │
│  적절한 것을 골라   │
│  JSON으로 반환"    │
└────────┬──────────┘
         │  {"tool_calls": [{"name": "search_web", "parameters": {...}}]}
         ▼
┌─────────────┐
│ 도구 실행    │
└────┬────────┘
         │
         ▼
┌───────────────────┐
│ Main Model (메인)  │  ← 도구 결과를 컨텍스트에 주입하여 최종 응답 생성
└───────────────────┘
```

**적용 조건**: Function Calling 미지원 모델 또는 사용자가 "default" 모드로 설정한 경우

**코드 위치**: `middleware.py` L1181-1390 (`chat_completion_tools_handler`)

---

## 5. 핵심 Agent Loop 코드 분석

### 5.1 메인 에이전트 루프

**파일**: `backend/open_webui/utils/middleware.py` L4065-4432

```python
tool_call_retries = 0

# ======== AGENT LOOP ========
while len(tool_calls) > 0 and tool_call_retries < CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES:
    tool_call_retries += 1

    # 1단계: 대기 중인 tool_calls 배치 가져오기
    response_tool_calls = tool_calls.pop(0)

    # 2단계: 각 도구 실행
    for tool_call in response_tool_calls:
        tool_function_name = tool_call['function']['name']
        tool_function_params = json.loads(tool_call['function']['arguments'])

        if is_direct_tool:
            # MCP/외부 서버 도구 → 클라이언트로 이벤트 전송
            tool_result = await event_caller({
                'type': 'execute:tool',
                'data': { 'name': tool_function_name, 'params': tool_function_params }
            })
        else:
            # 로컬 도구 → 직접 함수 호출
            tool_result = await tool_function(**tool_function_params)

    # 3단계: 도구 결과를 대화에 추가
    output.append({
        'type': 'function_call_output',
        'call_id': tool_call_id,
        'output': [{'type': 'input_text', 'text': result_content}],
        'status': 'completed',
    })

    # 4단계: 도구 결과 포함하여 LLM 재호출
    tool_messages = convert_output_to_messages(output, raw=True)
    new_form_data['messages'] = [*original_messages, *tool_messages]

    res = await generate_chat_completion(
        request, new_form_data, user,
        bypass_system_prompt=True,  # 시스템 프롬프트 중복 방지
    )

    # 5단계: 새 응답에서 tool_calls 감지 → 있으면 루프 계속, 없으면 종료
    if new_tool_calls_detected:
        tool_calls.append(new_tool_calls)
    else:
        break  # 루프 종료, 최종 응답 반환
```

### 5.2 최대 반복 횟수

**파일**: `backend/open_webui/env.py` L622-630

```python
CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES = int(
    os.environ.get('CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES', '30')
)
```

- **기본값**: 30회
- 무한 루프 방지를 위한 안전장치
- 환경변수로 조정 가능

---

## 6. 도구(Tool) 시스템 전체 구조

### 6.1 도구의 4가지 유형

```
┌─────────────────────────────────────────────────────────┐
│                    Open WebUI Tool System                │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Built-in    │  │ Custom      │  │ MCP Server      │ │
│  │ Tools       │  │ Tools       │  │ Tools           │ │
│  │ (내장 도구)  │  │ (사용자 정의)│  │ (외부 프로토콜)  │ │
│  ├─────────────┤  ├─────────────┤  ├─────────────────┤ │
│  │• search_web │  │• Python     │  │• MCP 프로토콜로  │ │
│  │• fetch_url  │  │  함수로 작성 │  │  연결된 외부     │ │
│  │• execute_   │  │• UI에서     │  │  서비스          │ │
│  │  code       │  │  등록/관리   │  │• client.py로    │ │
│  │• generate_  │  │• DB에 저장  │  │  통신            │ │
│  │  image      │  │             │  │                 │ │
│  │• memory_*   │  │             │  │                 │ │
│  │• knowledge_*│  │             │  │                 │ │
│  │• notes_*    │  │             │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Tool Server (OpenAPI)                               │ │
│  │ • OpenAPI 스펙 기반 HTTP 엔드포인트를 도구로 변환     │ │
│  │ • REST API를 자동으로 도구 함수로 래핑               │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 6.2 내장 도구 목록

**파일**: `backend/open_webui/tools/builtin.py` (2,274줄)

| 카테고리 | 도구 함수 | 설명 |
|---------|----------|------|
| **시간** | `get_current_timestamp()` | 현재 시간 조회 |
| | `calculate_timestamp()` | 시간 계산 |
| **웹** | `search_web()` | 웹 검색 |
| | `fetch_url()` | URL 내용 가져오기 |
| **이미지** | `generate_image()` | 이미지 생성 |
| | `edit_image()` | 이미지 편집 |
| **코드** | `execute_code()` | 코드 실행 (Pyodide/Jupyter) |
| **메모리** | `search_memories()` | 기억 검색 |
| | `add_memory()` | 기억 추가 |
| | `replace_memory_content()` | 기억 수정 |
| | `delete_memory()` | 기억 삭제 |
| **노트** | `search_notes()`, `write_note()` | 노트 검색/작성 |
| **지식** | `list_knowledge_bases()` | 지식 베이스 목록 |
| | `query_knowledge_bases()` | 지식 베이스 질의 |
| | `search_knowledge_files()` | 지식 파일 검색 |
| **채팅** | `search_chats()`, `view_chat()` | 과거 대화 검색 |
| **채널** | `search_channels()` | 채널 메시지 검색 |

### 6.3 도구 실행 흐름 상세

```
도구 등록 (DB 저장)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ process_chat_payload() — middleware.py:2109                  │
│                                                             │
│ 1. tool_ids 추출 (사용자가 선택한 도구 목록)                    │
│ 2. get_tools() 호출 — 도구 모듈 로딩 및 래핑                   │
│    ├─ 로컬 도구: Python 모듈 → async 함수로 래핑               │
│    ├─ MCP 도구: MCPClient.list_tool_specs() → 래핑            │
│    └─ 서버 도구: OpenAPI 스펙 → HTTP 요청 함수로 래핑           │
│ 3. 도구 스펙을 OpenAI Function Calling 형식으로 변환            │
│ 4. 모드 결정:                                                │
│    ├─ Native: form_data['tools'] = specs                    │
│    └─ Default: chat_completion_tools_handler() 호출          │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 도구 실행 모드별 상세 비교

### 7.1 Native vs Default 비교표

| 항목 | Native Mode | Default Mode |
|------|------------|--------------|
| **도구 선택 주체** | 메인 LLM이 직접 결정 | 보조 Task Model이 결정 |
| **tool_calls 생성** | LLM 응답의 `tool_calls` 필드 | 보조 LLM의 JSON 출력 파싱 |
| **루프 구조** | while 루프 (최대 30회) | 단일 호출 (1회) |
| **적용 대상** | GPT-4, Claude 등 FC 지원 모델 | Llama, Mistral 등 미지원 모델 |
| **설정 방법** | `function_calling: "native"` | `function_calling: "default"` (기본값) |
| **장점** | 정확한 도구 선택, 다중 턴 가능 | 모든 모델에서 동작 |
| **단점** | FC 지원 모델 필요 | 단일 턴만 가능, 정확도 낮을 수 있음 |

### 7.2 Native Mode — 멀티턴 에이전트 루프

```
Turn 1: 사용자 → LLM → tool_calls: [search_web("날씨")]
         → search_web 실행 → 결과: "서울 25도"

Turn 2: [원본 메시지 + 도구 결과] → LLM → tool_calls: [fetch_url("weather.com")]
         → fetch_url 실행 → 결과: "상세 날씨 정보..."

Turn 3: [원본 메시지 + 도구 결과1 + 도구 결과2] → LLM → 최종 응답
         "서울의 현재 날씨는 25도이며..."
```

이 루프가 최대 30회까지 반복될 수 있어, LLM이 **자율적으로 여러 도구를 순차적으로 호출**하며 문제를 해결할 수 있습니다.

### 7.3 Default Mode — 단일턴 도구 호출

```
Step 1: Task Model에게 "다음 도구 중 어떤 것을 호출할지 JSON으로 답하라"
        → {"tool_calls": [{"name": "search_web", "parameters": {"query": "날씨"}}]}

Step 2: search_web("날씨") 실행 → 결과 수집

Step 3: Main Model에게 [원본 메시지 + 도구 결과] 전달 → 최종 응답 생성
```

---

## 8. 스트리밍 처리에서의 도구 호출 감지

### 8.1 SSE 스트림에서 tool_calls 조립

**파일**: `middleware.py` L3679-3741

LLM의 스트리밍 응답에서 `tool_calls`는 **델타 청크**로 분할되어 옵니다:

```
chunk 1: {"delta": {"tool_calls": [{"index": 0, "function": {"name": "search"}}]}}
chunk 2: {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"q"}}]}}
chunk 3: {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "uery\":"}}]}}
chunk 4: {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"test\"}"}}]}}
```

Open WebUI는 이를 실시간으로 **조립(accumulate)**하여 완전한 tool_call 객체를 구성합니다.

### 8.2 복수 JSON 분할 처리

**파일**: `middleware.py` L172-218 (`_split_tool_calls`)

일부 모델은 하나의 tool_call에 **여러 JSON 객체**를 연결해서 반환합니다:

```json
{"arguments": "{\"query\":\"A\"}{\"query\":\"B\"}"}
```

이를 감지하여 개별 tool_call로 분리합니다.

---

## 9. MCP (Model Context Protocol) 통합

### 9.1 MCP 클라이언트 구조

**파일**: `backend/open_webui/utils/mcp/client.py`

```python
class MCPClient:
    async def connect()             # MCP 서버 연결
    async def list_tool_specs()     # 사용 가능한 도구 스펙 조회
    async def call_tool(name, args) # 도구 실행
    async def list_resources()      # 리소스 목록 조회
    async def read_resource(uri)    # 리소스 읽기
```

### 9.2 MCP 도구가 에이전트 루프에 통합되는 방식

```
MCP Server (외부)                 Open WebUI
┌──────────────┐                ┌──────────────────────┐
│ Tool: weather │◄──────────────│ MCPClient.call_tool() │
│ Tool: calc    │               │                      │
└──────────────┘                │ ↑ Agent Loop에서      │
                                │   도구 실행 시 호출    │
                                └──────────────────────┘
```

MCP 도구도 로컬 도구와 동일하게 Agent Loop 내에서 실행됩니다. 차이점은 실행 시 `MCPClient.call_tool()`을 통해 외부 서버와 통신한다는 것뿐입니다.

---

## 10. 코드 인터프리터 통합

**파일**: `middleware.py` L4434-4559

Agent Loop의 일부로 **코드 인터프리터**도 통합되어 있습니다:

```python
# 코드 인터프리터 감지
if DETECT_CODE_INTERPRETER:
    # LLM 응답에서 코드 블록 감지
    # Pyodide 또는 Jupyter로 실행
    # 실행 결과를 다시 LLM에 전달
```

지원 엔진:
- **Pyodide**: 브라우저 내 Python 실행
- **Jupyter**: Jupyter 커널 기반 실행

---

## 11. 이벤트 시스템 (실시간 UI 피드백)

도구 실행 중 **실시간 이벤트**를 프론트엔드로 전송합니다:

```python
# 도구 실행 상태 이벤트
await event_emitter({
    "type": "status",
    "data": {"description": "Searching the web...", "done": False}
})

# 도구 실행 완료 이벤트
await event_emitter({
    "type": "status",
    "data": {"description": "Search completed", "done": True}
})

# 파일 첨부 이벤트
await event_emitter({
    "type": "files",
    "data": {"files": [{"url": "...", "name": "result.png"}]}
})

# 인용 소스 이벤트 (search_web, fetch_url 등)
await event_emitter({
    "type": "citation",
    "data": {"source": {"name": "...", "url": "..."}}
})
```

---

## 12. 전체 요청 흐름도 (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        사용자 메시지 전송                             │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST /api/chat/completions                     (main.py:1635)     │
│  ├─ 모델 접근 권한 확인                                              │
│  ├─ 메타데이터 추출 (chat_id, tool_ids, session_id)                  │
│  └─ process_chat() 호출                                             │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  process_chat_payload()                    (middleware.py:2109)     │
│  ├─ 시스템 프롬프트 처리                                              │
│  ├─ 기능 활성화 (메모리, 웹검색, 이미지생성, 코드실행)                   │
│  ├─ 도구 로딩:                                                      │
│  │   ├─ 내장 도구 (builtin.py)                                      │
│  │   ├─ 사용자 정의 도구 (DB)                                        │
│  │   ├─ MCP 서버 도구 (mcp/client.py)                               │
│  │   └─ OpenAPI 서버 도구 (tools.py)                                │
│  ├─ 파이프라인 inlet 필터 실행                                        │
│  └─ 모드 결정: Native vs Default                                    │
│      ├─ Native → tools를 LLM 요청에 포함                            │
│      └─ Default → chat_completion_tools_handler() 호출              │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  generate_chat_completion()                      (chat.py:158)     │
│  ├─ Ollama 모델 → generate_ollama_chat_completion()                │
│  ├─ OpenAI 모델 → generate_openai_chat_completion()                │
│  └─ Pipe 모델 → generate_function_chat_completion()                │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM Provider (OpenAI / Anthropic / Ollama / etc.)                 │
│  → SSE 스트리밍 응답 반환                                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  streaming_chat_response_handler()         (middleware.py:3179)     │
│  ├─ SSE 청크 파싱                                                   │
│  ├─ 텍스트 콘텐츠 → 프론트엔드로 스트리밍                              │
│  └─ tool_calls 델타 → 조립 및 축적                                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          ▼
┌═════════════════════════════════════════════════════════════════════┐
║  AGENT LOOP                                (middleware.py:4065)    ║
║                                                                    ║
║  while tool_calls > 0 AND retries < 30:                           ║
║    │                                                               ║
║    ├─ 1. tool_calls 배치 추출                                      ║
║    │                                                               ║
║    ├─ 2. 각 도구 실행                                              ║
║    │   ├─ 로컬: await tool_function(**params)                      ║
║    │   ├─ MCP:  await mcp_client.call_tool(name, params)          ║
║    │   └─ 서버: await event_caller({type: 'execute:tool', ...})   ║
║    │                                                               ║
║    ├─ 3. 결과를 function_call_output으로 변환                       ║
║    │                                                               ║
║    ├─ 4. [원본 메시지 + 도구 결과] 로 LLM 재호출                     ║
║    │                                                               ║
║    └─ 5. 새 tool_calls 감지 → 있으면 계속, 없으면 break             ║
║                                                                    ║
╚═════════════════════════════════════════════════════════════════════╝
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  최종 응답                                                          │
│  ├─ 텍스트 응답 스트리밍                                              │
│  ├─ 인용(citations) 첨부                                            │
│  ├─ DB에 메시지 저장                                                 │
│  └─ 파이프라인 outlet 필터 실행                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. 주요 파일 참조 맵

| 파일 | 핵심 역할 | 주요 라인 |
|------|----------|----------|
| `main.py` | `/api/chat/completions` 엔드포인트 | L1635-1873 |
| `utils/middleware.py` | **Agent Loop 핵심**, 페이로드 처리 | L4065-4432 (루프), L2109-2747 (페이로드) |
| `utils/middleware.py` | Default 모드 도구 핸들러 | L1181-1390 |
| `utils/chat.py` | LLM 백엔드 라우팅 | L158-300 |
| `utils/tools.py` | 도구 로딩, 래핑, 스펙 변환 | L97-242, L1191-1337 |
| `utils/mcp/client.py` | MCP 프로토콜 클라이언트 | L35-137 |
| `tools/builtin.py` | 내장 도구 구현 (30+ 함수) | 전체 (2,274줄) |
| `utils/anthropic.py` | Anthropic ↔ OpenAI 변환 | L93-526 |
| `utils/misc.py` | output → messages 변환 | L132-276 |
| `env.py` | `MAX_TOOL_CALL_RETRIES=30` | L622-630 |
| `models/tools.py` | 도구 DB 모델 | L23-292 |
| `routers/tools.py` | 도구 CRUD API | 전체 |
| `functions.py` | Pipe/Filter 함수 실행 | L57-341 |

---

## 14. 설계 철학: 왜 LangGraph/ReAct를 쓰지 않는가?

### 14.1 Open WebUI의 접근법

Open WebUI는 의도적으로 **최소주의적 에이전트 설계**를 채택합니다:

1. **LLM의 내장 추론에 의존**: GPT-4, Claude 등 최신 LLM은 이미 tool_calls를 통해 어떤 도구를 호출할지 스스로 결정할 수 있습니다. 별도의 ReAct Thought 단계나 Plan 생성 단계를 코드로 강제할 필요가 없습니다.

2. **프레임워크 의존성 최소화**: LangGraph나 LangChain Agent에 의존하면 버전 호환성, API 변경, 추가 추상화 오버헤드가 발생합니다. Open WebUI는 OpenAI Function Calling 프로토콜이라는 **표준 인터페이스**만 사용합니다.

3. **범용성 우선**: 다양한 LLM 백엔드(OpenAI, Anthropic, Ollama, Azure 등)를 지원해야 하므로, 특정 프레임워크에 종속되지 않는 단순한 루프가 더 적합합니다.

4. **투명성**: 코드 흐름이 단순하여 디버깅과 커스터마이징이 쉽습니다. LangGraph의 복잡한 상태 머신 대신, 누구나 이해할 수 있는 while 루프입니다.

### 14.2 비교 요약

```
LangGraph 방식:
  StateGraph → Node → Edge → 조건부 분기 → 서브그래프 → 컴파일 → 실행
  (높은 추상화, 복잡한 워크플로에 적합)

Open WebUI 방식:
  while tool_calls and retries < 30:
      execute_tools()
      call_llm_again()
  (낮은 추상화, 단순하지만 효과적)
```

### 14.3 한계점

- **복잡한 멀티 에이전트 협업** 불가 (예: Supervisor → Worker 패턴)
- **조건부 분기** 없음 (도구 결과에 따른 다른 경로 선택)
- **사전 계획(Planning)** 없음 (LLM의 암묵적 계획에 의존)
- **자기 반성(Self-Reflection)** 없음 (실행 결과 평가 단계 없음)
- **메모리 기반 장기 계획** 없음 (세션 내 도구 결과만 축적)

이러한 한계는 Open WebUI가 **범용 AI 채팅 인터페이스**이지 **자율 에이전트 프레임워크**가 아니기 때문에 의도된 트레이드오프입니다.

---

## 15. 결론

Open WebUI의 에이전틱 AI 구현은 **"Simple Agent Loop"** 패턴입니다:

```
LLM 호출 → tool_calls 감지 → 도구 실행 → 결과 주입 → LLM 재호출 (반복)
```

- LangGraph, ReAct, Plan-and-Solve 등 **학술적 에이전트 패턴은 사용하지 않음**
- LangChain은 **RAG 유틸리티(문서 로딩, 텍스트 분할, 검색)**에만 사용
- **OpenAI Function Calling 프로토콜**을 표준으로 채택
- 최대 **30회 반복**으로 자율적 멀티턴 도구 사용 가능
- **4가지 도구 소스** 지원: 내장, 사용자 정의, MCP, OpenAPI 서버
- Native/Default 두 모드로 Function Calling 지원/미지원 모델 모두 대응

이는 "최소한의 코드로 최대한의 범용성"을 추구하는 실용적 설계입니다.
