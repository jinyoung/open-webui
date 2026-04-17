# Open WebUI 벡터 데이터베이스 아키텍처 분석 보고서

## 1. 개요

Open WebUI는 RAG(Retrieval Augmented Generation) 기능을 위해 **12종의 벡터 데이터베이스**를 지원합니다. 이는 다양한 배포 환경(개인 로컬 → 중소규모 → 엔터프라이즈)에서 사용자가 이미 운영 중인 인프라에 맞춰 유연하게 선택할 수 있도록 설계된 것입니다.

**기본 벡터 DB**: ChromaDB (로컬 파일 기반, 별도 서버 불필요)

---

## 2. 지원하는 벡터 데이터베이스 목록

| # | 벡터 DB | 타입 | 주요 특징 | 적합한 환경 |
|---|---------|------|-----------|-------------|
| 1 | **ChromaDB** | 임베디드/HTTP | 기본값, 로컬 파일 저장, 별도 설치 불필요 | 개인/개발 환경 |
| 2 | **Milvus** | 독립형 서버 | HNSW/IVF_FLAT/DiskANN 인덱스, 멀티테넌시 지원 | 중대규모 운영 |
| 3 | **Qdrant** | 독립형 서버 | gRPC/HTTP, 멀티테넌시 기본 활성화 | 중대규모 운영 |
| 4 | **Pinecone** | 클라우드 관리형 | AWS/GCP/Azure 서버리스, gRPC 지원 | 클라우드 네이티브 |
| 5 | **Elasticsearch** | 검색 엔진 | Dense Vector, Cloud ID 지원 | 기존 ELK 스택 보유 시 |
| 6 | **OpenSearch** | 검색 엔진 | KNN + FAISS 엔진, Elasticsearch 대안 | AWS 환경 |
| 7 | **PGVector** | RDBMS 확장 | PostgreSQL 확장, HNSW/IVFFlat, pgcrypto 암호화 | 기존 PostgreSQL 운영 시 |
| 8 | **Weaviate** | 독립형 서버 | HTTP/gRPC 이중 프로토콜 | 중대규모 운영 |
| 9 | **MariaDB Vector** | RDBMS 확장 | MariaDB 네이티브 벡터, HNSW 인덱스 | 기존 MariaDB 운영 시 |
| 10 | **OpenGauss** | RDBMS 확장 | PostgreSQL 호환, 화웨이 오픈소스 DB | 중국 시장/화웨이 인프라 |
| 11 | **Oracle 23ai** | RDBMS 확장 | Wallet 인증, 커넥션 풀링 | Oracle 엔터프라이즈 환경 |
| 12 | **S3Vector** | 클라우드 스토리지 | AWS S3 기반 벡터 저장 | AWS 경량 환경 |

---

## 3. 왜 이렇게 많은 벡터 DB를 지원하는가?

### 3.1 다양한 배포 시나리오 대응

```
┌─────────────────────────────────────────────────────────────────┐
│                    배포 시나리오별 선택 가이드                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [개인/개발]  ──→  ChromaDB (기본값, 제로 설정)                    │
│                    Milvus Lite (로컬 파일 모드)                    │
│                                                                 │
│  [기존 DB 활용] ──→ PGVector (PostgreSQL 이미 사용 중)            │
│                    MariaDB Vector (MariaDB 이미 사용 중)          │
│                    Oracle 23ai (Oracle 이미 사용 중)              │
│                    OpenGauss (OpenGauss 이미 사용 중)             │
│                                                                 │
│  [검색 엔진 활용] ──→ Elasticsearch (ELK 스택 보유)               │
│                      OpenSearch (AWS 관리형)                     │
│                                                                 │
│  [전용 벡터 DB] ──→ Milvus (고성능, 다양한 인덱스)                 │
│                    Qdrant (gRPC, 멀티테넌시)                      │
│                    Weaviate (스키마 기반)                         │
│                                                                 │
│  [클라우드 관리형] ──→ Pinecone (서버리스)                         │
│                      S3Vector (AWS S3)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 핵심 이유

1. **기존 인프라 재사용**: PostgreSQL, MariaDB, Oracle, Elasticsearch를 이미 운영 중인 조직은 새 인프라 없이 벡터 검색 가능
2. **확장성 단계**: 로컬 ChromaDB에서 시작 → 운영 규모 성장 시 Milvus/Qdrant로 마이그레이션
3. **클라우드 벤더 종속 회피**: AWS(OpenSearch, S3Vector), GCP/Azure(Pinecone) 등 다양한 클라우드 지원
4. **엔터프라이즈 요구사항**: Oracle Wallet 인증, pgcrypto 암호화, 멀티테넌시 등 기업 환경 필수 기능
5. **커뮤니티 기여**: 오픈소스 프로젝트 특성상 각 DB 사용자 커뮤니티가 자체 구현을 기여

---

## 4. 아키텍처 설계

### 4.1 전체 구조

```
backend/open_webui/retrieval/vector/
├── main.py              # VectorDBBase 추상 클래스 (인터페이스 정의)
├── type.py              # VectorType 열거형 (12종 DB 타입)
├── factory.py           # Factory 패턴으로 DB 인스턴스 생성
├── utils.py             # 메타데이터 처리 유틸리티
└── dbs/                 # 각 벡터 DB 구현체
    ├── chroma.py
    ├── milvus.py
    ├── milvus_multitenancy.py
    ├── qdrant.py
    ├── qdrant_multitenancy.py
    ├── pinecone.py
    ├── elasticsearch.py
    ├── opensearch.py
    ├── pgvector.py
    ├── weaviate.py
    ├── mariadb_vector.py
    ├── opengauss.py
    ├── oracle23ai.py
    └── s3vector.py
```

### 4.2 추상화 계층 (Strategy Pattern)

```python
# main.py - 공통 인터페이스
class VectorDBBase(ABC):
    @abstractmethod
    def has_collection(collection_name: str) -> bool
    def delete_collection(collection_name: str) -> None
    def insert(collection_name: str, items: List[VectorItem]) -> None
    def upsert(collection_name: str, items: List[VectorItem]) -> None
    def search(collection_name: str, vectors: List, limit: int) -> SearchResult
    def query(collection_name: str, filter: Dict) -> GetResult
    def get(collection_name: str) -> GetResult
    def delete(collection_name: str, ids: List[str]) -> None
    def reset() -> None
```

```python
# factory.py - 환경변수 기반 DB 선택
VECTOR_DB_CLIENT = Vector.get_vector(VECTOR_DB)  # 환경변수: VECTOR_DB=chroma
```

모든 벡터 DB 구현체가 동일한 인터페이스(`VectorDBBase`)를 구현하므로, 환경변수 하나(`VECTOR_DB`)만 변경하면 애플리케이션 코드 수정 없이 벡터 DB를 교체할 수 있습니다.

### 4.3 데이터 모델

```python
# VectorItem - 벡터 DB에 저장되는 단위
{
    "id": "uuid",                    # 청크 고유 ID
    "text": "문서 텍스트 청크",        # 원본 텍스트
    "vector": [0.1, 0.2, ...],       # 임베딩 벡터
    "metadata": {                     # 메타데이터
        "file_id": "...",
        "name": "filename.pdf",
        "source": "filename.pdf",
        "created_by": "user_id",
        "hash": "document_hash",
        "embedding_config": {
            "engine": "openai",
            "model": "text-embedding-3-small"
        }
    }
}
```

---

## 5. RAG 파이프라인 전체 흐름

### 5.1 문서 수집 (Ingestion) 파이프라인

```
문서 업로드 (PDF, DOCX, TXT, ...)
       │
       ▼
┌──────────────────────┐
│  콘텐츠 추출 (Loader) │  ← Datalab, MinerU, Mistral OCR, YouTube 등
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  텍스트 분할 (Chunking)│  ← RecursiveCharacterTextSplitter (기본)
│                      │    또는 MarkdownHeaderTextSplitter
│  CHUNK_SIZE=1000     │
│  CHUNK_OVERLAP=20    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  임베딩 생성          │  ← Local(sentence-transformers) / Ollama
│                      │    / OpenAI / Azure OpenAI
│  기본 모델:           │
│  all-MiniLM-L6-v2   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  벡터 DB 저장         │  ← ChromaDB / Milvus / Qdrant / ...
│  (insert/upsert)     │    환경변수 VECTOR_DB로 선택
└──────────────────────┘
```

### 5.2 검색 (Retrieval) 파이프라인

```
사용자 질의 (Query)
       │
       ▼
┌──────────────────────┐
│  질의 임베딩 생성      │  ← 수집 시와 동일한 모델 사용
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  벡터 유사도 검색 (Vector Search)          │
│  ┌────────────────┐  ┌────────────────┐  │
│  │ 벡터 검색 (ANN) │  │ BM25 키워드 검색│  │  ← ENABLE_RAG_HYBRID_SEARCH=true
│  │  TOP_K=10      │  │ (선택적)       │  │
│  └───────┬────────┘  └───────┬────────┘  │
│          │                   │            │
│          └───────┬───────────┘            │
│                  │                        │
│          ┌───────▼────────┐               │
│          │ Reciprocal Rank│               │
│          │ Fusion (RRF)   │               │
│          └───────┬────────┘               │
└──────────────────┼───────────────────────┘
                   │
                   ▼
┌──────────────────────┐
│  리랭킹 (선택적)       │  ← CrossEncoder / ColBERT / 외부 API
│  TOP_K_RERANKER=10   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  컨텍스트 조립         │  ← RAG 템플릿에 검색 결과 주입
│  + RAG 프롬프트 생성   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM 응답 생성        │  ← 컨텍스트 포함된 프롬프트로 생성
│  (with citations)    │
└──────────────────────┘
```

---

## 6. 벡터 DB별 상세 설정

### 6.1 환경변수 설정 방법

벡터 DB 선택은 단일 환경변수로 수행됩니다:

```bash
VECTOR_DB=chroma   # 기본값
```

### 6.2 주요 벡터 DB 설정 요약

#### ChromaDB (기본값)
```bash
VECTOR_DB=chroma
CHROMA_DATA_PATH=/app/backend/data/vector_db    # 로컬 저장 경로
# 원격 서버 사용 시:
CHROMA_HTTP_HOST=chroma-server
CHROMA_HTTP_PORT=8000
CHROMA_HTTP_SSL=false
CHROMA_CLIENT_AUTH_PROVIDER=...
CHROMA_CLIENT_AUTH_CREDENTIALS=...
```

#### Milvus
```bash
VECTOR_DB=milvus
MILVUS_URI=/app/backend/data/vector_db/milvus.db   # Lite 모드 (로컬)
# 또는 원격 서버:
MILVUS_URI=http://milvus-server:19530
MILVUS_TOKEN=...
MILVUS_INDEX_TYPE=HNSW          # HNSW, IVF_FLAT, DISKANN, FLAT, AUTOINDEX
MILVUS_METRIC_TYPE=COSINE
ENABLE_MILVUS_MULTITENANCY_MODE=false
```

#### Qdrant
```bash
VECTOR_DB=qdrant
QDRANT_URI=http://qdrant-server:6333
QDRANT_API_KEY=...
QDRANT_PREFER_GRPC=false
ENABLE_QDRANT_MULTITENANCY_MODE=true   # 기본값: 활성화
```

#### PGVector (PostgreSQL)
```bash
VECTOR_DB=pgvector
PGVECTOR_DB_URL=postgresql://user:pass@host:5432/dbname
PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH=1536
PGVECTOR_USE_HALFVEC=false       # 2000차원 초과 시 true
PGVECTOR_INDEX_METHOD=hnsw       # hnsw 또는 ivfflat
PGVECTOR_PGCRYPTO=false          # 벡터 데이터 암호화
```

#### Pinecone (클라우드)
```bash
VECTOR_DB=pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
PINECONE_INDEX_NAME=open-webui-index
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws               # aws, gcp, azure
```

#### Elasticsearch
```bash
VECTOR_DB=elasticsearch
ELASTICSEARCH_URL=https://localhost:9200
ELASTICSEARCH_API_KEY=...
# 또는 Elastic Cloud:
ELASTICSEARCH_CLOUD_ID=...
ELASTICSEARCH_INDEX_PREFIX=open_webui_collections
```

#### OpenSearch
```bash
VECTOR_DB=opensearch
OPENSEARCH_URI=https://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=...
OPENSEARCH_SSL=true
```

#### Weaviate
```bash
VECTOR_DB=weaviate
WEAVIATE_HTTP_HOST=weaviate-server
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_HOST=weaviate-server
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=...
```

#### MariaDB Vector
```bash
VECTOR_DB=mariadb-vector
MARIADB_VECTOR_DB_URL=mariadb+mariadbconnector://user:pass@host:3306/db
MARIADB_VECTOR_DISTANCE_STRATEGY=cosine
MARIADB_VECTOR_INDEX_M=8
```

#### Oracle 23ai
```bash
VECTOR_DB=oracle23ai
ORACLE_DB_USER=...
ORACLE_DB_PASSWORD=...
ORACLE_DB_DSN=...
ORACLE_DB_USE_WALLET=false
ORACLE_VECTOR_LENGTH=768
```

---

## 7. 멀티테넌시 지원

Milvus와 Qdrant는 별도의 멀티테넌시 구현을 제공합니다.

### 컬렉션 매핑 전략 (멀티테넌시 모드)

```
컬렉션 이름 패턴              →  공유 컬렉션
─────────────────────────────────────────────
user-memory-*                →  MEMORY_COLLECTION
file-*                       →  FILE_COLLECTION
web-search-*                 →  WEB_SEARCH_COLLECTION
63자 hex hash                →  HASH_BASED_COLLECTION
기타                          →  KNOWLEDGE_COLLECTION
```

각 문서에 `resource_id` (Milvus) 또는 `tenant_id` (Qdrant)를 태깅하여 테넌트별 격리를 보장합니다.

---

## 8. 임베딩 엔진 구성

벡터 DB와 함께 사용되는 임베딩 엔진도 교체 가능합니다:

| 엔진 | 환경변수 | 기본 모델 | 비고 |
|------|----------|-----------|------|
| **로컬** (기본) | `RAG_EMBEDDING_ENGINE=''` | `all-MiniLM-L6-v2` | 서버 내 직접 실행 |
| **Ollama** | `RAG_EMBEDDING_ENGINE=ollama` | 사용자 지정 | Ollama 서버 필요 |
| **OpenAI** | `RAG_EMBEDDING_ENGINE=openai` | 사용자 지정 | API 키 필요 |
| **Azure OpenAI** | `RAG_EMBEDDING_ENGINE=azure_openai` | 사용자 지정 | Azure 엔드포인트 필요 |

### 주요 임베딩 설정

```bash
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_BATCH_SIZE=1           # 배치 크기
ENABLE_ASYNC_EMBEDDING=true          # 비동기 임베딩
RAG_EMBEDDING_CONCURRENT_REQUESTS=0  # 동시 요청 수 (0=무제한)
RAG_EMBEDDING_QUERY_PREFIX=          # 질의 시 접두사
RAG_EMBEDDING_CONTENT_PREFIX=        # 문서 저장 시 접두사
```

---

## 9. 검색 고급 기능

### 9.1 하이브리드 검색 (Hybrid Search)

```bash
ENABLE_RAG_HYBRID_SEARCH=false       # 활성화 시 true
HYBRID_BM25_WEIGHT=0.3               # BM25 가중치 (0~1)
```

벡터 유사도 검색 + BM25 키워드 검색을 **Reciprocal Rank Fusion (RRF)**으로 결합합니다.

### 9.2 리랭킹 (Reranking)

```bash
RAG_RERANKING_MODEL=                 # 리랭커 모델 (비워두면 비활성)
TOP_K_RERANKER=10                    # 리랭킹 대상 결과 수
RELEVANCE_THRESHOLD=0.0              # 최소 관련성 점수
```

CrossEncoder, ColBERT, 또는 외부 API 기반 리랭킹을 지원합니다.

---

## 10. 인덱스 알고리즘 비교

| 벡터 DB | 지원 인덱스 | 기본값 | 특징 |
|---------|------------|--------|------|
| Milvus | HNSW, IVF_FLAT, DiskANN, FLAT, AUTOINDEX | HNSW | 가장 다양한 옵션 |
| Qdrant | HNSW | HNSW | M 파라미터 조정 가능 |
| PGVector | HNSW, IVFFlat | 없음 | 인덱스 없이도 동작 |
| OpenSearch | HNSW (FAISS) | HNSW | ef_construction=128, m=16 |
| Elasticsearch | Dense Vector | 자동 | Lucene 기반 |
| MariaDB | HNSW | HNSW | M=8 기본 |
| Weaviate | HNSW | 자동 | 내부 관리 |
| Pinecone | 관리형 | 자동 | 사용자 설정 불필요 |

---

## 11. 거리/유사도 정규화

각 벡터 DB가 반환하는 거리 범위가 다르므로, Open WebUI는 이를 `[0, 1]` 범위로 정규화합니다:

| 벡터 DB | 원본 거리 범위 | 정규화 방식 |
|---------|---------------|------------|
| ChromaDB | [2, 0] (코사인) | `(2 - distance) / 2` |
| Milvus | [-1, 1] | `(1 + distance) / 2` |
| Qdrant | [-1, 1] | `(1 + distance) / 2` |
| Weaviate | [0, 2] | `1 - (distance / 2)` |
| Pinecone | 메트릭별 상이 | 메트릭별 개별 처리 |

---

## 12. 주요 파일 참조

| 파일 경로 | 역할 |
|-----------|------|
| `backend/open_webui/retrieval/vector/main.py` | VectorDBBase 추상 클래스 |
| `backend/open_webui/retrieval/vector/type.py` | VectorType 열거형 |
| `backend/open_webui/retrieval/vector/factory.py` | Factory 패턴 - DB 인스턴스 생성 |
| `backend/open_webui/retrieval/vector/dbs/*.py` | 각 벡터 DB 구현체 (14개 파일) |
| `backend/open_webui/retrieval/utils.py` | 검색/임베딩 유틸리티 |
| `backend/open_webui/utils/embeddings.py` | 임베딩 엔진 디스패치 |
| `backend/open_webui/routers/retrieval.py` | RAG API 엔드포인트 |
| `backend/open_webui/routers/knowledge.py` | Knowledge Base 관리 |
| `backend/open_webui/config.py` (L2161-2480) | 모든 벡터 DB 환경변수 정의 |
| `backend/open_webui/utils/task.py` | RAG 템플릿 처리 |

---

## 13. 결론

Open WebUI가 12종의 벡터 DB를 지원하는 이유는 **"어떤 환경에서든 벡터 검색을 사용할 수 있게"** 하기 위함입니다:

- **진입 장벽 최소화**: ChromaDB 기본값으로 설치 즉시 RAG 사용 가능
- **기존 인프라 활용**: PostgreSQL, MariaDB, Oracle, Elasticsearch 등 이미 운영 중인 DB에 벡터 기능 추가
- **확장 경로 제공**: 로컬 ChromaDB → 전용 벡터 DB(Milvus, Qdrant) → 클라우드 관리형(Pinecone)으로 점진적 확장
- **벤더 락인 방지**: Strategy Pattern으로 환경변수 하나만 변경하면 DB 교체 가능

이 설계는 Open WebUI의 "누구나 쉽게 사용할 수 있는 AI 인터페이스"라는 철학을 벡터 DB 계층에서도 그대로 구현한 것입니다.
