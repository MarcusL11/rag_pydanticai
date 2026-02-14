# Drug Info RAG with PydanticAI

A Retrieval-Augmented Generation (RAG) application that answers questions about pharmaceutical drugs using scraped medical data. Built with PydanticAI, LangChain, and PostgreSQL/pgvector.

## Features

- **Web Scraping**: Crawls and extracts drug information using Firecrawl from Mayo Clinic
- **Vector Storage**: Stores embeddings in PostgreSQL with pgvector extension
- **Intelligent Retrieval**: PydanticAI agent with automated similarity search tool
- **Web Chat Interface**: Interactive browser-based chat UI via `agent.to_web()`
- **Multiple Vector Store Options**: Supports both PostgreSQL/pgvector and Pinecone

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Firecrawl    │────▶│   Chunking   │────▶│  PostgreSQL  │
│  Scraper     │     │   (Recursive)│     │   + pgvector │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web Chat   │────▶│ PydanticAI   │────▶│   Similarity │
│     UI       │     │    Agent     │     │    Search    │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Screenshots

### Web Chat Interface

![Web Chat Interface - Main View](image_1.png)
*The PydanticAI web chat interface showing real-time streaming responses and conversation history*

![Web Chat Interface - Tool Calls](image_2.png)
*Tool call visualization showing when the agent retrieves information from the vector store*

## Prerequisites

- Python 3.12+
- PostgreSQL 14+ with pgvector extension (or Docker)
- OpenAI API key
- Firecrawl API key (optional, for re-scraping)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drug-info-rag.git
cd drug-info-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here  # Optional
```

## Database Setup

### Option 1: Docker (Recommended)

```bash
# Run PostgreSQL with pgvector
docker run -d \
  -e POSTGRES_PASSWORD=pass \
  -p 5432:5432 \
  -v postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

### Option 2: Local PostgreSQL

```bash
# Install pgvector extension
psql -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Usage

### Option 1: Command Line Interface

```bash
python pydantic_example.py
```

You'll be prompted to enter a question:

```
Hi, I am your friendly Assistant -- Ask a drug-related question: What are the side effects of sertraline?
```

### Option 2: Web Chat Interface

```bash
python pydantic_web.py
```

Then open your browser to: **<http://localhost:8000>**

Features:

- Real-time streaming responses
- Tool call visualization (see when retrieval happens)
- Conversation history
- Multi-turn dialogue support

## Project Structure

```
day_1_rag/
├── pydantic_example.py      # CLI version with PGVector
├── pydantic_web.py          # Web UI version with PydanticAI
├── scrape_data.py           # Firecrawl scraper (optional re-scraping)
├── blog.md                  # Detailed implementation notes
├── data/
│   └── raw_documents/       # Scraped JSON files (auto-generated)
└── README.md               # This file
```

## How It Works

### 1. Data Scraping (Optional)

If you want to re-scrape data:

```python
from firecrawl import Firecrawl

app = Firecrawl()
crawl_result = app.crawl(
    "https://www.mayoclinic.org/drugs-supplements",
    limit=10,
    scrape_options={"formats": ["markdown"]},
)
```

### 2. Vector Store Population

The application automatically checks if the vector store is populated. If not, it:

1. Loads JSON files from `data/raw_documents/`
2. Chunks text using RecursiveCharacterTextSplitter (500 chars, 50 overlap)
3. Embeds chunks using OpenAI's text-embedding-3-small (1536 dimensions)
4. Stores in PostgreSQL with pgvector

### 3. PydanticAI Agent Setup

**Dependencies (`RAGDeps`):**

```python
@dataclass
class RAGDeps:
    vector_store: PGVectorStore
    embeddings: OpenAIEmbeddings
```

**Response Structure (`RAGResponse`):**

```python
class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

> [!TIP]
> `RAGDeps` uses `@dataclass` (simple dependency container) while `RAGResponse` extends `BaseModel` (enables Pydantic validation and JSON schema generation for the LLM's structured output).

**Retrieval Tool:**

```python
@rag_agent.tool
async def retrieve_drug_info(ctx: RunContext[RAGDeps], query: str, k: int = 3) -> str:
    """Retrieve drug information from the vector store"""
    results = ctx.deps.vector_store.similarity_search(query, k=k)
    # Formats and returns context for the LLM
```

### 4. Agent Configuration

```python
rag_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=RAGDeps,
    system_prompt="""
    You are a helpful medical assistant specializing in drug information.
    Use the retrieve tool to find information when users ask questions.
    Always cite your sources and be accurate.
    """,
)
```

### 5. Web Interface

Convert agent to web UI:

```python
deps = RAGDeps(vector_store=vector_store, embeddings=embeddings)
app = rag_agent.to_web(deps=deps)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Key Implementation Details

### Chunking Strategy

Uses `RecursiveCharacterTextSplitter` with this priority order:

1. Paragraph breaks (`\n\n`)
2. Line breaks (`\n`)
3. Sentence ends (`.`)
4. Spaces
5. Any character

This preserves semantic coherence while staying within 500-character chunks.

### Usage Limits

The web UI version removes PydanticAI's default request limits:

```python
usage_limits = UsageLimits(request_limit=None, response_tokens_limit=50000)
```

### Vector Store Options

The code supports both:

- **PostgreSQL + pgvector**: Self-hosted, SQL integration, no vendor lock-in
- **Pinecone**: Managed service, automatic scaling (commented in code)

## Technologies Used

| Component | Technology |
|-----------|------------|
| **Agent Framework** | PydanticAI |
| **Vector Store** | PostgreSQL + pgvector (or Pinecone) |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Web Scraping** | Firecrawl |
| **Web Framework** | Starlette (via PydanticAI) + Uvicorn |
| **Data Validation** | Pydantic |

## Learning Resources

This project demonstrates:

- [Best Chunking Strategies for RAG in 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [How to build a production agentic app, the Pydantic Way](https://pydantic.dev/articles/building-agentic-application)

See `blog.md` for cimplementation notes.

## License

MIT License - feel free to use this for your own projects!
