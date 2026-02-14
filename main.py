import json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic_ai import Agent, RunContext

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path("data/raw_documents")
BASE_URL = "https://www.mayoclinic.org/drugs-supplements"
CRAWL_LIMIT = 10


# ============================================================================
# SCRAPING FUNCTION
# ============================================================================
def scrape_and_save_data():
    """Scrape drug information from Mayo Clinic and save to disk"""
    app = Firecrawl()

    print(f"Crawling {BASE_URL}...")
    crawl_result = app.crawl(
        BASE_URL,
        limit=CRAWL_LIMIT,
        scrape_options={"formats": ["markdown"]},
    )

    # Extract URLs from crawl results
    label_urls = []
    if crawl_result.data:
        for page in crawl_result.data:
            if hasattr(page, "metadata") and page.metadata:
                url = getattr(page.metadata, "source_url", None) or getattr(
                    page.metadata, "url", None
                )
                if url and url != BASE_URL:
                    label_urls.append(url)

    print(f"Discovered {len(label_urls)} drug information pages")

    if not label_urls:
        print("No URLs found to scrape")
        return

    # Batch scrape all discovered pages
    print("Batch scraping pages...")
    batch_job = app.batch_scrape(label_urls, formats=["markdown"])

    # Process and save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(batch_job.data):
        doc = {
            "url": result.metadata.url if result.metadata else "",
            "markdown": result.markdown,
            "title": result.metadata.title if result.metadata else "",
            "description": result.metadata.description if result.metadata else "",
        }

        filepath = DATA_DIR / f"drug_info_{i:02d}.json"
        with open(filepath, "w") as f:
            json.dump(doc, f, indent=2)

        print(f"Saved: {doc['title'] or 'Unknown'} -> {filepath}")

    print(f"\nSaved {len(batch_job.data)} documents to {DATA_DIR}")


# ============================================================================
# VECTOR STORE SETUP
# ============================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Setup PGVector
CONNECTION_STRING = "postgresql+psycopg://mal:pass@localhost:5432/mal"
engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name="drug_info",
    embedding_service=embeddings,
)
# Check and populate if empty
test_results = vector_store.similarity_search("test", k=1)
if len(test_results) == 0:
    print("Vector store is empty. Populating...")

    json_files = list(DATA_DIR.glob("drug_info_*.json"))
    if not json_files:
        print("No JSON files found. Scraping data first...")
        scrape_and_save_data()

    documents = []
    for json_file in DATA_DIR.glob("drug_info_*.json"):
        with open(json_file) as f:
            doc_data = json.load(f)
            documents.append(
                Document(
                    page_content=doc_data["markdown"],
                    metadata={"source": doc_data["title"], "url": doc_data["url"]},
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    print(f"Adding {len(chunks)} chunks to vector store...")
    vector_store.add_documents(chunks)
    print("✓ Done!")


# ============================================================================
# PYDANTICAI AGENT WITH WEB UI
# ============================================================================
@dataclass
class RAGDeps:
    """Dependencies for the agent"""

    vector_store: PineconeVectorStore | PGVectorStore


# Create the agent - NO output_type needed for chat UI!
rag_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=RAGDeps,
    system_prompt="""
    You are a helpful medical assistant specializing in drug information.
    Use the retrieve tool to find information about drugs when users ask questions.
    Always cite your sources and be accurate.
    """,
)


@rag_agent.tool
async def retrieve_drug_info(ctx: RunContext[RAGDeps], query: str, k: int = 3) -> str:
    """
    Retrieve information about drugs from the medical database.

    Args:
        query: Search terms like drug name or medical condition
        k: Number of results to retrieve
    """
    results = ctx.deps.vector_store.similarity_search(query, k=k)

    if not results:
        return "No information found in the database."

    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[{i}] Source: {source}\n{doc.page_content}\n")

    return "\n---\n".join(context_parts)


# ============================================================================
# WEB UI SETUP
# ============================================================================
# Create dependencies instance
deps = RAGDeps(vector_store=vector_store)
# Create the web app - this replaces ask_drug_question!
app = rag_agent.to_web(deps=deps)

# Then run it with uvicorn main:app --reload.
