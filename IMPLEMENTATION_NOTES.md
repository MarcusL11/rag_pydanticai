## Full Pipeline with PydanticAI

### Firecrawl to scrape the data

- Set up firecrawl to scrape the data from the website.
- batch the data into list of dictionaries with keys url, markdown, title, and description.
- Save it to a directory to be used for chunking and embedding.

```python
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

```

### Populate the Vector Store with the scraped data

Choose a vector store and embedding model to use. In this example, I am going with PostgreSQL and OpenAI's text-embedding-3-small.

Then initialize the embedding model and configure the vector store.

Launch up your PostgreSQL database configure the vector store using Langchains's Postgres library for  `PGEngine` and `PGVectorStore`.

```python
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
```

>[!NOTE]
> Optionally, you can use Pinecone as your vector store, but I am using PostgreSQL for this example.

Then you will iterate through the sources saved from the scraping step, chunk the data using the recursive chunking strategy, and add the chunks to the vector store.  

Since I want to run this Agent Chat session multiple times, I'll warp it in a conditional statement to check if the vector store is already populated with the data, so that I don't have to repeat the chunking and embedding process every time I run the code.

### Setting up the PydanticAI Agent

Create a `RAGDeps` dataclass. `RAGDeps` is used to define the dependencies that will be injected into the RAG pipeline. It includes the vector store and the embedding model that will be used by the agent's retrieval tool.

```python
@dataclass
class RAGDeps:
    """Dependencies for the agent"""
    vector_store: PineconeVectorStore | PGVectorStore
```

Now, initialize the agent with a model, dependencies, structured output, and a system prompt. Then create a tool that allows the agent to perform similarity search on the vector store.

```python
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
```

Finally, you can test the agent by asking it a question and it will return the answer along with the source documents that were retrieved from the vector store. Then we can access this agent through a web UI that PydanticAI provides out of the box.

```python
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
```

Simply run `uvicorn main:app --reload`.
