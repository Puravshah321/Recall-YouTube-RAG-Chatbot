# Recall — YouTube RAG Chatbot

> Ask anything about any YouTube video — powered by Groq LPU, local embeddings, and the YouTube Data API v3.

<br>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-recall--youtube--rag--chatbot-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://recall-youtube-rag-chatbot.onrender.com/)

> 🚀 **Try it live:** [https://recall-youtube-rag-chatbot.onrender.com/](https://recall-youtube-rag-chatbot.onrender.com/)

<br>

**Recall** is an end-to-end **Retrieval-Augmented Generation (RAG)** application that transforms any YouTube video into an intelligent, conversational knowledge system.

You paste a YouTube URL → Recall fetches the transcript via the **YouTube Data API v3** → builds a semantic vector index using local HuggingFace embeddings → and lets you chat with that video in plain English.

Embeddings are generated **locally** using `BAAI/bge-small-en-v1.5` (no API cost, full privacy). Inference is handled by **Groq's free cloud API** on their LPU hardware for blazing-fast responses. Transcript fetching is powered by the **official Google Cloud YouTube Data API v3**, with automatic fallback tiers to maximise coverage.

<br>

---

## Pipeline Architecture

<br>

![YouTube RAG Pipeline](src/youtube%20rag%20pipleline.png)

<br>

---

## Pipeline Explained

<br>

### 1 · User Input

The user pastes a YouTube video URL into the sidebar. `extract_video_id()` in `youtube_utils.py` uses Python's `urllib.parse` to isolate the video ID from both long-form (`youtube.com/watch?v=...`) and short-form (`youtu.be/...`) URL formats.

<br>

### 2 · Transcript Extraction (4-Tier Fallback)

`get_transcript()` uses the YouTube Data API v3 as the primary engine, with automatic fallback layers:

| Tier | Method | When it activates |
|------|--------|-------------------|
| 1️⃣ | `youtube.captions().download()` — official SRT download | Always tried first |
| 2️⃣ | `youtube-transcript-api` — English captions | If Tier 1 returns 401/403 |
| 3️⃣ | `youtube-transcript-api` — any available language | If no English track found |
| 4️⃣ | `youtube.videos().list()` — title + description | Last resort if all captions fail |

All output is merged into a single raw text string — the raw input for the entire pipeline.

<br>

### 3 · Text Splitting & Chunking

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([full_text])
```

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | `1000` chars | ≈ 250 tokens — well within `bge-small`'s 512-token limit |
| `chunk_overlap` | `200` chars | 20% overlap prevents answers being split across chunk boundaries |

<br>

### 4 · Embedding Generation

```python
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device': 'cpu'}
)
```

Each chunk is converted into a **384-dimensional dense vector**. These vectors encode *semantic meaning* — enabling meaning-based retrieval instead of keyword matching.

<br>

### 5 · FAISS Vector Index

```python
vector_store = FAISS.from_documents(chunks, embeddings)
```

All chunk embeddings are stored in a FAISS in-memory flat L2 index. A typical 15-minute video produces ~80–150 chunks — trivially small for FAISS, searched in microseconds on CPU with zero external infrastructure.

<br>

### 6 · Query Embedding & Retrieval

```python
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 6}
)
```

When the user submits a question, it is embedded by the same model. FAISS returns the **top `k=6`** most semantically relevant transcript chunks.

<br>

### 7 · Prompt Construction

A strict closed-world `PromptTemplate` is assembled. Retrieved chunks are injected as `{context}` and the user's question as `{question}`. The LLM is instructed to answer **only** from the provided context and say *"I don't know"* if the context is insufficient — keeping responses grounded.

<br>

### 8 · LLM Inference (Groq)

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
```

The prompt is routed to **Groq** running `llama-3.1-8b-instant` on custom LPU hardware — significantly faster than local CPU inference.

<br>

### 9 · Answer Output

`StrOutputParser` extracts the plain text response, which is displayed in the Streamlit chat interface with a timestamp.

<br>

---

## The Full Journey — Every Stage, Every Problem

This project went through **five major architectural shifts** before reaching its current form. Each stage solved the previous stage's problem and introduced a new one.

<br>

### Stage 1 — Google AI Studio + Gemini (First Prototype)

The very first version used **Google AI Studio's Gemini API** for both embedding generation and chat inference — the fastest way to bootstrap a RAG prototype.

**The problem:** Gemini's free tier enforces strict **rate limits and per-minute token quotas**. On videos longer than 10–15 minutes, the embedding step alone would push the session close to its ceiling. Running the full pipeline — extract → embed → retrieve → generate — would reliably trigger a `RESOURCE_EXHAUSTED` (HTTP 429) error mid-session.

A chatbot that crashes halfway through a conversation is not a product.

<br>

### Stage 2 — Hugging Face Embeddings + Ollama LLM (Local Everything)

To eliminate the API dependency entirely, two substitutions were made:

| Component | Before | After |
|---|---|---|
| Embeddings | Gemini Embedding API | `BAAI/bge-small-en-v1.5` via Hugging Face (local) |
| Chat Inference | Gemini Flash API | Microsoft `phi` model via Ollama (local) |

**Result:** Zero API failures, zero token quotas, full transcript privacy. Worked perfectly on local hardware.

<br>

### Stage 3 — The Ollama Deployment Problem

Ollama works beautifully on local hardware — but deploying it to the cloud exposed a new hard constraint.

The `phi` model requires approximately **2 GB of RAM** to load and serve. Cloud free tiers (Render, Railway, Fly.io) cap memory at **512 MB**. Ollama cannot run inside those constraints.

Building a Docker image with Ollama baked in also ballooned the container size to several gigabytes and introduced a fragile startup sequence (start daemon → wait for model pull → start Streamlit) — completely unsuitable for a stateless web service.

<br>

### Stage 4 — Groq (Cloud LLM, Local Embeddings)

The solution was to keep the **local embedding pipeline unchanged** (HuggingFace + FAISS runs fine in 512 MB) and replace only the LLM call with **Groq's free cloud API**.

| Component | Before | After |
|---|---|---|
| Chat Inference | Microsoft `phi` via Ollama | `llama-3.1-8b-instant` via Groq LPU |

Groq runs inference on custom **LPU (Language Processing Unit)** hardware — faster than local CPU inference, requires zero server-side RAM, and comes with a generous free tier.

**However:** the initial model used was `llama3-70b-8192`, which was later **decommissioned by Groq**. The app began throwing `model_decommissioned` errors. Updated to `llama-3.1-8b-instant` — the current stable replacement.

<br>

### Stage 5 — Transcript Layer: apify → YouTube Data API v3 (Current)

The transcript fetching layer also went through multiple iterations:

#### Phase 5a — `youtube-transcript-api` (direct scraping library)
The initial simple approach — worked for most public videos but had **no authentication**, was fragile to YouTube's occasional blocking of scraping, and provided no fallback for age-gated or restricted videos.

#### Phase 5b — Apify Actor (`thescrapelab/apify-youtube-transcript-scraper-2-0`)
To get production reliability, switched to **Apify** — a managed web scraping cloud platform with a dedicated YouTube transcript actor.

**The problems with Apify:**
- Required an `APIFY_API_TOKEN` secret on top of all existing keys
- Every transcript fetch made **remote API calls to Apify's cloud** — introducing latency and runtime cost
- The free tier limits were tight; the actor sometimes returned empty datasets for restricted videos
- Extra dependency (`apify-client`) added ~15 MB and a cloud billing risk

#### Phase 5c — YouTube Data API v3 (Current Final Solution) ✅

Switched to the **official Google Cloud YouTube Data API v3** using a project API key from Google Cloud Console.

| Aspect | Apify | YouTube Data API v3 |
|---|---|---|
| Authentication | Third-party token | Official Google Cloud API key |
| Data source | Scraped | Official Google servers |
| Reliability | Fragile to YouTube changes | Stable official contract |
| Fallback logic | None | 4-tier automatic fallback |
| Free quota | Very limited | 10,000 units/day |
| Dependency | `apify-client` | `google-api-python-client` |

The new `get_transcript()` in `youtube_utils.py` uses a **4-tier fallback strategy**:
1. Official `captions().download()` — SRT via API key
2. `youtube-transcript-api` — English public captions
3. `youtube-transcript-api` — any language
4. `videos().list()` — title + description as fallback context

<br>

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![YouTube Data API](https://img.shields.io/badge/YouTube_Data_API_v3-FF0000?style=flat-square&logo=youtube&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

<br>

---

## How We Built This

<br>

### Step 1 — URL Parsing &nbsp;`youtube_utils.py`

```python
parsed_url = urlparse(url)
query_params = parse_qs(parsed_url.query)
return query_params.get("v", [None])[0]
```

Handles both `youtube.com/watch?v=...` and `youtu.be/...` formats, returning the raw video ID string.

<br>

### Step 2 — Transcript Extraction &nbsp;`youtube_utils.py`

```python
youtube = build("youtube", "v3", developerKey=api_key)

# Try official caption download first
captions_response = youtube.captions().list(part="snippet", videoId=video_id).execute()
raw_bytes = youtube.captions().download(id=caption_id, tfmt="srt").execute()
transcript_text = _parse_srt(raw_bytes.decode("utf-8"))
```

The SRT parser (`_parse_srt`) strips timestamp and index lines using regex, returning clean continuous text.

<br>

### Step 3 — Text Splitting &nbsp;`chatbot_engine.py`

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([full_text])
```

<br>

### Step 4 — Embedding Generation &nbsp;`chatbot_engine.py`

```python
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    model_kwargs={'device': 'cpu'}
)
```

**Why `BAAI/bge-small-en-v1.5`?**
- Top-tier score on the [MTEB Retrieval Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) at its size class
- Only ~133 MB — loads in seconds, runs comfortably on CPU
- Designed for **asymmetric retrieval**: short queries fetching longer passages — exactly this use case

<br>

### Step 5 — FAISS Vector Store &nbsp;`chatbot_engine.py`

```python
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 6})
```

| Parameter | Value | Reason |
|---|---|---|
| `k` | `6` | 6 × ~250 tokens ≈ 1500 tokens context — fits within Llama's window |
| `search_type` | `'similarity'` | Cosine similarity — most accurate for small indexes |

<br>

### Step 6 — LLM Setup &nbsp;`chatbot_engine.py`

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
```

| Parameter | Value | Reason |
|---|---|---|
| `model` | `llama-3.1-8b-instant` | Replaced deprecated `llama3-8b-8192`; fast on Groq LPU |
| `temperature` | `0.3` | Low enough for factual grounding; high enough to avoid robotic output |

> `temperature=0.0` is fully deterministic but sounds mechanical.
> Above `0.5`, the model starts drifting beyond the retrieved context.

<br>

### Step 7 — Prompt Template &nbsp;`chatbot_engine.py`

```python
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the provided transcript context.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)
```

The **closed-world instruction** ("Answer ONLY using...") is the most critical line. Without it, the model blends retrieved context with its own training data — producing confident but **ungrounded** answers.

<br>

### Step 8 — Chain Assembly &nbsp;`chatbot_engine.py`

```python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
main_chain = parallel_chain | prompt | llm | parser
```

LangChain's LCEL composes the pipeline with the `|` pipe operator:

```
Query → [Retrieval ‖ Passthrough] → Prompt → LLM → String Output
```

`RunnableParallel` runs retrieval and question passthrough simultaneously, reducing total latency.

<br>

### Step 9 — Streamlit Frontend &nbsp;`app.py`

| Feature | Detail |
|---|---|
| Sidebar input | Accepts the YouTube URL |
| Initialize Engine | Triggers the full RAG pipeline build |
| Video preview | Embedded inline after URL is loaded |
| Stat cards | Live message count + engine status |
| Chat interface | `st.chat_input` + `st.chat_message` with per-message timestamps |
| Clear Chat | Resets conversation without rebuilding the engine |

<br>

---

## Key Design Decisions & Tradeoffs

<br>

### Embeddings — Hugging Face vs API

| | Hugging Face (chosen) | API-based |
|---|---|---|
| Cost | Free forever | Usage-billed |
| Privacy | Fully local | Data leaves machine |
| Speed | ~0.5–2 s/batch on CPU | Faster, but network round-trip |
| Reliability | No quota | Rate-limited |
| First run | ~133 MB download | Instant |

<br>

### LLM — Groq vs Ollama (Local)

| | Groq (chosen) | Ollama |
|---|---|---|
| Speed | ~200–500 tokens/s on LPU | ~5–15 tokens/s on CPU |
| RAM usage | 0 (server-side) | 2–8 GB depending on model |
| Cloud deployable | ✅ Yes | ❌ Not on free tiers |
| Offline | ❌ Needs internet | ✅ Fully offline |
| Free tier | Generous (daily quota) | Unlimited (hardware bound) |

> The Ollama code is preserved as commented-out blocks in `chatbot_engine.py` for local development use.

<br>

### Transcript — YouTube Data API v3 vs Apify

| | YouTube Data API v3 (chosen) | Apify |
|---|---|---|
| Source | Official Google API | Third-party scraper |
| Stability | Contractually stable | Fragile to YouTube changes |
| Auth | Google Cloud API key | Apify platform token |
| Free quota | 10,000 units/day | Very limited |
| Fallback logic | 4-tier built-in | None |
| Package | `google-api-python-client` | `apify-client` |

<br>

---

## Project Structure

```
youtube_chatbot/
│
├── app.py                 ← Streamlit UI (Recall frontend)
├── chatbot_engine.py      ← RAG chain: split → embed → index → retrieve → generate
├── youtube_utils.py       ← URL parsing & YouTube Data API v3 transcript extraction
├── requirements.txt       ← All Python dependencies
├── .env                   ← Environment variables (GROQ_API_KEY, YOUTUBE_API_KEY)
├── Dockerfile             ← Production container for Render deployment
├── render.yaml            ← Render deployment config
│
├── src/
│   └── youtube rag pipleline.png
│
└── README.md
```

<br>

---

## How to Run

<br>

### Option A — Run Locally (with Groq + YouTube Data API)

**Prerequisites**
- Python 3.9+
- Free [Groq API key](https://console.groq.com)
- [Google Cloud API key](https://console.cloud.google.com) with **YouTube Data API v3** enabled

**1. Clone the repository**

```bash
git clone https://github.com/Puravshah321/langchain_campusx.git
cd langchain_campusx/youtube_chatbot
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your API keys in `.env`**

```env
GROQ_API_KEY=your_groq_api_key_here
YOUTUBE_API_KEY=your_google_cloud_api_key_here
```

**4. Run the app**

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`.

<br>

### Option B — Run Locally with Ollama (No Cloud LLM)

**Prerequisites**
- Python 3.9+
- [Ollama](https://ollama.com/) installed
- Google Cloud API key with YouTube Data API v3 enabled

**1. Pull and serve the model**

```bash
ollama pull phi
ollama serve
```

**2. In `chatbot_engine.py`, comment out Groq and uncomment Ollama:**

```python
# Uncomment this:
llm = ChatOllama(model="phi", temperature=0.3)

# Comment out this:
# llm = ChatGroq(...)
```

**3. Set only YouTube key in `.env`**

```env
YOUTUBE_API_KEY=your_google_cloud_api_key_here
```

**4. Run**

```bash
python -m streamlit run app.py
```

<br>

### Option C — Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo, set **Root Directory** to `youtube_chatbot`
4. Render auto-detects the `Dockerfile` ✅
5. Set **Plan** to **Free**
6. Add environment variables:
   - `GROQ_API_KEY` = your Groq key
   - `YOUTUBE_API_KEY` = your Google Cloud key
7. Click **Deploy**

<br>

**Using the app**

1. Paste any YouTube URL in the sidebar
2. Click **Initialize Engine**
3. Wait for *"Engine ready. Start asking."*
4. Ask anything about the video

<br>

---

## Contributors

**Purav Shah**
