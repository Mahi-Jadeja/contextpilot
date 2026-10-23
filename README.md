# ContextPilot

### Adaptive Multi-Modal Context Fusion for Enterprise Financial LLMs

[![Live Application](https://img.shields.io/badge/Live_Deployment-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://contextpilot.streamlit.app/)
[![Repository](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Mahi-Jadeja/contextpilot)

---

ContextPilot is an enterprise decision-support demo that sits between business intelligence dashboards and large language models (LLMs). It retrieves relevant financial charts for a natural-language query, combines text and image similarity scores, filters duplicate chart descriptions, and packs selected chart context into a token budget before generating an answer.

The app also adds Indian Rupee (₹) financial examples to the prompt based on chart descriptions. These example metrics are deterministic demo values in the application code; the current app does **not** query a live Supabase database. Treat the metrics as synthetic demonstration data, not production financial data.

**Live Demo:** [contextpilot.streamlit.app](https://contextpilot.streamlit.app/)

## Core Features

| Feature | Description |
| :--- | :--- |
| **Dual-modal FAISS retrieval** | Encodes text queries and chart images in a shared 512-dimensional CLIP space and retrieves candidates from text and vision indices. |
| **Weighted score fusion** | Combines text and image similarity scores to rank charts for a query. |
| **Duplicate suppression** | Removes repeated chart descriptions so retrieved context can cover distinct analytical views. |
| **Token-budget optimization** | Uses `tiktoken` to estimate prompt size and greedily packs chart context within the selected budget. |
| **Financial demo context** | Adds deterministic Indian Rupee metrics to prompts based on chart descriptions and their relevant table labels. |
| **Multi-provider LLM routing** | Tries configured Groq models first, then Gemini models, and uses an offline mock response if providers are unavailable. |
| **Interactive Streamlit pages** | Explore the system overview, query retrieval, evaluation visualizations, and token packing. |

## How It Works

```text
User financial query
        |
        v
CLIP text encoder ------> FAISS text index
CLIP image encoder -----> FAISS vision index
        |                         |
        +------ candidate retrieval
                        |
                        v
          Weighted score fusion
          Description deduplication
          Top-K chart selection
                        |
                        v
       Add deterministic demo metrics
       Pack chart context to token budget
                        |
                        v
           Groq -> Gemini -> Mock
                        |
                        v
          Grounded demo response
          with chart references
```

ContextPilot does not replace a data warehouse or BI tool. It demonstrates how chart retrieval, context selection, and LLM response generation can be combined in a Streamlit application.

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| Frontend | Streamlit, Plotly, custom CSS |
| Backend | Python, Pandas, NumPy |
| Image and text embeddings | Hugging Face Transformers, OpenAI CLIP ViT-B/32 |
| Vector search | FAISS CPU, inner-product indices |
| Tokenization | `tiktoken` (`cl100k_base`) |
| LLM providers | Groq, Google Gemini |
| Chart metadata | CSV files |
| Deployment | Streamlit Community Cloud |

## Dataset and Demo Scope

- **Chart corpus:** 174 PNG charts across bar, horizontal bar, line, pie, scatter, and heatmap visualizations.
- **Metadata:** Chart descriptions, query intents, relevant table labels, and query-context triplets in `data/metadata/`.
- **Financial examples:** Deterministic demo metrics include Footwear revenue of ₹71,43,934.46, Accessories revenue of ₹70,34,730.61, and a Footwear return rate of 40.72%.
- **Schema context:** The demo refers to a 13-table commerce schema, including orders, order items, products, customers, sellers, returns, payments, shipments, inventory, and settlement-related tables. These are contextual labels; the running app does not connect to a relational database.

All commerce data and financial examples are synthetic and intended for academic evaluation. The application is not connected to live banking or production commerce systems.

## Application Pages

| Page | Purpose |
| :--- | :--- |
| **Overview** | System overview, architecture, and readiness information. |
| **Query Explorer** | Interactive query retrieval, fusion weights, chart results, token packing, and LLM responses. |
| **Evaluation & Analysis** | Benchmark cards and Plotly visualizations for retrieval and dataset analysis. |
| **Token Optimizer** | Playground for visualizing included and pruned chart context under a token budget. |

## Project Structure

```text
contextpilot/
├── app/
│   ├── core/
│   │   ├── config.py            # Paths and defaults
│   │   ├── data_loader.py       # Chart metadata and image loading
│   │   ├── embeddings.py        # CLIP text and image encoders
│   │   ├── retrieval.py         # FAISS candidate retrieval
│   │   ├── fusion.py            # Score fusion, deduplication, demo metrics
│   │   ├── token_optimizer.py   # Token-budget packing
│   │   ├── llm_client.py        # Groq, Gemini, and offline mock client
│   │   └── precompute.py        # Embedding and FAISS cache generation
│   ├── pages/
│   │   ├── 1_dashboard.py       # Overview
│   │   ├── 2_explorer.py        # Query Explorer
│   │   ├── 3_analysis.py        # Evaluation and analysis
│   │   └── 4_playground.py      # Token Optimizer
│   ├── ui/                      # Theme, styles, and reusable components
│   └── main.py                  # Streamlit entry point
├── data/
│   ├── charts/                  # Chart PNGs
│   └── metadata/                # Chart metadata and query triplets
├── cache/
│   ├── embeddings/              # Precomputed CLIP embeddings
│   └── faiss/                   # Text and vision FAISS indices
├── requirements.txt
└── README.md
```

## Local Setup

```bash
git clone https://github.com/Mahi-Jadeja/contextpilot.git
cd contextpilot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure provider keys using environment variables or a local Streamlit secrets file:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_key"
GEMINI_API_KEY = "your_gemini_key"
```

Keep real API keys out of source control. The `.streamlit/secrets.toml` file is for local use; on Streamlit Community Cloud, configure the keys in the app's **Secrets** settings.

Start the app from the project root:

```bash
PYTHONPATH=. ./venv/bin/streamlit run app/main.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deployment

| Resource | Link |
| :--- | :--- |
| Live application | [contextpilot.streamlit.app](https://contextpilot.streamlit.app/) |
| Source repository | [github.com/Mahi-Jadeja/contextpilot](https://github.com/Mahi-Jadeja/contextpilot) |

For Streamlit Community Cloud, set the main file path to `app/main.py` and add `GROQ_API_KEY` and `GEMINI_API_KEY` in the app's **Secrets** settings.

## Team and Assignment

| Field | Details |
| :--- | :--- |
| Institution | Symbiosis Institute of Technology (SIT), Pune |
| Program | B.Tech Artificial Intelligence |
| Assessment | ESE Mini Project |
| Group | Group 24 |
| Team members | Aarushi Mathur (PRN: 23070126002), Mahi Jadeja (PRN: 23070126068), Prisha Jaiswal (PRN: 23070126095), Tanmay Joshi (PRN: 23070126134) |
| Live application | [contextpilot.streamlit.app](https://contextpilot.streamlit.app/) |