# Obsidian Knowledge MCP

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP%202.7.0-orange.svg)](https://github.com/jlowin/fastmcp)
[![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant%20Local-red.svg)](https://qdrant.tech/)

A **local-first, privacy-preserving Model Context Protocol (MCP) server** designed for personal Obsidian Vaults. It provides hybrid dense/sparse retrieval with Korean-optimized re-ranking across Markdown, TXT, and text-extractable PDFs without sending a single byte of your personal notes to third-party cloud APIs.

---

## 1. Why This Project? (Engineering Background & Design Decisions)

### 1.1 Zero Cloud Leakage: Absolute Local Privacy
Personal notes in Obsidian frequently contain private journals, research drafts, financial thoughts, or credentials. Relying on cloud-based vector databases or third-party embedding APIs introduces unavoidable privacy risks.
- **100% On-Device**: Embeddings (`BGE-m3-ko`), cross-encoder re-ranking, and vector search (`Qdrant` on Docker) execute entirely on your local machine.
- **No External Telemetry**: No telemetry or query tracking leaves your computer.

### 1.2 Multi-Agent Resource Contention & System Freezing
In a modern development setup, multiple AI agents (e.g., **Codex**, **Claude Code**, and **Antigravity**) often run simultaneously. Under the standard MCP `stdio` model, each client launched its own Python worker process:
- **Duplicate Model Loading**: Each worker loaded both `multilingual-e5-large` (~2.2GB) and `BGE-m3-ko` (~2.2GB) into memory. Spawning 3 clients instantly consumed **over 13.5GB+ of RAM**.
- **VRAM Depletion & OS Freezing**: Running heavy neural models simultaneously exceeded the 12GB VRAM limit on consumer GPUs (such as the RTX 4070 Ti). Windows Display Driver Model (WDDM) responded by paging excess VRAM over the PCIe bus into system RAM, saturating the bus and causing Desktop Window Manager (`dwm.exe`) freezes and mouse lockups.
- **CPU Starvation**: Uncapped PyTorch / ONNX threads commandeered 100% of all logical CPU cores during tokenization.

### 1.3 The Solution: Single SSE Daemon + Lightweight Stdio Proxy
To permanently solve these resource bottlenecks, the architecture was restructured:
1. **Single SSE Daemon**: A dedicated background service (`FastMCP` with SSE on port 8765) hosts exactly **one** shared copy of the embedding and re-ranking models (~4.6GB total).
2. **Ultra-Lightweight Stdio Proxy**: AI clients launch a tiny proxy script (~15MB RAM) that imports **zero heavy ML libraries** (`torch`, `onnxruntime`, or `transformers`). The proxy forwards JSON-RPC messages between stdio and the SSE daemon over local HTTP.
3. **5-Layer Resource Guardrails**:
   - **CPU Thread Clamping**: PyTorch threads are clamped (`torch.set_num_threads(min(4, os.cpu_count()))`) to prevent CPU starvation.
   - **Batched Inferences**: FastEmbed and SentenceTransformer batch sizes are restricted to 32.
   - **Chunked Qdrant Upserts**: Points are upserted in batches of 64 to stop WSL2 Docker (`vmmemWSL`) memory ballooning.
   - **File Size Ceiling**: Files over 30MB are automatically skipped to protect parser memory.
   - **Container Memory Limit**: Qdrant container is capped at 4GB RAM via Docker Compose.

### 1.4 Hybrid Retrieval + Cross-Encoder Re-ranking
- **Dense Vector Search**: Powered by `dragonkue/BGE-m3-ko` (1024 dimensions) for deep semantic matching.
- **Sparse BM25 Index**: Built directly inside Qdrant to capture exact technical terms, symbols, and code identifiers.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse candidates into a unified rank.
- **Cross-Encoder Re-ranking**: Optional second-stage re-ranking via `dragonkue/bge-reranker-v2-m3-ko` running on CPU to ensure high relevance without consuming GPU VRAM.

---

## 2. Architecture

```text
[ Obsidian Vault Files ] (.md, .txt, .pdf)
         │
         ▼
[ AI MCP Clients ] (Codex, Claude Code, Antigravity)
         │  (stdio)
         ▼
[ Stdio Proxy (~15MB) ]  <-- Independent lightweight process per client
         │  (HTTP / SSE: http://127.0.0.1:8765)
         ▼
[ Single SSE Daemon (Port 8765) ]
   ├── BGE-m3-ko Embedding Model (GPU / CUDA)
   ├── bge-reranker-v2-m3-ko (CPU Warmup)
   └── FastMCP Tools: qdrant-find, knowledge-index-status, knowledge-index-sync
         │
         ├──> [ Local Docker Qdrant ] (Port 6333 / 6334)
         │       └── Dense vectors, BM25 index, text chunks & payloads
         │
         └──> [ SQLite State ] (Vault/.knowledge/state.sqlite3)
                 └── File sync manifests, generation logs, query history
```

> [!TIP]
> View the full interactive diagram in your browser:  
> 🔗 [Interactive Architecture Diagram (HTML)](docs/obsidian-knowledge-architecture.html)

---

## 3. Installation & Prerequisites

### 3.1 Prerequisites
- **Python**: 3.11 or higher
- **Docker Desktop**: Running with WSL2 backend (for Qdrant)
- **NVIDIA GPU** (Recommended): CUDA 12.x compatible GPU for fast embedding inference (falls back to CPU if unavailable).

### 3.2 Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GooDongWoo/obsidian-knowledge-mcp.git
   cd obsidian-knowledge-mcp
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -e .
   ```

4. **Install PyTorch with CUDA support** (Optional, for GPU acceleration):
   ```powershell
   pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
   ```
   Verify GPU availability:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```

---

## 4. Quick Start

### 4.1 Set Environment Variables
Set the paths to your Obsidian Vault and this project repository:
```powershell
$env:KNOWLEDGE_VAULT_ROOT = "C:\Path\To\Your\Obsidian"
$env:KNOWLEDGE_PROJECT_ROOT = "C:\Path\To\obsidian-knowledge-mcp"
```

### 4.2 Initial Indexing
Ensure Docker Desktop is running, then execute the initial index:
```powershell
.\.venv\Scripts\knowledge-mcp.exe index
```
- This automatically starts the Qdrant container if it is not already running.
- The first run downloads the embedding model and builds the initial index. Subsequent runs are fully incremental.

### 4.3 Background Daemon Management
Manage the shared background daemon using the `daemon` subcommands:
```powershell
# Start the background daemon
.\.venv\Scripts\knowledge-mcp.exe daemon start

# Check status and PID
.\.venv\Scripts\knowledge-mcp.exe daemon status

# Stop the daemon (unloads models from memory)
.\.venv\Scripts\knowledge-mcp.exe daemon stop
```

---

## 5. MCP Client Configuration

Register the server with your favorite AI coding assistants.

### 5.1 OpenAI Codex (`~/.codex/config.toml`)
```toml
[mcp_servers.obsidian_knowledge]
command = "C:\\Path\\To\\obsidian-knowledge-mcp\\.venv\\Scripts\\knowledge-mcp.exe"
args = ["serve", "--client", "codex"]
env = { KNOWLEDGE_VAULT_ROOT = "C:\\Path\\To\\Your\\Obsidian", KNOWLEDGE_PROJECT_ROOT = "C:\\Path\\To\\obsidian-knowledge-mcp" }
startup_timeout_sec = 900
```

### 5.2 Claude Code (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "obsidian_knowledge": {
      "command": "C:\\Path\\To\\obsidian-knowledge-mcp\\.venv\\Scripts\\knowledge-mcp.exe",
      "args": ["serve", "--client", "claude-code"],
      "env": {
        "KNOWLEDGE_VAULT_ROOT": "C:\\Path\\To\\Your\\Obsidian",
        "KNOWLEDGE_PROJECT_ROOT": "C:\\Path\\To\\obsidian-knowledge-mcp"
      }
    }
  }
}
```

### 5.3 Antigravity
Register in your workspace MCP configuration with `--client antigravity` and the corresponding environment variables.

---

## 6. MCP Tools Reference

| Tool Name | Description | Key Parameters |
| :--- | :--- | :--- |
| `qdrant-find` | Hybrid semantic + BM25 search across your Vault. | `query` (str, required)<br>`document_type` (list[str])<br>`file_type` (`["md", "txt", "pdf"]`)<br>`created_from`/`created_to` (`YYYY-MM-DD`)<br>`include_private` (bool, default: `false`)<br>`rerank` (bool, default: `false`)<br>`limit` (int, default: 8) |
| `knowledge-index-status` | Inspect indexing health, point counts, and errors. | *None* |
| `knowledge-index-sync` | Trigger an on-demand incremental sync from chat. | `rebuild` (bool, default: `false`) |

---

## 7. Documentation Index

For in-depth guides, operational scripts, and diagnostics:

- 📖 **[Korean User Guide (한국어 운영 가이드)](docs/user-guide-ko.md)**: Detailed PowerShell commands, manual rebuilds, step-by-step operations, and local setup scripts.
- 🔍 **[Qdrant Status & Diagnostics Guide](docs/qdrant-status-and-diagnostics.md)**: Explains status keywords (`completed`, `partial`, `parse_failed`), Qdrant REST APIs, SQLite schema, and unindexed file tracking scripts.
- 🏷️ **[Metadata & Configuration Guide](docs/metadata-and-configuration.md)**: Formatting YAML frontmatter, sidecar files, `.knowledgeignore`, `.knowledge-types.yaml`, and privacy levels (`public`/`private`).
- ⚡ **[Daemon Architecture & Resource Limits Session Log](docs/2026-09-20-daemon-architecture-and-resource-limits.md)**: Deep-dive root-cause analysis on RAM freezing and the 5-layer guardrail implementation.
- 📊 **[Interactive Architecture Diagram](docs/obsidian-knowledge-architecture.html)**: Standalone HTML architecture map with themes and component inspection.
