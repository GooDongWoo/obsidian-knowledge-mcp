# Obsidian Knowledge MCP

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP%204.0.10-orange.svg)](https://github.com/PrefectHQ/fastmcp)
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

### 1.3 The Solution: Single HTTP Daemon + Lightweight Stdio Protocol Proxy
To permanently solve these resource bottlenecks, the architecture was restructured:
1. **Single HTTP Daemon**: A dedicated background service (`FastMCP 4.0.10`, MCP SDK `2.2.0`) hosts exactly **one** shared copy of the embedding and re-ranking models (~4.6GB total). It serves Streamable HTTP at `http://127.0.0.1:8765/mcp` and process health at `/health`.
2. **Lightweight Stdio Protocol Proxy**: AI clients launch `knowledge-mcp serve` without importing heavy ML libraries. The official FastMCP proxy mirrors the client's protocol era: modern clients use MCP `2026-07-28` on both sides, while legacy clients retain normal initialize-based negotiation. Modern `server/discover` requests are supported without a forced legacy fallback.
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

<p align="center">
  <a href="docs/obsidian-knowledge-architecture.html">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/architecture-light.png">
      <img alt="Obsidian Knowledge MCP Architecture" src="docs/architecture-dark.png" width="100%">
    </picture>
  </a>
</p>

<p align="center">
  <em>💡 Click the diagram above to open the <strong><a href="docs/obsidian-knowledge-architecture.html">Interactive Architecture Diagram (HTML)</a></strong> in your browser (supports zoom/pan, route inspection, and node search).</em>
</p>

<details>
<summary><b>View Text-based ASCII Diagram</b></summary>

```text
[ Obsidian Vault Files ] (.md, .txt, .pdf)
         │
         ▼
[ AI MCP Clients ] (Codex, Claude Code, Antigravity)
         │  (stdio)
         ▼
[ Stdio Protocol Proxy ]  <-- Official FastMCP proxy per client; era mirroring
         │  (Streamable HTTP: http://127.0.0.1:8765/mcp)
         ▼
[ Single HTTP Daemon (Port 8765, /health) ]
   ├── BGE-m3-ko Embedding Model (GPU / CUDA)
   ├── bge-reranker-v2-m3-ko (CPU Warmup)
   └── FastMCP Tools: qdrant-find, knowledge-index-status, knowledge-index-sync
         │
         ├──> [ Local Docker Qdrant ] (Port 6333 / 6334)
         │       └── Dense vectors, BM25 index, text chunks & payloads
         │
         └──> [ SQLite State ] (Project/.knowledge/state.sqlite3)
                 └── File sync manifests, generation logs, query history
```
</details>

Streamable HTTP uses one `/mcp` endpoint with POST requests and JSON or request-scoped SSE responses. Those SSE responses are distinct from the retired HTTP+SSE transport's persistent `/sse` connection and separate message endpoint. Modern requests use discovery and per-request protocol metadata without requiring `initialize`, a protocol session ID, or a separate GET stream. Direct HTTP, the default stdio proxy, and `serve --standalone` support MCP `2026-07-28`; standalone loads its own models and exposes the two read tools (`qdrant-find`, `knowledge-index-status`).

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

3. **Install PyTorch with CUDA support** (Optional, for GPU acceleration):
   ```powershell
   python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install dependencies**:
   ```powershell
   python -m pip install -e .
   python -m pip check
   ```
   To reproduce the Windows Python 3.13 CUDA environment, install the CUDA PyTorch build above first, then use `python -m pip install -c constraints/fastmcp4-windows-py313.txt -e '.[dev]'`. This constraints file describes that environment; it is not a cross-platform lockfile. FastMCP is pinned to `4.0.10` and the MCP SDK to `2.2.0`.

   Verify GPU availability:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```

The `qdrant-mcp` extra has been removed: its old `mcp-server-qdrant==0.8.1` dependency pins FastMCP `2.7.0` and Pydantic `<2.12.0`. If you need that separate server, keep it in a separate FastMCP 2 environment.

For an existing installation, create a fresh `.venv` in a separate checkout/worktree. Retain the original checkout, editable environment, and client settings for rollback; do not upgrade the original environment in place or copy a virtual environment. The migration changes transport and protocol, with no required collection rebuild, model change, or SQLite/Qdrant format change.

---

## 4. Quick Start

### 4.1 Set Environment Variables
Set the paths to your Obsidian Vault and this project repository:
```powershell
$env:KNOWLEDGE_VAULT_ROOT = "C:\Path\To\Your\Obsidian"
$env:KNOWLEDGE_PROJECT_ROOT = "C:\Path\To\obsidian-knowledge-mcp"
```

The project `.env` is loaded before FastMCP imports. Keep the daemon bound to localhost and use these defaults from `.env.example`:

```dotenv
FASTMCP_CHECK_FOR_UPDATES=off
FASTMCP_TELEMETRY_MODE=off
FASTMCP_SHOW_SERVER_BANNER=false
FASTMCP_DEPRECATION_WARNINGS=true
```

No global OpenTelemetry exporter is configured by this project. SDK tracing dependencies alone do not imply remote export; embedding and search remain local.

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

### 5.4 Existing Clients and Direct HTTP
For stdio clients, retain the `serve` command and point it to the new checkout's `.venv\Scripts\knowledge-mcp.exe`. For direct HTTP clients, change the URL from `/sse` to `http://127.0.0.1:8765/mcp` and select Streamable HTTP transport. Legacy client compatibility does not require the old `/sse` endpoint.

During cutover, close the old clients/proxies, stop the old daemon, and confirm its port/PID has been released before starting the new daemon. Check `/health`, discovery, the tool list, and a search, then reconnect clients. Do not run both daemons against the same runtime directory/port. To roll back, stop the new clients/daemon and restore the retained original checkout, environment, and client settings (including `/sse` for old direct HTTP configurations); do not delete the index.

---

## 6. MCP Tools Reference

| Tool Name | Description | Key Parameters |
| :--- | :--- | :--- |
| `qdrant-find` | Hybrid semantic + BM25 search across your Vault. | `query` (str, required)<br>`document_type` (list[str])<br>`file_type` (`["md", "txt", "pdf"]`)<br>`created_from`/`created_to` (`YYYY-MM-DD`)<br>`include_private` (bool, default: `false`)<br>`rerank` (bool, default: `false`)<br>`limit` (int, default: 8) |
| `knowledge-index-status` | Inspect indexing health, point counts, and errors. | *None* |
| `knowledge-index-sync` | Trigger an on-demand incremental sync from chat. | `rebuild` (bool, default: `false`) |

The proxy's tool-call timeout defaults to 15 seconds (`KNOWLEDGE_PROXY_TIMEOUT` overrides it). A timeout or cancellation does not establish whether sync completed: cancellation can stop work after partial progress. An interrupted index run records `partial` with `sync_cancelled`; the next ordinary incremental sync reconciles its manifest and vector generations. The proxy does not automatically retry tool calls. Inspect `knowledge-index-status` and the daemon logs before deciding whether to retry an incremental sync; never automatically resend sync or rebuild after a timeout.

Concurrent clients share a process startup lock, and sync calls queue asynchronously before taking the filesystem writer lock. Startup timeout terminates the owned unready process tree before releasing the startup lock. The proxy reuses its verified SSL context while the SDK creates independent backend sessions, avoiding repeated Windows trust-store loading without sharing protocol sessions.

Model loading, initial incremental sync, and reranker warmup have a separate 900-second startup deadline (`KNOWLEDGE_DAEMON_START_TIMEOUT`). Keep the MCP client's startup timeout at least this long; the normal tool-call deadline remains 15 seconds.

---

## 7. Documentation Index

For in-depth guides, operational scripts, and diagnostics:

- 📖 **[Korean User Guide (한국어 운영 가이드)](docs/user-guide-ko.md)**: Detailed PowerShell commands, manual rebuilds, step-by-step operations, and local setup scripts.
- 🔍 **[Qdrant Status & Diagnostics Guide](docs/qdrant-status-and-diagnostics.md)**: Explains status keywords (`completed`, `partial`, `parse_failed`), Qdrant REST APIs, SQLite schema, and unindexed file tracking scripts.
- 🏷️ **[Metadata & Configuration Guide](docs/metadata-and-configuration.md)**: Formatting YAML frontmatter, sidecar files, `.knowledgeignore`, `.knowledge-types.yaml`, and privacy levels (`public`/`private`).
- ⚡ **[Daemon Architecture & Resource Limits Session Log](docs/2026-09-20-daemon-architecture-and-resource-limits.md)**: Deep-dive root-cause analysis on RAM freezing and the 5-layer guardrail implementation.
- 📊 **[Interactive Architecture Diagram](docs/obsidian-knowledge-architecture.html)**: Standalone HTML architecture map with themes and component inspection.
