---
name: querying-obsidian-knowledge
description: Use when searching an Obsidian Vault semantically for past experiences, projects, decisions, related notes, or duplicate and connection candidates.
---

# Querying Obsidian Knowledge

Use when finding existing knowledge or experiences in the Vault, or determining candidates for deduplication and connection with new records. Searching is a read-only investigation; a search request alone does not grant authority to modify the Vault or synchronize the index.

## 1. Tool Invocation Architecture and Client-Specific Protocols

The invocation method for `obsidian_knowledge` MCP tools may differ depending on the AI client being used:

- **Claude Code / OpenAI Codex (Native Bindings)**:
  - MCP tools are directly exposed in the callable function list (`qdrant-find`, `knowledge-index-status`, etc.).
  - In this case, invoke them **directly as native functions** without meta-tools.
- **Antigravity (Lazy Loading / Lazy MCP)**:
  - Tools are lazily loaded and not directly visible in the function declaration list to conserve tokens.
  - Invoke them via the meta-tool `call_mcp_tool(ServerName="obsidian_knowledge", ToolName="...", Arguments={...})`.
- ⚠️ **No Early Fallback**:
  - Do not mistakenly conclude "the tool does not exist" simply because it is not directly visible in the function list (such as in Antigravity environments) and bypass it with PowerShell/grep keyword search.
  - Fall back to keyword search only when an actual execution error response is received during the tool call (e.g., daemon not running).

---

## 2. Tool Invocation Specifications and Parameters

### A. Search: `qdrant-find`

#### Invocation Examples
- **Claude Code / Codex (Direct Invocation)**:
  ```json
  qdrant-find(
    query="<query for independent unit concept>",
    rerank=true,
    limit=8,
    include_private=false
  )
  ```
- **Antigravity (Meta-Tool Invocation)**:
  ```json
  call_mcp_tool(
    ServerName="obsidian_knowledge",
    ToolName="qdrant-find",
    Arguments={
      "query": "<query for independent unit concept>",
      "rerank": true,
      "limit": 8,
      "include_private": false
    }
  )
  ```

#### Parameter Constraints
- `query` (string, required): Search query string (minimum 1 character excluding whitespace).
- `limit` (integer, default: 8): **Range of 1 to 20 required** (must never exceed 20 due to the `le=20` constraint).
- `rerank` (boolean, default: `false`):
  - `false`: RRF (Dense + BM25) hybrid fusion score (0 to less than 1).
  - `true`: Cross-Encoder (`bge-reranker-v2-m3-ko`) second-stage reranking score. Recommended when semantic relevance is critical.
- `include_private` (boolean, default: `false`): Whether to include `security: private` documents. Set to `true` only when the user explicitly requests or approves including private documents.
- `document_type` (list[string] | null): Document type filter.
  - Default supported types: `"diary"` (`10_Daily/**`), `"experience"` (`**/*경험*`), `"brainstorming"` (`**/*아이디어*`, `**/*브레인스토밍*`), `"award"` (`**/*수상*`, `**/*상장*`), `"other"` (miscellaneous).
- `file_type` (list[string] | null): File extension filter (choose from `["md"]`, `["txt"]`, `["pdf"]`).
- `created_from` / `created_to` (string | null): Creation date range (`YYYY-MM-DD` format, 00:00:00 to 23:59:59 Seoul time).
- `modified_from` / `modified_to` (string | null): Modification date range (`YYYY-MM-DD` format).

#### Returned Data (`SearchResult`) Structure and Source Verification
Search results are returned in the following structure:
```json
{
  "source_path": "Vault relative path (e.g., 20_Projects/.../note.md)",
  "document": "Chunk body text",
  "heading_path": ["Main Heading", "Subheading"],
  "start_line": 15,
  "end_line": 32,
  "page": null,
  "score": 0.032,
  "document_type": "diary",
  "file_type": "md",
  "security_level": "public",
  "created_at": "...",
  "modified_at": "..."
}
```
- **Source Verification**: When checking the original text with `view_file`, refer to `source_path` along with `start_line` ~ `end_line` and `heading_path` to understand the exact context.

---

### B. Index Status Check: `knowledge-index-status`

Call to check index health before searching, or to verify reflection after synchronization.

- **Claude Code / Codex (Direct Invocation)**:
  ```json
  knowledge-index-status()
  ```
- **Antigravity (Meta-Tool Invocation)**:
  ```json
  call_mcp_tool(
    ServerName="obsidian_knowledge",
    ToolName="knowledge-index-status",
    Arguments={}
  )
  ```

- Return Status:
  - `status`: `"completed"` (normal), `"partial"` (some documents failed parsing or were unextracted), `"failed"` (error).
  - Even if the status is `partial`, search can proceed because most valid documents are indexed, but state in the response that results may be incomplete.

---

### C. Index Synchronization: `knowledge-index-sync`

Call when the user explicitly requests synchronization or when updating the index after writing new documents to the Vault.

- **Claude Code / Codex (Direct Invocation)**:
  ```json
  knowledge-index-sync(rebuild=false)
  ```
- **Antigravity (Meta-Tool Invocation)**:
  ```json
  call_mcp_tool(
    ServerName="obsidian_knowledge",
    ToolName="knowledge-index-sync",
    Arguments={
      "rebuild": false
    }
  )
  ```

- ⚠️ **Handling the 15-Second Proxy Timeout**:
  - The timeout for the FastMCP stdio proxy is 15 seconds. If an indexing operation takes longer, a timeout message may be returned, but synchronization may still be progressing in the background daemon.
  - Even if a timeout occurs, do not immediately assume failure or re-invoke synchronization; instead, **call `knowledge-index-status` to check the status**.
  - `rebuild: true` deletes the entire index and reconstructs it from scratch, so never execute it without explicit user approval.

---

## 3. Search Procedure and Query Decomposition

1. **Identify Search Scope and Independent Unit Concepts**:
   - Do not send a long, compound request as a single search query.
   - Decompose into independent unit knowledge—such as project names, problems/symptoms, technical elements, design decisions, and verification methods—and query each separately.
   - However, do not artificially over-fragment a single concept.
2. **Tool Invocation**:
   - Call `qdrant-find` via `call_mcp_tool` (or native function where applicable).
   - Complement queries with keywords/synonyms if necessary.
3. **Source Cross-Verification**:
   - Run `view_file` based on the returned `source_path` and `start_line`~`end_line` to verify original context.
   - Do not answer the user or update notes based solely on search snippets without verification.

---

## 4. Example: Compound Query Decomposition

For a request such as "Check if my CUDA kernel optimization experience can be connected to the current project":

1. `CUDA kernel optimization performance bottleneck measurement experience`
2. `CUDA shared memory or memory coalescing adoption decision`
3. `Goals, results, and lessons learned from CUDA projects`

Verify the actual project name, user role, measurement evidence, and decision context from the source text for each result.

---

## 5. Common Misconceptions and Prevention Rules

- ❌ Misjudging "tool missing" because `qdrant-find` is not in the tool declaration list and immediately falling back to text grep search (-> **Invoke via `call_mcp_tool`**)
- ❌ Passing a value exceeding 20 to the `limit` parameter, causing a validation error (-> **Specify at most 20 or less**)
- ❌ Reading only the search chunk text and skipping verification of the source text (`source_path`)
- ❌ Blindly calling `rebuild: true` when a 15-second timeout occurs
