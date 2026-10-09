from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path


DENSE_MODELS = (
    "dragonkue/BGE-m3-ko",
)


def _load_env_file(project_dir: Path | None = None) -> None:
    """Load variables from .env file into os.environ if present."""
    candidates: list[Path] = []
    env_project = os.environ.get("KNOWLEDGE_PROJECT_ROOT")
    if env_project:
        candidates.append(Path(env_project).expanduser().resolve() / ".env")
    package_project = Path(__file__).resolve().parents[2]
    if project_dir and project_dir.resolve() != package_project.resolve():
        candidates.append(project_dir.resolve() / ".env")
    candidates.append(Path.cwd().resolve() / ".env")
    candidates.append(package_project.resolve() / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            try:
                import dotenv
                dotenv.load_dotenv(dotenv_path=candidate, override=False)
            except ImportError:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
            break


@dataclass(frozen=True, slots=True)
class Settings:
    DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
    DEFAULT_DENSE_MODEL = DENSE_MODELS[0]
    vault_root: Path
    runtime_dir: Path
    qdrant_url: str
    collection_name: str
    dense_model: str
    client_name: str
    project_root: Path | None = None
    embedding_batch_size: int = 8
    reranker_batch_size: int = 8
    cpu_threads: int = 4
    cuda_memory_fraction: float | None = None
    onnx_gpu_mem_limit: int | None = None
    onnx_arena_extend_strategy: str = "kSameAsRequested"
    onnx_intra_op_num_threads: int = 4

    qdrant_executable: str | None = None
    qdrant_native_storage: Path | None = None

    def __post_init__(self) -> None:
        if self.project_root is None:
            object.__setattr__(self, "project_root", self.vault_root / "00_System" / "knowledge-mcp")
        if self.project_root.resolve() == self.vault_root.resolve():
            raise ValueError("project_root must be outside the Vault root")
        if self.qdrant_native_storage is not None:
            value = Path(self.qdrant_native_storage).expanduser()
            if not value.is_absolute():
                raise ValueError("qdrant_native_storage must be an absolute path")
            object.__setattr__(self, "qdrant_native_storage", value.resolve())
        for name in ("embedding_batch_size", "reranker_batch_size", "cpu_threads", "onnx_intra_op_num_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if "batch_size" in name and value > 256:
                raise ValueError(f"{name} must be between 1 and 256")
        if self.cuda_memory_fraction is not None:
            if (isinstance(self.cuda_memory_fraction, bool)
                    or not isinstance(self.cuda_memory_fraction, (int, float))
                    or not 0 < self.cuda_memory_fraction <= 1):
                raise ValueError("cuda_memory_fraction must be greater than 0 and at most 1")
            object.__setattr__(self, "cuda_memory_fraction", float(self.cuda_memory_fraction))
        if self.onnx_gpu_mem_limit is not None:
            if type(self.onnx_gpu_mem_limit) is not int or self.onnx_gpu_mem_limit < 1:
                raise ValueError("onnx_gpu_mem_limit must be positive bytes")
        if self.onnx_arena_extend_strategy not in ("kNextPowerOfTwo", "kSameAsRequested"):
            raise ValueError("onnx_arena_extend_strategy must be kNextPowerOfTwo or kSameAsRequested")

    @property
    def project_dir(self) -> Path:
        return self.project_root

    @property
    def qdrant_storage_dir(self) -> Path:
        if self.qdrant_native_storage is not None:
            return self.qdrant_native_storage
        # Short paths avoid Windows Gridstore failures under deep checkouts.
        identity = os.path.normcase(str(self.project_root.resolve()))
        return Path.home() / ".knowledge-qdrant" / sha256(identity.encode()).hexdigest()[:12]

    @classmethod
    def from_paths(cls, *, vault_root: Path, project_root: Path) -> "Settings":
        vault_root = Path(vault_root).resolve()
        project_root = Path(project_root).resolve()
        return cls(
            vault_root=vault_root,
            runtime_dir=project_root / ".knowledge",
            qdrant_url=cls.DEFAULT_QDRANT_URL,
            collection_name="obsidian_knowledge_bge_m3_ko_v1",
            dense_model=cls.DEFAULT_DENSE_MODEL,
            client_name="test",
            project_root=project_root,
        )

    @classmethod
    def from_env(cls, client_name: str) -> "Settings":
        package_project = Path(__file__).resolve().parents[2]
        _load_env_file(package_project)
        raw_vault = os.environ.get("KNOWLEDGE_VAULT_ROOT")
        if not raw_vault:
            raise ValueError(
                "KNOWLEDGE_VAULT_ROOT must be configured in environment or .env file"
            )
        vault_root = Path(raw_vault).expanduser().resolve()
        project_root = Path(os.environ.get("KNOWLEDGE_PROJECT_ROOT", package_project)).expanduser().resolve()
        qdrant_url = os.environ.get("KNOWLEDGE_QDRANT_URL", cls.DEFAULT_QDRANT_URL)
        if qdrant_url != cls.DEFAULT_QDRANT_URL:
            raise ValueError("KNOWLEDGE_QDRANT_URL must be the local Qdrant URL")
        dense_model = os.environ.get("KNOWLEDGE_DENSE_MODEL", cls.DEFAULT_DENSE_MODEL)
        if dense_model not in DENSE_MODELS:
            raise ValueError(f"KNOWLEDGE_DENSE_MODEL must be one of {DENSE_MODELS}")
        return cls(
            vault_root=vault_root,
            runtime_dir=project_root / ".knowledge",
            qdrant_url=qdrant_url,
            collection_name=os.environ.get("KNOWLEDGE_COLLECTION", "obsidian_knowledge_bge_m3_ko_v1"),
            dense_model=dense_model,
            client_name=client_name,
            project_root=project_root,
            qdrant_executable=os.environ.get("KNOWLEDGE_QDRANT_EXECUTABLE") or None,
            qdrant_native_storage=Path(os.environ["KNOWLEDGE_QDRANT_NATIVE_STORAGE"])
                if os.environ.get("KNOWLEDGE_QDRANT_NATIVE_STORAGE") else None,
            embedding_batch_size=int(os.environ.get("KNOWLEDGE_EMBEDDING_BATCH_SIZE", "8")),
            reranker_batch_size=int(os.environ.get("KNOWLEDGE_RERANKER_BATCH_SIZE", "8")),
            cpu_threads=int(os.environ.get("KNOWLEDGE_CPU_THREADS", "4")),
            cuda_memory_fraction=float(os.environ["KNOWLEDGE_CUDA_MEMORY_FRACTION"])
                if os.environ.get("KNOWLEDGE_CUDA_MEMORY_FRACTION") else None,
            onnx_gpu_mem_limit=int(os.environ["KNOWLEDGE_ONNX_GPU_MEM_LIMIT"])
                if os.environ.get("KNOWLEDGE_ONNX_GPU_MEM_LIMIT") else None,
            onnx_arena_extend_strategy=os.environ.get("KNOWLEDGE_ONNX_ARENA_EXTEND_STRATEGY", "kSameAsRequested"),
            onnx_intra_op_num_threads=int(os.environ.get("KNOWLEDGE_ONNX_INTRA_OP_NUM_THREADS", "4")),
        )
