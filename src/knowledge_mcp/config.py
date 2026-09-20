from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.project_root is None:
            object.__setattr__(self, "project_root", self.vault_root / "00_System" / "knowledge-mcp")
        if self.project_root.resolve() == self.vault_root.resolve():
            raise ValueError("project_root must be outside the Vault root")

    @property
    def project_dir(self) -> Path:
        return self.project_root

    @property
    def qdrant_storage_dir(self) -> Path:
        return self.runtime_dir / "qdrant"

    @classmethod
    def from_paths(cls, *, vault_root: Path, project_root: Path) -> "Settings":
        vault_root = Path(vault_root).resolve()
        project_root = Path(project_root).resolve()
        return cls(
            vault_root=vault_root,
            runtime_dir=vault_root / ".knowledge",
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
            runtime_dir=vault_root / ".knowledge",
            qdrant_url=qdrant_url,
            collection_name=os.environ.get("KNOWLEDGE_COLLECTION", "obsidian_knowledge_bge_m3_ko_v1"),
            dense_model=dense_model,
            client_name=client_name,
            project_root=project_root,
        )
