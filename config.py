from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Academic APIs
    semantic_scholar_key: str = ""
    arxiv_base_url: str = "https://export.arxiv.org/api"

    # LangSmith (optional tracing)
    langsmith_api_key: str = ""
    langsmith_project: str = "vfs-neuro-symbolic"
    langchain_tracing_v2: bool = False

    # App
    log_level: str = "INFO"
    max_parallel_agents: int = 4
    max_papers: int = 50
    min_triple_confidence: float = 0.7
    clingo_max_models: int = 10


settings = Settings()


if __name__ == "__main__":
    print(settings.model_dump())
