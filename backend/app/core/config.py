from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upload_dir: str = "data/uploads"
    index_dir: str = "data/index"
    metadata_dir: str = "data/metadata"

    embedding_model_name: str = "./models/bge-small-zh-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5

    rerank_enabled: bool = True
    rerank_model_name: str = "./models/bge-reranker-v2-m3"
    
    # 权重，用于调整词汇匹配的贡献
    lexical_boost_weight: float = 1.0

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model_name: str = ""

    class Config:
        env_file = ".env"


settings = Settings()