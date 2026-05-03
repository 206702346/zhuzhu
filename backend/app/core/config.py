from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upload_dir: str = "data/uploads"
    index_dir: str = "data/index"
    metadata_dir: str = "data/metadata"

    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model_name: str = ""

    class Config:
        env_file = ".env"


settings = Settings()