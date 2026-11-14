from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",extra="ignore")
    aws_region:str='us-east-1'
    aws_access_key_id:str=''
    aws_secret_access_key:str=''
    s3_bucket:str=''
    database_url:str=''  # Render PostgreSQL External Database URL
    # --- Bedrock (disabled until Bedrock model access is approved; uncomment then) ---
    # bedrock_embed_model:str='amazon.titan-embed-text-v2:0'
    # bedrock_chat_model:str='amazon.nova-lite-v1:0'
    # --- Ollama (temporary local stand-in for Bedrock) ---
    ollama_base_url:str='http://localhost:11434'
    ollama_chat_model:str='qwen3:8b'
    ollama_embed_model:str='mxbai-embed-large'  # 1024 dims, same as Titan v2
    embed_dim:int=1024
    max_upload_bytes:int=10485760
    pinecone_api_key:str=''
    pinecone_index_name:str='atlas-enterprise-ai-index'
    pinecone_cloud:str='aws'
    pinecone_region:str='us-east-1'
    pinecone_ready_timeout:int=180
    min_relevance:float=0.25
    hitl_enabled:bool=True
    memory_turns:int=6

@lru_cache

def get_settings() -> Settings:
    return Settings()