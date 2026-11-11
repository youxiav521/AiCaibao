from __future__ import annotations

from typing import Any, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def env_field(default: Any, *names: str) -> Any:
    return Field(default=default, validation_alias=AliasChoices(*names))


class RagSettings(BaseSettings):
    """RAG 运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    vector_db_provider: str = env_field("milvus", "VECTOR_DB_PROVIDER")
    vector_langchain_collection: str = env_field("rag_grand_children_grandchildren","VECTOR_GRANDCHILDREN_COLLECTION",)

    milvus_host: str = env_field("127.0.0.1", "MILVUS_HOST")
    milvus_port: str = env_field("19530", "MILVUS_PORT")
    milvus_db_name: Optional[str] = env_field(None, "MILVUS_DB_NAME")
    milvus_metric: str = env_field("COSINE", "MILVUS_METRIC")
    milvus_index: str = env_field("IVF_FLAT", "MILVUS_INDEX")
    milvus_text_max_length: int = env_field(32768, "MILVUS_TEXT_MAX_LENGTH")
    milvus_insert_batch: int = env_field(256, "MILVUS_INSERT_BATCH")
    milvus_metadata_json_max_bytes: int = env_field(65000, "MILVUS_METADATA_JSON_MAX_BYTES")
    milvus_nlist: int = env_field(4096, "MILVUS_NLIST")
    shards_num: int = env_field(2, "SHARDS_NUM")

    es_host: str = env_field("10.192.72.13:9200", "ES_HOST")
    es_username: Optional[str] = env_field(None, "ES_USERNAME")
    es_password: Optional[str] = env_field(None, "ES_PASSWORD")
    es_request_timeout_s: int = env_field(120, "ES_REQUEST_TIMEOUT_S")
    es_bulk_chunk_size: int = env_field(200, "ES_BULK_CHUNK_SIZE")
    es_bulk_max_chunk_bytes: int = env_field(20 * 1024 * 1024, "ES_BULK_MAX_CHUNK_BYTES")
    es_bulk_refresh: bool = env_field(False, "ES_BULK_REFRESH")
    es_bulk_max_retries: int = env_field(2, "ES_BULK_MAX_RETRIES")
    es_bulk_initial_backoff_s: float = env_field(1.0, "ES_BULK_INITIAL_BACKOFF_S")
    es_bulk_max_backoff_s: float = env_field(8.0, "ES_BULK_MAX_BACKOFF_S")
    es_retry_on_timeout: bool = env_field(True, "ES_RETRY_ON_TIMEOUT")
    rag_backfill_batch_size: int = env_field(200, "RAG_BACKFILL_BATCH_SIZE")

    embedding_model: str = env_field("Qwen3-Embedding-8B", "EMBEDDING_MODEL")
    embedding_dim: int = env_field(4096, "EMBEDDING_DIM")
    embedding_send_dimensions: bool = env_field(False, "EMBEDDING_SEND_DIMENSIONS")
    embedding_batch_size: int = env_field(64, "EMBEDDING_BATCH_SIZE")
    embedding_cache_enabled: bool = env_field(True, "EMBEDDING_CACHE_ENABLED")
    embedding_cache_bucket: str = env_field("embedding-cache", "EMBEDDING_CACHE_BUCKET")
    embedding_cache_ttl_days: int = env_field(7, "EMBEDDING_CACHE_TTL_DAYS")
    embedding_cache_lookup_concurrency: int = env_field(32, "EMBEDDING_CACHE_LOOKUP_CONCURRENCY")
    embedding_base_url: Optional[str] = env_field(None, "EMBEDDING_API_BASE")
    embedding_api_key: Optional[str] = env_field(None, "EMBEDDING_API_KEY", "EMBEDDING_API_KEY_")
    pcr_minio_bucket: str = env_field("rag-index", "PCR_MINIO_BUCKET")
    rag_storage_collection_name: str = env_field("yu_rag_chunks", "RAG_STORAGE_COLLECTION_NAME")

    rerank_enabled: bool = env_field(True, "RERANK_ENABLED")
    rerank_model: str = env_field("Qwen3-Reranker", "RERANK_MODEL")
    rerank_base_url: Optional[str] = env_field(None, "RERANK_API_BASE", "RERANKER_API_BASE")
    rerank_api_key: Optional[str] = env_field(None, "RERANK_API_KEY")
    rerank_top_k: Optional[int] = env_field(None, "RERANK_TOP_K")
    rerank_endpoint_path: str = env_field("/v1/rerank", "RERANK_ENDPOINT_PATH", "RERANK_PATH")
    rerank_score_endpoint_path: str = env_field("/v1/score", "RERANK_SCORE_ENDPOINT_PATH", "SCORE_PATH")
    rerank_max_total_length: int = env_field(32768, "RERANK_MAX_TOTAL_LENGTH")
    rerank_score_concurrency: int = env_field(40, "RERANK_SCORE_CONCURRENCY")
    # 相关性阈值：重排器给出的 relevance_score 低于该值时视为不相关，丢弃该切片。
    # 仅用于"查询文本与文档完全不相关"时的兜底过滤，避免 kNN 强行返回低相关结果。
    rerank_score_threshold: float = env_field(0.5, "RERANK_SCORE_THRESHOLD")

    p_min: int = env_field(700, "P_MIN")
    p_max: int = env_field(2000, "P_MAX")
    p_hard_max: int = env_field(2400, "P_HARD_MAX")
    orphan_min: int = env_field(400, "ORPHAN_MIN")
    c_min: int = env_field(180, "C_MIN")
    c_max: int = env_field(380, "C_MAX")
    child_overlap: int = env_field(40, "CHILD_OVERLAP")

    default_top_k: int = env_field(8, "DEFAULT_TOP_K")
    index_wait_timeout_s: float = env_field(900.0, "RAG_INDEX_WAIT_TIMEOUT_S")
    delete_wait_timeout_s: float = env_field(900.0, "RAG_DELETE_WAIT_TIMEOUT_S")
    parent_summary_repair_wait_timeout_s: float = env_field(360.0, "RAG_PARENT_SUMMARY_REPAIR_WAIT_TIMEOUT_S")

    hybrid_enabled: bool = env_field(True, "HYBRID_ENABLED")
    hybrid_rrf_k: int = env_field(60, "HYBRID_RRF_K")
    hybrid_fusion_method: Optional[str] = env_field(None, "HYBRID_FUSION_METHOD")
    hybrid_vector_weight: float = env_field(0.7, "HYBRID_VECTOR_WEIGHT")
    hybrid_text_weight: float = env_field(0.3, "HYBRID_TEXT_WEIGHT")
    text_max_length: int = env_field(32768, "TEXT_MAX_LENGTH")

    table_split_threshold: int = env_field(1000, "TABLE_SPLIT_THRESHOLD")
    table_row_text_max_length: int = env_field(1000, "TABLE_ROW_TEXT_MAX_LENGTH")
    table_semantic_enrichment_enabled: bool = env_field(True, "TABLE_SEMANTIC_ENRICHMENT_ENABLED")
    table_semantic_enrichment_llm_caption_enabled: bool = env_field(
        True,
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED",
    )
    table_semantic_enrichment_llm_caption_overwrite_existing: bool = env_field(
        False,
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_OVERWRITE_EXISTING",
        "TABLE_LLM_CATION",
    )
    table_semantic_enrichment_llm_caption_model_id: str = env_field(
        "model_bailian_table_header_aided",
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_MODEL_ID",
    )
    table_semantic_enrichment_llm_caption_prev_max: int = env_field(
        3,
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_PREV_MAX",
    )
    table_semantic_enrichment_llm_caption_prev_max_bytes: int = env_field(
        6000,
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_PREV_MAX_BYTES",
    )
    table_semantic_enrichment_llm_caption_table_max_bytes: int = env_field(
        12000,
        "TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_TABLE_MAX_BYTES",
    )

    summary_index_enabled: bool = env_field(True, "SUMMARY_INDEX_ENABLED")
    summary_leaf_max_chars: int = env_field(2000, "SUMMARY_LEAF_MAX_CHARS")
    summary_leaf_max_recursion: int = env_field(3, "SUMMARY_LEAF_MAX_RECURSION")
    image_summary_llm_model_id: str = env_field("model_bailian_qwen3.5_35b", "IMAGE_SUMMARY_LLM_MODEL_ID")

    embedding_instruct_enabled: bool = env_field(False, "EMBEDDING_INSTRUCT_ENABLED")
    embedding_query_instruction: str = env_field(
        "Given a financial question, retrieve relevant passages from company annual reports and financial documents that contain the answer",
        "EMBEDDING_QUERY_INSTRUCTION",
    )

    rag_reset_enabled: bool = env_field(False, "RAG_RESET_ENABLED")
    rag_reset_password: Optional[str] = env_field(None, "RAG_RESET_PASSWORD", "RAG_ADMIN_TOKEN")
    embedding_dim_enabled: bool = env_field(True, "EMBEDDING_DIM_ENABLED")
    structure_recall_model_id: str = env_field("structure_recall_model","STRUCTURE_RECALL_MODEL_ID")

    structure_recall_prompt_max_tokens: int = env_field(300000,"STRUCTURE_RECALL_PROMPT_MAX_TOKENS")

def get_rag_settings() -> RagSettings:
    return RagSettings()


RAG_SETTINGS = get_rag_settings()
