from src.base.redis_client import RedisClient

_SUMMARY_JOB_ID_KEY_PREFIX = "rag:index:summary_job:"


def _key(file_id: str) -> str:
    return f"{_SUMMARY_JOB_ID_KEY_PREFIX}{file_id}"


async def set_current_summary_job_id(file_id: str, summary_job_id: str) -> None:
    redis = await RedisClient.get_client(decode_responses=True)
    await redis.set(_key(file_id), summary_job_id)


async def get_current_summary_job_id(file_id: str) -> str:
    redis = await RedisClient.get_client(decode_responses=True)
    return str(await redis.get(_key(file_id)) or "")


async def clear_current_summary_job_id(file_id: str) -> None:
    redis = await RedisClient.get_client(decode_responses=True)
    await redis.delete(_key(file_id))
