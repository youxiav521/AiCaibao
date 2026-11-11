from typing import Callable, Dict, Optional

from loguru import logger

try:
    from src.crypt.sm4.sm4_crypt import SM4
except ImportError:
    logger.warning("SM4 模块导入失败，SM 加密类型不可用")
    SM4 = None


DECRYPT_PREFIX = "DECRYPT"
DECRYPT_PREFIX_LENGTH = len(DECRYPT_PREFIX)
ALLOWED_DELIMITERS = {":", "@", "#", "$"}


_decryptors: Dict[str, Callable[[str], str]] = {}


def register_decryptor(crypt_type: str, decrypt_func: Callable[[str], str]) -> None:
    _decryptors[crypt_type] = decrypt_func
    logger.debug("注册解密器 | crypt_type={}", crypt_type)


def get_decryptor(crypt_type: str) -> Optional[Callable[[str], str]]:
    return _decryptors.get(crypt_type)


def decrypt_value(value: str, crypt_type: str) -> str:
    if not isinstance(value, str) or not value.startswith(DECRYPT_PREFIX):
        return value

    if len(value) <= DECRYPT_PREFIX_LENGTH + 1:
        return value

    delimiter = value[DECRYPT_PREFIX_LENGTH]
    if delimiter not in ALLOWED_DELIMITERS:
        return value

    decryptor = get_decryptor(crypt_type)
    if not decryptor:
        logger.warning("未找到解密器 | crypt_type={}", crypt_type)
        return value

    ciphertext = value[DECRYPT_PREFIX_LENGTH + 1 :]

    try:
        return decryptor(ciphertext)
    except Exception:
        logger.exception("解密失败 | crypt_type={}, value={}", crypt_type, value[:20])
        return value


if SM4:
    register_decryptor("SM", SM4.decrypt_wrap)