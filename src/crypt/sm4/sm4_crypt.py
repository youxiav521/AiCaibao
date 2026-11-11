from typing import Union, Optional
from loguru import logger

from .base import SM4Base
from .mode import CryptMode
from .validation import SM4Validation


class SM4:
    """SM4加密工具类"""

    @staticmethod
    def encrypt(
        crypt_mode: CryptMode,
        data: Union[str, bytes],
        key: Optional[Union[str, bytes]] = None,
        iv: Optional[bytes] = None,
        padding: bool = False,
    ) -> bytes:
        """
        SM4加密

        Args:
            crypt_mode: 加密模式
            data: 待加密数据，字符串或字节
            key: 密钥(16字节)，None则自动生成
            iv: 初始化向量(16字节)，除ECB外其他模式必需

        Returns:
            加密后的数据
        """
        logger.debug(f"原始数据: {data.hex() if isinstance(data, bytes) else data}")
        encode_data = SM4Validation.prepare_data(data)
        key = SM4Validation.validate_key(key)
        SM4Validation.validate_iv(iv, crypt_mode, "加密")

        return SM4Base(crypt_mode, key, iv).encrypt(encode_data, padding)

    @staticmethod
    def decrypt(
        crypt_mode: CryptMode,
        data: Union[str, bytes],
        key: Optional[bytes] = None,
        iv: Optional[bytes] = None,
        padding: bool = False,
    ) -> bytes:
        """
        SM4解密

        Args:
            crypt_mode: 解密模式
            data: 待解密数据
            key: 密钥(16字节)
            iv: 初始化向量(16字节)，除ECB外其他模式必需

        Returns:
            解密后的数据
        """
        logger.debug(f"待解密数据: {data.hex() if isinstance(data, bytes) else data}")
        key = SM4Validation.validate_key(key)
        SM4Validation.validate_iv(iv, crypt_mode, "解密")

        return SM4Base(crypt_mode, key, iv).decrypt(data, padding)

    @staticmethod
    def encrypt_wrap(data: Union[str, bytes]) -> str:
        """
        SM4 WRAP加密
        """
        key = "9ef904449bea2d731b6365a670f5d89a"
        encrypt_data = SM4.encrypt(crypt_mode=CryptMode.WRAP, data=data, key=key)
        return encrypt_data.hex()

    @staticmethod
    def decrypt_wrap(data: Union[str, bytes]) -> str:
        """
        SM4 WRAP解密
        """
        key = "9ef904449bea2d731b6365a670f5d89a"
        decrypt_data = SM4.decrypt(crypt_mode=CryptMode.WRAP, data=bytes.fromhex(data), key=key)
        return decrypt_data.decode("utf-8")


def test_sm4():
    """测试函数"""
    try:
        key = "9ef904449bea2d731b6365a670f5d89a"

        plaintext_wrap = "minioadmin".encode("utf-8")
        ciphertext_wrap = SM4.encrypt(CryptMode.WRAP, plaintext_wrap, key)
        ciphertext_wrap = bytes.fromhex("1157954e1de97c854a61fe2a646bd5b9")
        _ = SM4.decrypt(CryptMode.WRAP, ciphertext_wrap, key)

    except Exception as e:
        logger.error(f"加解密错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_sm4()