from gmssl import func
from typing import Union, Optional

from src.crypt.sm4.mode import CryptMode
from src.crypt.sm4.constants import SM4Constants


class SM4Validation:
    """参数验证工具类"""

    @staticmethod
    def validate_key(key: Optional[Union[str, bytes]]) -> bytes:
        """验证并处理密钥"""
        if key is None:
            key = bytes.fromhex(SM4Constants.KEY)
        elif isinstance(key, str):
            key = bytes.fromhex(key)

        if len(key) != SM4Constants.KEY_SIZE:
            raise ValueError(f"密钥必须为{SM4Constants.KEY_SIZE}字节")
        return key

    @staticmethod
    def validate_iv(
        iv: Optional[Union[str, bytes]], mode: CryptMode, operation: str
    ) -> Optional[bytes]:
        """验证IV参数"""
        if mode in [CryptMode.ECB, CryptMode.WRAP]:
            return iv

        if iv is None:
            raise ValueError(f"{operation}模式: {mode.value}, 向量[iv]不能为空")
        elif isinstance(iv, str):
            iv = bytes.fromhex(iv)

        if len(iv) != SM4Constants.IV_SIZE:
            raise ValueError(f"IV必须为{SM4Constants.IV_SIZE}字节")
        return iv

    @staticmethod
    def prepare_data(data: Union[str, bytes]) -> bytes:
        """准备待处理数据"""
        if isinstance(data, str):
            return data.encode("utf-8")
        return data
