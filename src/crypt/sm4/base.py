from typing import Optional
from gmssl import sm4
from loguru import logger

from src.crypt.sm4.mode import CryptMode
from src.crypt.sm4.constants import SM4Constants
from src.crypt.sm4.validation import SM4Validation


class SM4Base:
    """SM4基础加密类"""

    def __init__(
        self,
        crypt_mode: CryptMode,
        key: Optional[bytes] = None,
        iv: Optional[bytes] = None,
    ) -> None:
        """
        初始化SM4加密器

        Args:
            crypt_mode: 加密模式
            key: 密钥，None则自动生成
            iv: 初始化向量
        """
        self.key = SM4Validation.validate_key(key)
        self.iv = SM4Validation.validate_iv(iv, crypt_mode, "加密")
        self.crypt_mode = crypt_mode
        self.crypt_sm4 = sm4.CryptSM4()

        logger.debug(f"密钥: {self.key.hex()}")
        if self.iv is not None:
            logger.debug(f"IV: {self.iv.hex()}")

    def encrypt(self, data: bytes, padding: bool) -> bytes:
        """加密数据"""
        self.crypt_sm4.set_key(self.key, sm4.SM4_ENCRYPT)
        return self._process_data(data, encrypt=True, padding=padding)

    def decrypt(self, ciphertext: bytes, padding: bool) -> bytes:
        """解密数据"""
        self.crypt_sm4.set_key(self.key, sm4.SM4_DECRYPT)
        return self._process_data(ciphertext, encrypt=False, padding=padding)

    def _process_data(self, data: bytes, encrypt: bool, padding: bool) -> bytes:
        """统一处理数据加解密"""
        mode_handlers = {
            CryptMode.ECB: lambda d: self.crypt_sm4.crypt_ecb(d),
            CryptMode.CBC: lambda d: self.crypt_sm4.crypt_cbc(self.iv, d),
            CryptMode.CFB: lambda d: self._crypt_cfb(d, encrypt),
            CryptMode.OFB: lambda d: self._crypt_ofb(d),
            CryptMode.CTR: lambda d: self._crypt_ctr(d),
            CryptMode.WRAP: lambda d: self._crypt_wrap(d, encrypt, padding),
        }

        handler = mode_handlers.get(self.crypt_mode)
        if handler is None:
            raise ValueError(
                f"不支持的{'加密' if encrypt else '解密'}模式: {self.crypt_mode.value}"
            )

        result = handler(data)

        if encrypt:
            logger.debug(f"{self.crypt_mode.value} 加密后的数据: {result.hex()}")
        else:
            logger.debug(f"{self.crypt_mode.value} 解密后的数据: {result.decode('utf-8')}")
        return result

    def _crypt_cfb(self, data: bytes, encrypt: bool) -> bytes:
        """CFB模式加解密"""
        result = bytearray()
        feedback = bytearray(self.iv)

        for i in range(0, len(data), SM4Constants.BLOCK_SIZE):
            encrypted_feedback = self.crypt_sm4.crypt_ecb(bytes(feedback))
            block = data[i : i + SM4Constants.BLOCK_SIZE]

            if encrypt:
                cipher_block = self._xor_blocks(block, encrypted_feedback)
                result.extend(cipher_block)
                feedback = cipher_block + feedback[len(cipher_block) :]
            else:
                plain_block = self._xor_blocks(block, encrypted_feedback)
                result.extend(plain_block)
                feedback = block + feedback[len(block) :]

            feedback = feedback[: SM4Constants.BLOCK_SIZE]

        return bytes(result)

    def _crypt_ofb(self, data: bytes) -> bytes:
        """OFB模式加解密（加解密过程相同）"""
        result = bytearray()
        feedback = bytearray(self.iv)

        for i in range(0, len(data), SM4Constants.BLOCK_SIZE):
            encrypted_feedback = self.crypt_sm4.crypt_ecb(bytes(feedback))
            block = data[i : i + SM4Constants.BLOCK_SIZE]

            processed_block = self._xor_blocks(block, encrypted_feedback)
            result.extend(processed_block)

            feedback = encrypted_feedback + feedback[len(encrypted_feedback) :]
            feedback = feedback[: SM4Constants.BLOCK_SIZE]

        return bytes(result)

    def _crypt_ctr(self, data: bytes) -> bytes:
        """CTR模式加解密（加解密过程相同）"""
        result = bytearray()
        counter = bytearray(self.iv)

        for i in range(0, len(data), SM4Constants.BLOCK_SIZE):
            encrypted_counter = self.crypt_sm4.crypt_ecb(bytes(counter))
            block = data[i : i + SM4Constants.BLOCK_SIZE]

            processed_block = self._xor_blocks(block, encrypted_counter)
            result.extend(processed_block)

            self._increment_counter(counter)

        return bytes(result)

    def _crypt_wrap(self, data: bytes, encrypt: bool, padding: bool) -> bytes:
        """WRAP模式（基于ECB的密钥包装）"""
        if encrypt:
            crypted_data = data
            if padding:
                """加密：添加PKCS7填充"""
                crypted_data = self._add_pkcs7_padding(data)
            return self.crypt_sm4.crypt_ecb(crypted_data)
        else:
            decrypted = self.crypt_sm4.crypt_ecb(data)
            if padding:
                """解密：去除PKCS7填充"""
                return self._remove_pkcs7_padding(decrypted)
            else:
                return decrypted

    @staticmethod
    def _xor_blocks(block1: bytes, block2: bytes) -> bytes:
        """对两个块进行异或操作"""
        return bytes(a ^ b for a, b in zip(block1, block2[: len(block1)]))

    @staticmethod
    def _increment_counter(counter: bytearray) -> None:
        """计数器递增（大端序）"""
        for i in range(len(counter) - 1, -1, -1):
            if counter[i] == 0xFF:
                counter[i] = 0x00
            else:
                counter[i] += 1
                break

    @staticmethod
    def _add_pkcs7_padding(data: bytes) -> bytes:
        """添加PKCS7填充"""
        padding_len = SM4Constants.BLOCK_SIZE - (len(data) % SM4Constants.BLOCK_SIZE)
        return data + bytes([padding_len] * padding_len)

    @staticmethod
    def _remove_pkcs7_padding(data: bytes) -> bytes:
        """去除PKCS7填充"""
        padding_len = data[-1]
        if 1 <= padding_len <= SM4Constants.BLOCK_SIZE and all(
            b == padding_len for b in data[-padding_len:]
        ):
            return data[:-padding_len]
        else:
            logger.warning("WRAP模式解密：未检测到有效填充，返回原始数据")
            return data
