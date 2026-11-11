from enum import Enum


class CryptMode(Enum):
    ECB = "ECB"
    CBC = "CBC"
    CFB = "CFB"
    OFB = "OFB"
    CTR = "CTR"
    WRAP = "WRAP"
