from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path


class SecretStoreError(RuntimeError):
    pass


class LocalSecretStore:
    """Uses the process environment first and Windows DPAPI for local persistence."""

    def __init__(self, encrypted_path: str | Path):
        self.encrypted_path = Path(encrypted_path).expanduser().resolve()

    def configured(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY", "").strip()) or self.encrypted_path.is_file()

    def get(self) -> str:
        environment = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if environment:
            return environment
        if not self.encrypted_path.is_file():
            raise SecretStoreError("尚未配置 OpenRouter API Key")
        if os.name != "nt":
            raise SecretStoreError("当前系统请通过 OPENROUTER_API_KEY 环境变量配置密钥")
        encrypted = self.encrypted_path.read_bytes()
        try:
            return _unprotect(encrypted).decode("utf-8")
        except Exception as exc:
            raise SecretStoreError("无法解密本机 OpenRouter API Key") from exc

    def set(self, value: str) -> None:
        key = value.strip()
        if not key.startswith("sk-or-") or len(key) < 24:
            raise ValueError("OpenRouter API Key 格式无效")
        if os.name != "nt":
            raise SecretStoreError("当前系统不持久化密钥，请设置 OPENROUTER_API_KEY 环境变量")
        encrypted = _protect(key.encode("utf-8"))
        self.encrypted_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.encrypted_path.with_suffix(self.encrypted_path.suffix + ".tmp")
        temporary.write_bytes(encrypted)
        temporary.replace(self.encrypted_path)


class MemorySecretStore:
    """Test-only in-memory store; never selected by the application automatically."""

    def __init__(self, value: str = ""):
        self.value = value

    def configured(self) -> bool:
        return bool(self.value)

    def get(self) -> str:
        if not self.value:
            raise SecretStoreError("尚未配置 OpenRouter API Key")
        return self.value

    def set(self, value: str) -> None:
        self.value = value.strip()


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _to_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return blob, buffer


def _protect(data: bytes) -> bytes:
    source, source_buffer = _to_blob(data)
    destination = _DataBlob()
    protect = ctypes.windll.crypt32.CryptProtectData
    protect.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    protect.restype = wintypes.BOOL
    arguments = (
        ctypes.byref(source),
        "EvoAgent OpenRouter Key",
        None,
        None,
        None,
    )
    result = protect(*arguments, 0x01, ctypes.byref(destination))  # user scope
    if not result:
        # Some sandboxed desktop launches do not load the user DPAPI profile.
        # Machine-scope DPAPI still keeps the key encrypted at rest; access to
        # the ciphertext remains limited by the containing user directory ACL.
        result = protect(*arguments, 0x01 | 0x04, ctypes.byref(destination))
    del source_buffer
    if not result:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        _local_free(destination.pbData)


def _unprotect(data: bytes) -> bytes:
    source, source_buffer = _to_blob(data)
    destination = _DataBlob()
    unprotect = ctypes.windll.crypt32.CryptUnprotectData
    unprotect.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    unprotect.restype = wintypes.BOOL
    result = unprotect(
        ctypes.byref(source), None, None, None, None, 0x01, ctypes.byref(destination)
    )
    del source_buffer
    if not result:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        _local_free(destination.pbData)


def _local_free(pointer: ctypes.POINTER(ctypes.c_byte)) -> None:
    local_free = ctypes.windll.kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    local_free(ctypes.cast(pointer, ctypes.c_void_p))
