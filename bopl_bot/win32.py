"""Small Windows-only helpers (cursor clipping + async key state)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


def key_down(vk_code: int) -> bool:
    """Return True when the key is currently held."""
    # High bit set means key is currently down
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


def clip_cursor(left: int, top: int, right: int, bottom: int) -> None:
    r = RECT(left=left, top=top, right=right, bottom=bottom)
    user32.ClipCursor(ctypes.byref(r))


def release_cursor() -> None:
    user32.ClipCursor(None)
