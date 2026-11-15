"""Shared local text decoding for the CLI and the window."""

from pathlib import Path


def decode_text(data):
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            return data.decode('gb18030')
        except UnicodeDecodeError as error:
            raise ValueError('无法按 UTF-8 或 GB18030 读取文字。') from error


def read_text(path):
    return decode_text(Path(path).read_bytes())
