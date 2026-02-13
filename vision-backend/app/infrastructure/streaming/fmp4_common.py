"""
Shared fMP4 (fragmented MP4) utilities.

Used by both raw live streaming and agent processed-frame streaming
to parse the init segment (ftyp + moov) from FFmpeg output.
Avoids duplicate MP4 box parsing logic.
"""

from typing import Tuple


def try_extract_init_segment(
    parse_buf: bytearray,
    init_accum: bytearray,
) -> Tuple[bool, bytes]:
    """
    Parse MP4 boxes from parse_buf until we have a full init segment (ftyp + moov).

    Call this each time you append new chunk bytes to parse_buf.
    Consumes parsed boxes from parse_buf. Appends ftyp/moov to init_accum.

    Returns:
        (ready, init_segment): ready is True when moov was found; init_segment
        is the bytes to send to new viewers (copy of init_accum), or b"" if not ready.
    """
    buf = parse_buf

    def read_u32(b: bytes) -> int:
        return int.from_bytes(b, "big", signed=False)

    while True:
        if len(buf) < 8:
            return False, b""

        size = read_u32(buf[0:4])
        box_type = bytes(buf[4:8])

        header_len = 8
        if size == 1:
            if len(buf) < 16:
                return False, b""
            size = int.from_bytes(buf[8:16], "big", signed=False)
            header_len = 16
        elif size == 0:
            return False, b""

        if size < header_len or size > 50_000_000:
            return False, b""

        if len(buf) < size:
            return False, b""

        box = bytes(buf[:size])
        del buf[:size]

        if box_type in (b"ftyp", b"moov"):
            init_accum.extend(box)
            if box_type == b"moov":
                return True, bytes(init_accum)
