"""
Sentence boundary detection for streaming LLM output.
Splits text into complete sentences suitable for TTS.
"""

import re

# Chinese + English sentence endings
_SENTENCE_END = re.compile(r'(?<=[。！？；！？\n.!?;])')


class SentenceSplitter:
    def __init__(self):
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        """
        Feed new text, return any complete sentences detected.
        Incomplete trailing text is kept in buffer.
        """
        self.buffer += text
        segments = _SENTENCE_END.split(self.buffer)
        # segments are like ["a。", "b！", "c未完"] etc.
        # paired: even indices are text, odd indices are the delimiters
        # Actually re.split with lookbehind keeps the delimiter at end of each match.
        # So segments = ["第一句。", "第二句？", "第三句 unfinished"]

        complete = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if not seg:
                i += 1
                continue
            # Check if this segment ends with a sentence-ending character
            if re.search(r'[。！？！？\n.!?;]$', seg):
                complete.append(seg.strip())
                i += 1
            else:
                # This is the incomplete remainder
                self.buffer = seg
                break

        return complete

    def flush(self) -> list[str]:
        """Return any remaining text in buffer and reset."""
        remainder = self.buffer.strip()
        self.buffer = ""
        return [remainder] if remainder else []

    def has_remainder(self) -> bool:
        return len(self.buffer.strip()) > 0
