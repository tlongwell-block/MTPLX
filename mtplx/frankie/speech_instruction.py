"""Optional native Breeze delivery instruction; never part of spoken text."""


def validate_speech_instruction(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("speech_instruction must be a nonempty string or null.")
    if len(value.encode("utf-8")) > 512:
        raise ValueError("speech_instruction must be at most 512 UTF-8 bytes.")
    if "<ins_bos>" in value or "<ins_eos>" in value:
        raise ValueError("speech_instruction cannot contain native instruction control tokens.")
    return value
