"""Text boundaries shared by speech and its per-phrase conditioning."""
import re


def phrase_complete(text, next_token, *, first):
    if not text:
        return False
    if next_token in {'<tool_call>', '<think>'}:
        return True
    if not next_token.startswith((' ', '\n')):
        return False
    count = len(text.split())
    if count >= (16 if first else 32):
        return True
    ending = text.rstrip('"\'”’)]}')
    # An initial in a name is not a sentence boundary.
    if re.search(r'(?:^|\s)[A-Z]\.$', ending):
        return False
    if re.search(r'[.!?]$', ending) and (first or count >= 6):
        return True
    # Later clauses can use the already playing speech as a head start.
    return first and count >= 4 and bool(re.search(r'[;:,]$', ending))
