"""Text boundaries shared by speech and its per-phrase conditioning."""
import re

# Breeze renders each phrase as a whole utterance, so a phrase cut mid-sentence
# ends on a falling, finished contour. Phrases end only at sentence ends; a
# sentence that runs this long without one breaks at a clause, then anywhere.
CLAUSE_WORDS, MAX_WORDS = 40, 64


def phrase_complete(text, next_token, *, first, hungry=False):
    if not text:
        return False
    if next_token in {'<tool_call>', '<think>'}:
        return True
    if not next_token.startswith((' ', '\n')):
        return False
    count = len(text.split())
    if count >= MAX_WORDS:
        return True
    ending = text.rstrip('"\'”’)]}')
    # An initial in a name is not a sentence boundary.
    if re.search(r'(?:^|\s)[A-Z]\.$', ending):
        return False
    if re.search(r'[.!?]$', ending):
        # Short sentences ride together unless the mouth needs words now.
        return first or hungry or count >= 6
    return count >= CLAUSE_WORDS and bool(re.search(r'[;:,]$', ending))
