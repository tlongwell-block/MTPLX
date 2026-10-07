"""The Frankie realtime demo as it runs live: one module per behavior.

Each module exposes an explicit ``install`` (or factory) that the launcher
calls in order. Nothing here loads weights at import time; every asset is
passed in by path.
"""
