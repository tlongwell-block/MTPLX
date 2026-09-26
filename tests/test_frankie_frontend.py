"""Exercise the actual AudioWorklet and page handlers without audio hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_playback_and_interaction_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the browser JavaScript regression tests")
    subprocess.run(
        [node, "--test", str(Path(__file__).with_name("frankie_frontend.test.mjs"))],
        check=True,
        timeout=30,
    )
