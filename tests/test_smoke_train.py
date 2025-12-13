from pathlib import Path

import pytest

from slu import train


def test_train_placeholder_runs():
    # Skip if feature cache is absent (CI or clean checkout)
    if not Path("data/features").exists():
        pytest.skip("data/features not present; skipping smoke train")
    train.main()
