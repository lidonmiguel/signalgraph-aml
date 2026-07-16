from pathlib import Path

import pytest

from signalgraph_aml.benchmark import run_benchmark


def test_benchmark_requires_a_real_input_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_benchmark(tmp_path / "HI-Small_Trans.csv")
