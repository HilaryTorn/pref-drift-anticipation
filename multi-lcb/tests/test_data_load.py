import pytest
import os
from lcb_runner.benchmarks.code_generation import (
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
)
from tests.helpers import create_default_config


filter_threads = (
    "ignore:This process .*? is multi-threaded, use of fork:DeprecationWarning"
)


@pytest.mark.filterwarnings(filter_threads)
def test_load_data_v1():
    conf = create_default_config(debug=False)
    problems = load_code_generation_dataset_not_fast(conf)
    assert len(problems) > 0


@pytest.mark.filterwarnings(filter_threads)
def test_load_data_v2():
    conf = create_default_config(debug=False)
    problems = load_code_generation_dataset(conf)
    assert len(problems) > 0


@pytest.mark.filterwarnings(filter_threads)
def test_load_data_v3():
    conf = create_default_config(debug=True)
    problems = load_code_generation_dataset(conf)
    assert len(problems) > 0
