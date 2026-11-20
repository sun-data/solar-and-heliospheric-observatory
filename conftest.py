from typing import Iterator
import os
import pathlib
import pytest
import joblib
import numpy as np
import named_arrays as na
import soho
from soho.sumer._data import _url

name_test = "sum_20050415_153008_l1.tgz"
"""
A small level-1 file of SUMER (4 MB), whose window on O V 629.7 angstroms,
in the second order, is labelled with its place in the first order.
"""


@pytest.fixture(autouse=True, scope="session")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pathlib.Path]:
    """
    Download files and cache the answers of the catalogue somewhere other than
    the cache in the home directory of the user.

    The answers of the catalogue are cached in a temporary directory,
    so that every run of the tests asks the catalogue again.
    The files are downloaded into the directory named by the environment
    variable ``SOHO_TEST_DOWNLOADS``, if it is set,
    which the workflow of the tests keeps from one run to the next,
    since the files of the archive never change,
    and otherwise into the temporary directory.
    """
    directory = tmp_path_factory.mktemp("soho")
    downloads = os.environ.get("SOHO_TEST_DOWNLOADS")
    directory_downloads = directory if downloads is None else pathlib.Path(downloads)
    directory_downloads = directory_downloads.resolve()
    directory_downloads.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(soho, "directory_default", directory_downloads)
        monkeypatch.setattr(
            soho,
            "memory",
            joblib.Memory(location=directory / "joblib", verbose=0),
        )
        yield directory_downloads


@pytest.fixture(scope="session")
def path_test(cache: pathlib.Path) -> pathlib.Path:
    """The file :data:`name_test`, downloaded once for every test."""
    urls = na.ScalarArray(np.array([_url(name_test)]), axes="time")
    return pathlib.Path(str(soho.sumer.download(urls).ndarray[0]))
