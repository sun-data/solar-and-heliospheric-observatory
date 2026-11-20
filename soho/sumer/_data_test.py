import pathlib
import pytest
import requests
import numpy as np
import named_arrays as na
import soho
import soho.sumer._data
from soho.sumer._data import (
    _get,
    _url,
    _replace,
    _signature_gzip,
)

_name_test = "sum_20050415_153008_l1.tgz"
"""A small level-1 file, with a window on O V 629.7 angstroms."""


def _response(
    url: str,
    status: int,
    content: bytes = b"",
    headers: None | dict[str, str] = None,
) -> requests.Response:
    """A response from a server, made without connecting to one."""
    response = requests.Response()
    response.url = url
    response.status_code = status
    response.reason = "test"
    response._content = content
    response._content_consumed = True
    if headers is not None:
        response.headers.update(headers)
    return response


def test_url():
    url = _url(_name_test)
    assert url.endswith(f"/L1/2005/{_name_test}")


@pytest.mark.parametrize(
    argnames="time_start,time_stop,expected",
    argvalues=[
        ("2005-04-15T15:31:00", "2005-04-15T15:32:00", [_name_test]),
        ("1996-09-16T16:07:00", "1996-09-16T16:08:00", ["sum_19960916_160201_l1.tgz"]),
    ],
)
def test_urls(
    time_start: str,
    time_stop: str,
    expected: list[str],
):
    result = soho.sumer.urls(time_start, time_stop)
    assert isinstance(result, na.ScalarArray)
    assert result.axes == ("time",)
    names = [str(url).split("/")[-1] for url in result.ndarray]
    for name in expected:
        assert name in names


def test_download(tmp_path: pathlib.Path):
    urls = na.ScalarArray(np.array([_url(_name_test)]), axes="time")
    result = soho.sumer.download(urls, directory=tmp_path)
    assert result.axes == urls.axes
    path = pathlib.Path(str(result.ndarray[0]))
    assert path.is_file()
    assert path.read_bytes()[: len(_signature_gzip)] == _signature_gzip
    assert not list(path.parent.glob("*.part"))

    time_modified = path.stat().st_mtime_ns
    soho.sumer.download(urls, directory=tmp_path)
    assert path.stat().st_mtime_ns == time_modified


def test_download_short(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def get(url: str, num_retry: int = 5, **kwargs) -> requests.Response:
        """A response which ends before the length it promised."""
        response = _response(
            url,
            200,
            content=_signature_gzip + b"abc",
            headers={"Content-Length": "100"},
        )
        return response

    monkeypatch.setattr(soho.sumer._data, "_get", get)
    urls = na.ScalarArray(np.array([_url(_name_test)]), axes="time")
    with pytest.raises(ValueError, match="Got 5 of the 100 bytes"):
        soho.sumer.download(urls, directory=tmp_path)
    assert not list(tmp_path.rglob("*.tgz*"))


def test_download_not_gzip(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def get(url: str, num_retry: int = 5, **kwargs) -> requests.Response:
        """A response holding a page of HTML instead of a file."""
        return _response(url, 200, content=b"<html></html>")

    monkeypatch.setattr(soho.sumer._data, "_get", get)
    urls = na.ScalarArray(np.array([_url(_name_test)]), axes="time")
    with pytest.raises(ValueError, match="is not a gzip file"):
        soho.sumer.download(urls, directory=tmp_path)


@pytest.mark.parametrize(
    argnames="status,error",
    argvalues=[
        (404, FileNotFoundError),
        (403, requests.exceptions.HTTPError),
        (503, ConnectionError),
    ],
)
def test_get_error(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error: type[Exception],
):
    def get(url: str, **kwargs) -> requests.Response:
        """A response with the given status."""
        return _response(url, status)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(soho.sumer._data, "_delay_retry", 0)
    with pytest.raises(error):
        _get("https://example.com/file", num_retry=2)


def test_get_retry(monkeypatch: pytest.MonkeyPatch):
    statuses = [503, 200]

    def get(url: str, **kwargs) -> requests.Response:
        """A server which is busy the first time it is asked."""
        return _response(url, statuses.pop(0), content=b"ok")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(soho.sumer._data, "_delay_retry", 0)
    assert _get("https://example.com/file").content == b"ok"


def test_replace(tmp_path: pathlib.Path):
    source = tmp_path / "a"
    destination = tmp_path / "b"
    source.write_text("new")
    destination.write_text("old")
    _replace(source, destination)
    assert destination.read_text() == "new"
    assert not source.exists()
