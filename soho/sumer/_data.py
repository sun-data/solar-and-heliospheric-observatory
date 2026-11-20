import os
import csv
import time
import uuid
import typing
import pathlib
import functools
import concurrent.futures
import requests
import joblib
import numpy as np
import astropy.time
import named_arrays as na
import soho

__all__ = [
    "urls",
    "download",
]

_url_archive = (
    "https://sohodata.nascom.nasa.gov/archive/soho/private/data/processed/sumer/L1"
)
"""
The directory of the SOHO archive which holds the level-1 files of SUMER,
in one directory for each year.
"""

_url_tap = "https://ssa.esac.esa.int/ssa-sl-tap/tap/sync"
"""
The TAP service of ESA's SOHO Science Archive,
which holds the catalogue of the observations of SOHO.
"""

_num_workers = 4
"""
The number of files downloaded at once,
since most of the time of each one is spent waiting for the server.
"""

_delay_retry = 1
"""
The number of seconds to wait before trying a request again,
which is doubled before each further try.
"""

_status_retry = (408, 429)
"""
The statuses below 500 with which a server refuses a request that may
succeed if it is tried again.
"""

_errors_retry = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
"""The errors of a request which may not happen if it is tried again."""

_delay_replace = 0.1
"""
The number of seconds to wait before trying to move a downloaded file
into place again,
which is doubled before each further try.
"""

_signature_gzip = b"\x1f\x8b"
"""The first bytes of every gzip file, which every level-1 file of SUMER is."""


def _get(
    url: str,
    num_retry: int = 5,
    data: None | dict[str, str] = None,
    stream: bool = False,
) -> requests.Response:
    """
    Get a URL, or post to it, trying again if the connection fails,
    the server is busy or has an error, or the response is cut short.

    Each try waits twice as long as the one before it,
    starting from :data:`_delay_retry`.

    Parameters
    ----------
    url
        The URL to get.
    num_retry
        The number of times to try to connect to the server.
    data
        The form to post to the URL.
        If :obj:`None` (the default), the URL is got instead.
    stream
        Whether to leave the content of the response to be read
        a piece at a time, rather than reading it all at once.

    Raises
    ------
    FileNotFoundError
        If the server does not have the URL.
    requests.exceptions.HTTPError
        If the server refuses the request for a reason which trying again
        cannot change, such as a 403 status.
    ConnectionError
        If no attempt succeeded.
    """
    error = None
    for i in range(num_retry):
        if i > 0:
            time.sleep(_delay_retry * 2 ** (i - 1))
        try:
            if data is None:
                response = requests.get(url, timeout=60, stream=stream)
            else:
                response = requests.post(url, data=data, timeout=300)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = None if e.response is None else e.response.status_code
            if status == 404:
                raise FileNotFoundError(url) from e
            if status is not None and status < 500 and status not in _status_retry:
                raise
            error = e
        except _errors_retry as e:
            error = e
        else:
            return response

    raise ConnectionError(f"Could not get {url} in {num_retry} tries.") from error


def _url(name: str) -> str:
    """
    The URL at which the SOHO archive serves a level-1 file of SUMER.

    Parameters
    ----------
    name
        The name of the file, such as ``sum_19960916_160201_l1.tgz``,
        which begins with the date of its first exposure.
    """
    return f"{_url_archive}/{name[4:8]}/{name}"


def _catalog(
    time_start: str,
    time_stop: str,
    num_retry: int = 5,
) -> str:
    """
    Ask ESA's catalogue of SOHO for the level-1 files of SUMER whose
    observations overlap a time range.

    Parameters
    ----------
    time_start
        The start of the time range, in ISO format.
    time_stop
        The end of the time range, in ISO format.
    num_retry
        The number of times to try to connect to the server.

    Returns
    -------
        The answer of the catalogue, as comma-separated values with a header,
        with the name, start and end of each file.
    """
    adql = (
        "SELECT o.file_name, o.begin_date, o.end_date "
        "FROM soho.observation AS o "
        "JOIN soho.sumer_observation AS s ON o.observation_oid = s.observation_oid "
        "WHERE o.file_name LIKE 'sum%_l1.tgz' "
        f"AND o.begin_date < '{time_stop}' AND o.end_date > '{time_start}' "
        "ORDER BY o.begin_date"
    )
    data = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    return _get(_url_tap, num_retry, data=data).text


@functools.cache
def _catalog_cached(memory: joblib.Memory) -> typing.Callable[..., str]:
    """
    :func:`_catalog`, cached in the given cache.

    The cached function is made once for each cache,
    so that the cache records the code of the function it caches only once.

    Parameters
    ----------
    memory
        The cache to store the answers in.
    """
    return typing.cast(
        typing.Callable[..., str],
        memory.cache(_catalog, ignore=["num_retry"]),
    )


def urls(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    axis_time: str = "time",
    num_retry: int = 5,
) -> na.ScalarArray:
    """
    Find the level-1 files of SUMER whose observations overlap a time range,
    and return the URLs at which the SOHO archive serves them,
    in order of their first exposures.

    The files are found in ESA's catalogue of SOHO,
    and its answers are cached in :data:`soho.memory`.
    Each file holds the exposures of one or more observing sequences,
    each a gzipped tar archive of one FITS file for each exposure.

    Parameters
    ----------
    time_start
        The start of the time range.
    time_stop
        The end of the time range.
    axis_time
        The logical axis along which the URLs are arranged.
    num_retry
        The number of times to try to connect to the server.

    Examples
    --------

    Find the file which holds the quiet-Sun raster of O V observed on
    1996 September 16.

    .. jupyter-execute::

        import soho

        soho.sumer.urls(
            time_start="1996-09-16T16:07",
            time_stop="1996-09-16T16:08",
        )
    """
    start = astropy.time.Time(time_start).isot
    stop = astropy.time.Time(time_stop).isot

    text = _catalog_cached(soho.memory)(start, stop, num_retry)
    names = [row["file_name"] for row in csv.DictReader(text.splitlines())]
    names = list(dict.fromkeys(names))

    result = np.array([_url(name) for name in names], dtype=str)
    return na.ScalarArray(result, axes=axis_time)


def _replace(
    source: pathlib.Path,
    destination: pathlib.Path,
    num_retry: int = 5,
) -> None:
    """
    Move a file onto another, trying again if the move is not permitted.

    Windows does not move a file while another process has it open,
    as a virus scanner may have a new file for a moment.
    Each try waits twice as long as the one before it,
    starting from :data:`_delay_replace`.

    Parameters
    ----------
    source
        The file to move.
    destination
        Where to move the file, replacing any file there.
    num_retry
        The number of times to try to move the file.
    """
    for i in range(num_retry - 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            time.sleep(_delay_replace * 2**i)

    os.replace(source, destination)


def _path_local(
    url: str,
    directory: pathlib.Path,
) -> pathlib.Path:
    """
    Where a file on a server is placed under a directory,
    at the same path as on the server.

    Parameters
    ----------
    url
        The URL of the file.
    directory
        The directory to place the file in.

    Raises
    ------
    ValueError
        If the URL is not that of a file.
    """
    relative = "/".join(url.split("/")[3:])
    if not relative or relative.endswith("/"):
        raise ValueError(f"{url} is not the URL of a file.")
    return directory / relative


def _download_file(
    url: str,
    directory: pathlib.Path,
    overwrite: bool = False,
    num_retry: int = 5,
) -> pathlib.Path:
    """
    Download a file to a directory, unless it has been downloaded already,
    and check that it is whole and starts the way a gzip file does.

    The file is written next to its final place under a name no other
    download uses, and then moved, so an interrupted download never looks
    finished.

    Parameters
    ----------
    url
        The URL to download.
    directory
        The directory to place the file in,
        under the same path as on the server.
    overwrite
        Boolean flag controlling whether to download the file if it is
        already in `directory`.
    num_retry
        The number of times to try to connect to the server.

    Raises
    ------
    ValueError
        If the file is shorter than the server said it would be,
        or is not a gzip file.
    """
    path = _path_local(url, directory)

    if overwrite or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        try:
            with _get(url, num_retry, stream=True) as response:
                size = int(response.headers.get("Content-Length", -1))
                with open(part, "xb") as file:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        file.write(chunk)
            if size >= 0 and part.stat().st_size != size:
                raise ValueError(
                    f"Got {part.stat().st_size} of the {size} bytes of {url}."
                )
            with open(part, "rb") as file:
                start = file.read(len(_signature_gzip))
            if start != _signature_gzip:
                raise ValueError(f"{url} is not a gzip file, it starts with {start!r}.")
            try:
                _replace(part, path)
            except PermissionError:
                # Another process may have downloaded the file first and
                # still have it open, in which case its copy is kept,
                # unless a new copy was asked for.
                if overwrite or not path.is_file():
                    raise
        finally:
            part.unlink(missing_ok=True)

    return path


def download(
    urls: na.AbstractScalarArray,
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
    num_retry: int = 5,
) -> na.ScalarArray:
    """
    Download the given URLs to a directory,
    unless they have been downloaded already.

    The files are placed under `directory` with the same paths as on the
    server, and each one is checked against the length the server promised
    and checked to be a gzip file.

    Parameters
    ----------
    urls
        The URLs to download, such as those found by :func:`urls`.
    directory
        The directory to place the downloaded files in.
        If :obj:`None` (the default), :data:`soho.directory_default` is used.
    overwrite
        Boolean flag controlling whether to download files which are already
        in `directory`.
    num_retry
        The number of times to try to connect to the server.
    """
    root: pathlib.Path = soho.directory_default if directory is None else directory

    ndarray = np.asarray(urls.ndarray)

    def get(url: str) -> str:
        """Download one URL and return where it was placed."""
        return str(_download_file(url, root, overwrite, num_retry))

    # Each URL is downloaded once, even if it is given more than once.
    unique = list(dict.fromkeys(str(url) for url in ndarray.flat))

    with concurrent.futures.ThreadPoolExecutor(_num_workers) as executor:
        paths = dict(zip(unique, executor.map(get, unique)))

    paths = [paths[str(url)] for url in ndarray.flat]
    paths = np.array(paths, dtype=str).reshape(ndarray.shape)

    return na.ScalarArray(paths, axes=urls.axes)
