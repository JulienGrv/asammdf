""" common MDF file format module """

from __future__ import annotations

import bz2
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
import csv
from datetime import datetime, timezone
from enum import Enum
from functools import reduce
import gzip
from io import BufferedIOBase, BytesIO
import logging
import os
from pathlib import Path
import re
from shutil import copy, move
import sys
from tempfile import gettempdir, mkdtemp
from traceback import format_exc
from types import TracebackType
import typing
from typing import Any, BinaryIO, Optional, overload, Union
import warnings
from warnings import warn
import xml.etree.ElementTree as ET
import zipfile

from canmatrix import CanMatrix, Frame
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.api.extensions import ExtensionArray
from typing_extensions import Literal, TypedDict, Unpack

from . import tool
from .blocks import bus_logging_utils, mdf_v2, mdf_v3, mdf_v4, v2_v3_blocks, v4_blocks
from .blocks import v2_v3_constants as v23c
from .blocks import v4_constants as v4c
from .blocks.conversion_utils import from_dict
from .blocks.mdf_common import Group
from .blocks.options import FloatInterpolation, IntegerInterpolation
from .blocks.source_utils import Source
from .blocks.utils import (
    ChannelsDB,
    components,
    csv_bytearray2hex,
    csv_int2hex,
    downcast,
    FileLike,
    is_file_like,
    load_can_database,
    master_using_raster,
    matlab_compatible,
    MDF2_VERSIONS,
    MDF3_VERSIONS,
    MDF4_VERSIONS,
    MdfException,
    plausible_timestamps,
    randomized_string,
    SUPPORTED_VERSIONS,
    TERMINATED,
    UINT16_u,
    UINT64_u,
    UniqueDB,
    validate_version_argument,
)
from .blocks.v2_v3_blocks import ChannelConversion as ChannelConversionV3
from .blocks.v2_v3_blocks import ChannelExtension
from .blocks.v2_v3_blocks import HeaderBlock as HeaderV3
from .blocks.v4_blocks import ChannelConversion as ChannelConversionV4
from .blocks.v4_blocks import (
    EventBlock,
    FileHistory,
    FileIdentificationBlock,
    SourceInformation,
)
from .blocks.v4_blocks import HeaderBlock as HeaderV4
from .signal import Signal
from .types import (
    BusType,
    ChannelGroupType,
    ChannelsType,
    CompressionType,
    DbcFileType,
    EmptyChannelsType,
    FloatInterpolationModeType,
    IntInterpolationModeType,
    RasterType,
    StrOrBytesPathType,
    StrPathType,
    WritableBufferType,
)

try:
    import fsspec

    FSSPEF_AVAILABLE = True
except:
    FSSPEF_AVAILABLE = False

logger = logging.getLogger("asammdf")
LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


target_byte_order = "<=" if sys.byteorder == "little" else ">="


__all__ = ["MDF", "SUPPORTED_VERSIONS"]


class SearchMode(Enum):
    plain = "plain"
    regex = "regex"
    wildcard = "wildcard"


Version = Union[mdf_v4.Version, mdf_v3.Version]


def get_measurement_timestamp_and_version(mdf: FileLike | BinaryIO) -> tuple[datetime, Version]:
    id_block = FileIdentificationBlock(address=0, stream=mdf)

    version = id_block.mdf_version
    if version >= 400:
        header = HeaderV4
    else:
        header = HeaderV3

    header = header(address=64, stream=mdf)
    main_version, revision = divmod(version, 100)
    version = f"{main_version}.{revision}"

    return header.start_time, version


def get_temporary_filename(path: Path = Path("temporary.mf4"), dir: str | Path | None = None) -> Path:
    if not dir:
        folder = gettempdir()
    else:
        folder = dir
    mf4_path = path.with_suffix(".mf4")
    idx = 0
    while True:
        tmp_path = (Path(folder) / mf4_path.name).with_suffix(f".{idx}.mf4")
        if not tmp_path.exists():
            break
        else:
            idx += 1

    return tmp_path


class _MatlabKwargs(TypedDict, total=False):
    format: Literal["4", "5", "7.3"]
    do_compression: bool
    oned_as: Literal["column", "row"]


_ParquetCompression = Literal["gzip", "bz2", "brotli", "lz4", "zstd", "snappy", "none"]


class _ParquetKwargs(TypedDict, total=False):
    compression: _ParquetCompression


class _ExportKwargs(_MatlabKwargs, _ParquetKwargs, total=False):
    single_time_base: bool
    raster: float
    time_from_zero: bool
    use_display_names: bool
    empty_channels: Literal["skip"]
    reduce_memory_usage: bool
    time_as_date: bool
    ignore_value2text_conversions: bool
    raw: bool


class MDF:
    r"""Unified access to MDF v3 and v4 files. Underlying _mdf's attributes and
    methods are linked to the `MDF` object via *setattr*. This is done to expose
    them to the user code and for performance considerations.

    Parameters
    ----------
    name : string | BytesIO | zipfile.ZipFile | bz2.BZ2File | gzip.GzipFile
        mdf file name (if provided it must be a real file name), file-like object or
        compressed file opened as Python object

        .. versionchanged:: 6.2.0

            added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile

    version : string
        mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20',
        '3.30', '4.00', '4.10', '4.11', '4.20'); default '4.10'. This argument is
        only used for MDF objects created from scratch; for MDF objects created
        from a file the version is set to file version

    channels (None) : iterable
        channel names that will used for selective loading. This can dramatically
        improve the file loading time. Default None -> load all channels

        .. versionadded:: 6.1.0

        .. versionchanged:: 6.3.0 make the default None

    use_display_names (\*\*kwargs) : bool
        keyword only argument: for MDF4 files parse the XML channel comment to
        search for the display name; XML parsing is quite expensive so setting
        this to *False* can decrease the loading times very much; default
        *True*
    remove_source_from_channel_names (\*\*kwargs) : bool
        remove source from channel names ("Speed\XCP3" -> "Speed")
    copy_on_get (\*\*kwargs) : bool
        copy arrays in the get method; default *True*
    expand_zippedfile (\*\*kwargs) : bool
        only for bz2.BZ2File and gzip.GzipFile, load the file content into a
        BytesIO before parsing (avoids the huge performance penalty of doing
        random reads from the zipped file); default *True*
    raise_on_multiple_occurrences (\*\*kwargs) : bool
        raise exception when there are multiple channel occurrences in the file and
        the `get` call is ambiguous; default True

        .. versionadded:: 7.0.0

    temporary_folder (\*\*kwargs) : str | pathlib.Path
        folder to use for temporary files

        .. versionadded:: 7.0.0

    Examples
    --------
    >>> mdf = MDF(version='3.30') # new MDF object with version 3.30
    >>> mdf = MDF('path/to/file.mf4') # MDF loaded from file
    >>> mdf = MDF(BytesIO(data)) # MDF from file contents
    >>> mdf = MDF(zipfile.ZipFile('data.zip')) # MDF creating using the first valid MDF from archive
    >>> mdf = MDF(bz2.BZ2File('path/to/data.bz2', 'rb')) # MDF from bz2 object
    >>> mdf = MDF(gzip.GzipFile('path/to/data.gzip', 'rb')) # MDF from gzip object

    """

    def __init__(
        self,
        name: StrPathType | FileLike | None = None,
        version: Version = "4.10",
        channels: list[str] | None = None,
        **kwargs,
    ) -> None:
        self._mdf: mdf_v2.MDF2 | mdf_v3.MDF3 | mdf_v4.MDF4

        if "callback" in kwargs:
            kwargs["progress"] = kwargs["callback"]
            del kwargs["callback"]

        temporary_folder = kwargs.get("temporary_folder", None)
        if temporary_folder:
            try:
                os.makedirs(temporary_folder, exist_ok=True)
            except:
                kwargs["temporary_folder"] = None

        if name:
            if is_file_like(name):
                if isinstance(name, (BytesIO, BufferedIOBase)):
                    original_name = None
                    file_stream = name
                    do_close = False

                elif isinstance(name, bz2.BZ2File):
                    original_name = Path(name._fp.name)
                    tmp_name = get_temporary_filename(original_name, dir=temporary_folder)
                    tmp_name.write_bytes(name.read())
                    file_stream = open(tmp_name, "rb")
                    name = tmp_name

                    do_close = True

                elif isinstance(name, gzip.GzipFile):
                    original_name = Path(name.name)
                    tmp_name = get_temporary_filename(original_name, dir=temporary_folder)
                    tmp_name.write_bytes(name.read())
                    file_stream = open(tmp_name, "rb")
                    name = tmp_name

                    do_close = True

                elif FSSPEF_AVAILABLE and isinstance(name, fsspec.spec.AbstractBufferedFile):
                    original_name = "AzureFile"
                    file_stream = name
                    do_close = False

                else:
                    raise MdfException(f"{type(name)} is not supported as input for the MDF class")

            elif isinstance(name, zipfile.ZipFile):
                archive = name
                files = archive.namelist()

                for fname in files:
                    if Path(fname).suffix.lower() in (".mdf", ".dat", ".mf4"):
                        original_name = fname
                        break
                else:
                    raise Exception("invalid zipped MF4: no supported file found in the archive")

                name = get_temporary_filename(Path(original_name), dir=temporary_folder)

                tmpdir = mkdtemp()
                output = archive.extract(fname, tmpdir)
                move(output, name)

                file_stream = open(name, "rb")
                do_close = True

            else:
                name = original_name = Path(name)
                if not name.is_file() or not name.exists():
                    raise MdfException(f'File "{name}" does not exist')

                if original_name.suffix.lower() in (".mf4z", ".zip"):
                    name = get_temporary_filename(original_name, dir=temporary_folder)
                    with zipfile.ZipFile(original_name, allowZip64=True) as archive:
                        files = archive.namelist()
                        for fname in files:
                            if Path(fname).suffix.lower() in (".mdf", ".dat", ".mf4"):
                                break
                        else:
                            raise Exception("invalid zipped MF4: no supported file found in the archive")

                        tmpdir = mkdtemp()
                        output = archive.extract(fname, tmpdir)

                        move(output, name)

                file_stream = open(name, "rb")
                do_close = True

            file_stream.seek(0)
            magic_header = file_stream.read(8)

            if magic_header.strip() not in (b"MDF", b"UnFinMF"):
                if do_close:
                    file_stream.close()
                raise MdfException(f'"{name}" is not a valid ASAM MDF file: magic header is {magic_header}')

            file_stream.seek(8)
            version = file_stream.read(4).decode("ascii").strip(" \0")
            if not version:
                _, version = get_measurement_timestamp_and_version(file_stream)

            if do_close:
                file_stream.close()

            kwargs["original_name"] = original_name
            kwargs["__internal__"] = True

            if version in MDF3_VERSIONS:
                self._mdf = mdf_v3.MDF3(name, channels=channels, **kwargs)
            elif version in MDF4_VERSIONS:
                self._mdf = mdf_v4.MDF4(name, channels=channels, **kwargs)
            elif version in MDF2_VERSIONS:
                self._mdf = mdf_v2.MDF2(name, channels=channels, **kwargs)
            else:
                message = f'"{name}" is not a supported MDF file; "{version}" file version was found'
                raise MdfException(message)

        else:
            kwargs["original_name"] = None
            kwargs["__internal__"] = True
            version = validate_version_argument(version, mdf_v4.MDF4.default_version)
            if version in MDF2_VERSIONS:
                self._mdf = mdf_v2.MDF2(version=version, **kwargs)
            elif version in MDF3_VERSIONS:
                self._mdf = mdf_v3.MDF3(version=version, **kwargs)
            elif version in MDF4_VERSIONS:
                self._mdf = mdf_v4.MDF4(version=version, **kwargs)
            else:
                message = (
                    f'"{version}" is not a supported MDF file version; ' f"Supported versions are {SUPPORTED_VERSIONS}"
                )
                raise MdfException(message)

        # we need a backreference to the MDF object to avoid it being garbage
        # collected in code like this:
        # MDF(filename).convert('4.10')
        self._mdf._parent = self

    @property
    def attachments(self) -> list:
        return self._mdf.attachments

    @property
    def channels_db(self) -> ChannelsDB:
        return self._mdf.channels_db

    @property
    def events(self) -> list:
        return self._mdf.events

    @property
    def file_history(self) -> list[v4_blocks.FileHistory]:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("the attribute 'file_history' is only available for MDF4")
        return self._mdf.file_history

    @property
    def groups(self) -> list[Group]:
        return self._mdf.groups

    @property
    def header(self) -> v2_v3_blocks.HeaderBlock | v4_blocks.HeaderBlock:
        return self._mdf.header

    @property
    def masters_db(self) -> dict:
        return self._mdf.masters_db

    @property
    def name(self) -> Path:
        return self._mdf.name

    @property
    def original_name(self) -> Path:
        return self._mdf.original_name

    @original_name.setter
    def original_name(self, value: Path) -> None:
        self._mdf.original_name = value

    @property
    def password(self) -> str:
        return self._mdf.password

    @property
    def version(self) -> str:
        return self._mdf.version

    def __setattr__(self, item: str, value: Any) -> None:
        if item == "_mdf":
            super().__setattr__(item, value)
        else:
            if hasattr(self._mdf, item):
                warnings.warn(
                    "accessing an attribute from the underlying '_mdf' object might be deprecated in the future in favor of a public method on the MDF class. If no public method is available yet, please open a pull request to add it.",
                    category=PendingDeprecationWarning,
                    stacklevel=2,
                )
            setattr(self._mdf, item, value)

    def __getattr__(self, item: str) -> Any:
        if item != "_mdf":
            warnings.warn(
                "accessing an attribute from the underlying '_mdf' object might be deprecated in the future in favor of a public method on the MDF class. If no public method is available yet, please open a pull request to add it.",
                category=PendingDeprecationWarning,
                stacklevel=2,
            )
        return getattr(self._mdf, item)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._mdf)))

    def __enter__(self) -> MDF:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._mdf.close()
        except:
            print(format_exc())

    def __del__(self) -> None:
        try:
            self._mdf.close()
        except:
            pass

    def __lt__(self, other: MDF) -> bool:
        if self._mdf.header.start_time < other._mdf.header.start_time:
            return True
        elif self._mdf.header.start_time > other._mdf.header.start_time:
            return False
        else:
            t_min = []
            for i, group in enumerate(self._mdf.groups):
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in self._mdf.masters_db:
                    master_min = self._mdf.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        t_min.append(master_min[0])

            other_t_min = []
            for i, group in enumerate(other._mdf.groups):
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in other._mdf.masters_db:
                    master_min = other._mdf.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        other_t_min.append(master_min[0])

            if not t_min or not other_t_min:
                return True
            else:
                return min(t_min) < min(other_t_min)

    def _transfer_events(self, other: MDF) -> None:
        def get_scopes(event, events):
            if event.scopes:
                return event.scopes
            else:
                if event.parent is not None:
                    return get_scopes(events[event.parent], events)
                elif event.range_start is not None:
                    return get_scopes(events[event.range_start], events)
                else:
                    return event.scopes

        if isinstance(other._mdf, mdf_v4.MDF4):
            for event in other._mdf.events:
                if isinstance(self._mdf, mdf_v4.MDF4):
                    new_event = deepcopy(event)
                    event_valid = True
                    for i, ref in enumerate(new_event.scopes):
                        try:
                            dg_cntr, ch_cntr = ref
                            try:
                                self._mdf.groups[dg_cntr].channels[ch_cntr]
                            except:
                                event_valid = False
                        except TypeError:
                            dg_cntr = ref
                            try:
                                self._mdf.groups[dg_cntr]
                            except:
                                event_valid = False
                    # ignore attachments for now
                    for i in range(new_event.attachment_nr):
                        key = f"attachment_{i}_addr"
                        event[key] = 0
                    if event_valid:
                        self._mdf.events.append(new_event)
                else:
                    ev_type = event.event_type
                    ev_range = event.range_type
                    ev_base = event.sync_base
                    ev_factor = event.sync_factor

                    timestamp = ev_base * ev_factor

                    try:
                        comment = ET.fromstring(event.comment.replace(' xmlns="http://www.asam.net/mdf/v4"', ""))
                        pre = comment.find(".//pre_trigger_interval")
                        if pre is not None:
                            pre = float(pre.text)
                        else:
                            pre = 0.0
                        post = comment.find(".//post_trigger_interval")
                        if post is not None:
                            post = float(post.text)
                        else:
                            post = 0.0
                        comment = comment.find(".//TX")
                        if comment is not None:
                            comment = comment.text
                        else:
                            comment = ""

                    except:
                        pre = 0.0
                        post = 0.0
                        comment = event.comment

                    if comment:
                        comment += ": "

                    if ev_range == v4c.EVENT_RANGE_TYPE_BEGINNING:
                        comment += "Begin of "
                    elif ev_range == v4c.EVENT_RANGE_TYPE_END:
                        comment += "End of "
                    else:
                        comment += "Single point "

                    if ev_type == v4c.EVENT_TYPE_RECORDING:
                        comment += "recording"
                    elif ev_type == v4c.EVENT_TYPE_RECORDING_INTERRUPT:
                        comment += "recording interrupt"
                    elif ev_type == v4c.EVENT_TYPE_ACQUISITION_INTERRUPT:
                        comment += "acquisition interrupt"
                    elif ev_type == v4c.EVENT_TYPE_START_RECORDING_TRIGGER:
                        comment += "measurement start trigger"
                    elif ev_type == v4c.EVENT_TYPE_STOP_RECORDING_TRIGGER:
                        comment += "measurement stop trigger"
                    elif ev_type == v4c.EVENT_TYPE_TRIGGER:
                        comment += "trigger"
                    else:
                        comment += "marker"

                    scopes = get_scopes(event, other._mdf.events)
                    if scopes:
                        for i, ref in enumerate(scopes):
                            event_valid = True
                            try:
                                dg_cntr, ch_cntr = ref
                                try:
                                    (self._mdf.groups[dg_cntr])
                                except:
                                    event_valid = False
                            except TypeError:
                                dg_cntr = ref
                                try:
                                    (self._mdf.groups[dg_cntr])
                                except:
                                    event_valid = False
                            if event_valid:
                                self._mdf.add_trigger(
                                    dg_cntr,
                                    timestamp,
                                    pre_time=pre,
                                    post_time=post,
                                    comment=comment,
                                )
                    else:
                        for i, _ in enumerate(self._mdf.groups):
                            self._mdf.add_trigger(
                                i,
                                timestamp,
                                pre_time=pre,
                                post_time=post,
                                comment=comment,
                            )

        else:
            for trigger_info in other._mdf.iter_get_triggers():
                comment = trigger_info["comment"]
                timestamp = trigger_info["time"]
                group = trigger_info["group"]

                if isinstance(self._mdf, (mdf_v2.MDF2, mdf_v3.MDF3)):
                    self._mdf.add_trigger(
                        group,
                        timestamp,
                        pre_time=trigger_info["pre_time"],
                        post_time=trigger_info["post_time"],
                        comment=comment,
                    )
                else:
                    if timestamp:
                        ev_type = v4c.EVENT_TYPE_TRIGGER
                    else:
                        ev_type = v4c.EVENT_TYPE_START_RECORDING_TRIGGER
                    event = EventBlock(
                        event_type=ev_type,
                        sync_base=int(timestamp * 10**9),
                        sync_factor=10**-9,
                        scope_0_addr=0,
                    )
                    event.comment = comment
                    event.scopes.append(group)
                    self._mdf.events.append(event)

    def _transfer_header_data(self, other: MDF, message: str = "") -> None:
        self._mdf.header.author = other._mdf.header.author
        self._mdf.header.department = other._mdf.header.department
        self._mdf.header.project = other._mdf.header.project
        self._mdf.header.subject = other._mdf.header.subject
        self._mdf.header.comment = other._mdf.header.comment
        if isinstance(self._mdf, mdf_v4.MDF4) and message:
            fh = FileHistory()
            fh.comment = f"""<FHcomment>
    <TX>{message}</TX>
    <tool_id>{tool.__tool__}</tool_id>
    <tool_vendor>{tool.__vendor__}</tool_vendor>
    <tool_version>{tool.__version__}</tool_version>
</FHcomment>"""

            self._mdf.file_history = [fh]

    @staticmethod
    def _transfer_channel_group_data(sgroup: ChannelGroupType, ogroup: ChannelGroupType) -> None:
        if not (isinstance(sgroup, v4_blocks.ChannelGroup) and isinstance(ogroup, v4_blocks.ChannelGroup)):
            sgroup.comment = ogroup.comment
        else:
            sgroup.flags = ogroup.flags
            sgroup.path_separator = ogroup.path_separator
            sgroup.comment = ogroup.comment
            sgroup.acq_name = ogroup.acq_name
            acq_source = ogroup.acq_source
            if acq_source:
                sgroup.acq_source = acq_source.copy()

    def _transfer_metadata(self, other: MDF, message: str = "") -> None:
        self._transfer_events(other)
        self._transfer_header_data(other, message)

    def __contains__(self, channel: str) -> bool:
        """if *'channel name'* in *'mdf file'*"""
        return channel in self._mdf.channels_db

    def __iter__(self) -> Iterator[Signal]:
        """iterate over all the channels found in the file; master channels
        are skipped from iteration

        """
        yield from self.iter_channels()

    def configure(
        self,
        *,
        from_other: MDF | None = None,
        read_fragment_size: int | None = None,
        write_fragment_size: int | None = None,
        use_display_names: bool | None = None,
        single_bit_uint_as_bool: bool | None = None,
        integer_interpolation: IntInterpolationModeType | IntegerInterpolation | None = None,
        copy_on_get: bool | None = None,
        float_interpolation: FloatInterpolationModeType | FloatInterpolation | None = None,
        raise_on_multiple_occurrences: bool | None = None,
        temporary_folder: str | None = None,
        fill_0_for_missing_computation_channels: bool | None = None,
    ) -> None:
        """configure MDF parameters

        The default values for the options are the following:
        * read_fragment_size = 0
        * write_fragment_size = 4MB
        * use_display_names = False
        * single_bit_uint_as_bool = False
        * integer_interpolation = 0 (fill - use previous sample)
        * float_interpolation = 1 (linear interpolation)
        * copy_on_get = False
        * raise_on_multiple_occurrences = True
        * temporary_folder = ""
        * fill_0_for_missing_computation_channels = False

        Parameters
        ----------
        read_fragment_size : int
            size hint of split data blocks, default 8MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size
        write_fragment_size : int
            size hint of split data blocks, default 4MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size. Maximum size is 4MB to ensure
            compatibility with CANape
        use_display_names : bool
            search for display name in the Channel XML comment
        single_bit_uint_as_bool : bool
            return single bit channels are np.bool arrays
        integer_interpolation : int
            interpolation mode for integer channels:

                * 0 - repeat previous sample
                * 1 - use linear interpolation
                * 2 - hybrid interpolation: channels with integer data type (raw values) that have a
                  conversion that outputs float values will use linear interpolation, otherwise
                  the previous sample is used

                .. versionchanged:: 6.2.0
                    added hybrid mode interpolation

        copy_on_get : bool
            copy arrays in the get method

        float_interpolation : int
            interpolation mode for float channels:

                * 0 - repeat previous sample
                * 1 - use linear interpolation

                .. versionadded:: 6.2.0

        raise_on_multiple_occurrences : bool
            raise exception when there are multiple channel occurrences in the file and
            the `get` call is ambiguous; default True

            .. versionadded:: 6.2.0

        from_other : MDF
            copy configuration options from other MDF

            .. versionadded:: 6.2.0

        temporary_folder : str
            default folder for temporary files

            .. versionadded:: 7.0.0

        fill_0_for_missing_computation_channels : bool
            when a channel required by a computed channel is missing, then fill with 0 values.
            If false then the computation will fail and the computed channel will be marked as not existing.

            .. versionadded:: 7.1.0
        """

        if from_other is not None:
            self._mdf._read_fragment_size = from_other._mdf._read_fragment_size
            self._mdf._write_fragment_size = from_other._mdf._write_fragment_size
            self._mdf._use_display_names = from_other._mdf._use_display_names
            self._mdf._single_bit_uint_as_bool = from_other._mdf._single_bit_uint_as_bool
            self._mdf._integer_interpolation = from_other._mdf._integer_interpolation
            self._mdf.copy_on_get = from_other._mdf.copy_on_get
            self._mdf._float_interpolation = from_other._mdf._float_interpolation
            self._mdf._raise_on_multiple_occurrences = from_other._mdf._raise_on_multiple_occurrences

        if read_fragment_size is not None:
            self._mdf._read_fragment_size = int(read_fragment_size)

        if write_fragment_size is not None:
            self._mdf._write_fragment_size = min(int(write_fragment_size), 4 * 1024 * 1024)

        if use_display_names is not None:
            self._mdf._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._mdf._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

        if integer_interpolation is not None:
            self._mdf._integer_interpolation = IntegerInterpolation(integer_interpolation)

        if copy_on_get is not None:
            self._mdf.copy_on_get = copy_on_get

        if float_interpolation is not None:
            self._mdf._float_interpolation = FloatInterpolation(float_interpolation)

        if temporary_folder is not None:
            try:
                os.makedirs(temporary_folder, exist_ok=True)
                self._mdf.temporary_folder = temporary_folder
            except:
                self._mdf.temporary_folder = None

        if raise_on_multiple_occurrences is not None:
            self._mdf._raise_on_multiple_occurrences = bool(raise_on_multiple_occurrences)

    def load_data(
        self,
        group_nr: int,
        record_offset: int = 0,
        record_count: int | None = None,
        optimize_read: bool = False,
    ) -> Iterator[tuple[bytes, int, int, bytes | None]] | Iterator[tuple[bytes, int, int | None]]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf._load_data(
                self._mdf.groups[group_nr],
                record_offset=record_offset,
                record_count=record_count,
                optimize_read=optimize_read,
            )
        return self._mdf._load_data(self._mdf.groups[group_nr], record_offset=record_offset, record_count=record_count)

    @overload
    def append(
        self,
        signals: list[Signal] | Signal,
        acq_name: str | None = ...,
        acq_source: Source | None = ...,
        comment: str = ...,
        common_timebase: bool = ...,
        units: dict[str, str | bytes] | None = ...,
    ) -> int: ...

    @overload
    def append(
        self,
        signals: pd.DataFrame,
        acq_name: str | None = ...,
        acq_source: Source | None = ...,
        comment: str = ...,
        common_timebase: bool = ...,
        units: dict[str, str | bytes] | None = ...,
    ) -> None: ...

    def append(
        self,
        signals: list[Signal] | Signal | pd.DataFrame,
        acq_name: str | None = None,
        acq_source: Source | None = None,
        comment: str = "Python",
        common_timebase: bool = False,
        units: dict[str, str | bytes] | None = None,
    ) -> int | None:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.append(
                signals=signals,
                acq_name=acq_name,
                acq_source=acq_source,
                comment=comment,
                common_timebase=common_timebase,
                units=units,
            )
        return self._mdf.append(signals=signals, comment=comment, common_timebase=common_timebase, units=units)

    def attach(
        self,
        data: bytes,
        file_name: str | None = None,
        hash_sum: bytes | None = None,
        comment: str = "",
        compression: bool = True,
        mime: str = r"application/octet-stream",
        embedded: bool = True,
        password: str | bytes | None = None,
    ) -> int:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.attach(
                data,
                file_name=file_name,
                hash_sum=hash_sum,
                comment=comment,
                compression=compression,
                mime=mime,
                embedded=embedded,
                password=password,
            )
        raise MdfException("attach is only supported for MDF4 files")

    def close(self) -> None:
        return self._mdf.close()

    def extract_attachment(
        self,
        index: int | None = None,
        password: str | bytes | None = None,
    ) -> tuple[bytes, Path, bytes]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.extract_attachment(index=index, password=password)
        raise MdfException("extract_attachment is only supported for MDF4 files")

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: Literal[False] = ...,
        data: tuple[bytes, int, int | None] | tuple[bytes, int, int, bytes | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> Signal: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: Literal[True] = ...,
        data: tuple[bytes, int, int | None] | tuple[bytes, int, int, bytes | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[Any]]: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: bool = ...,
        data: tuple[bytes, int, int | None] | tuple[bytes, int, int, bytes | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> Signal | tuple[NDArray[Any], NDArray[Any]]: ...

    def get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: RasterType | None = None,
        samples_only: bool = False,
        data: tuple[bytes, int, int | None] | tuple[bytes, int, int, bytes | None] | None = None,
        raw: bool = False,
        ignore_invalidation_bits: bool = False,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_channel_validation: bool = False,
    ) -> Signal | tuple[NDArray[Any], NDArray[Any] | None]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.get(
                name=name,
                group=group,
                index=index,
                raster=raster,
                samples_only=samples_only,
                data=data,
                raw=raw,
                ignore_invalidation_bits=ignore_invalidation_bits,
                record_offset=record_offset,
                record_count=record_count,
                skip_channel_validation=skip_channel_validation,
            )
        return self._mdf.get(
            name=name,
            group=group,
            index=index,
            raster=raster,
            samples_only=samples_only,
            data=data,
            raw=raw,
            record_offset=record_offset,
            record_count=record_count,
            skip_channel_validation=skip_channel_validation,
        )

    def get_can_signal(
        self,
        name: str,
        database: CanMatrix | StrPathType | None = None,
        ignore_invalidation_bits: bool = False,
        data: bytes | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.get_can_signal(
                name=name,
                database=database,
                ignore_invalidation_bits=ignore_invalidation_bits,
                data=data,
                raw=raw,
                ignore_value2text_conversion=ignore_value2text_conversion,
            )
        raise MdfException("get_can_signal is only supported for MDF4 files")

    def get_lin_signal(
        self,
        name: str,
        database: CanMatrix | StrPathType | None = None,
        ignore_invalidation_bits: bool = False,
        data: bytes | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.get_lin_signal(
                name=name,
                database=database,
                ignore_invalidation_bits=ignore_invalidation_bits,
                data=data,
                raw=raw,
                ignore_value2text_conversion=ignore_value2text_conversion,
            )
        raise MdfException("get_lin_signal is only supported for MDF4 files")

    def get_bus_signal(
        self,
        bus: BusType,
        name: str,
        database: CanMatrix | StrPathType | None = None,
        ignore_invalidation_bits: bool = False,
        data: bytes | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.get_bus_signal(
                bus=bus,
                name=name,
                database=database,
                ignore_invalidation_bits=ignore_invalidation_bits,
                data=data,
                raw=raw,
                ignore_value2text_conversion=ignore_value2text_conversion,
            )
        raise MdfException("get_bus_signal is only supported for MDF4 files")

    @overload
    def convert(self, version: Version, progress: None = ...) -> MDF: ...

    @overload
    def convert(self, version: Version, progress: Callable[[int, int], None] | Any = ...) -> MDF | object: ...

    def convert(self, version: Version, progress: Callable[[int, int], None] | Any | None = None) -> MDF | object:
        """convert *MDF* to other version

        Parameters
        ----------
        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default '4.10'

        Returns
        -------
        out : MDF
            new *MDF* object

        """
        version = validate_version_argument(version, self._mdf.default_version)

        out = MDF(version=version, **self._mdf._kwargs)

        out.configure(from_other=self)

        out._mdf.header.start_time = self._mdf.header.start_time

        groups_nr = len(self._mdf.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    return TERMINATED

        cg_nr = None

        self.configure(copy_on_get=False)

        # walk through all groups and get all channels
        for i, virtual_group in enumerate(self._mdf.virtual_groups):
            selected_signals = self._mdf._yield_selected_signals(virtual_group, version=version)

            sigs = typing.cast(Optional[list[Signal]], next(selected_signals, None))
            if sigs:
                cg = self._mdf.groups[virtual_group].channel_group
                cg_nr = out._mdf.append(
                    sigs,
                    common_timebase=True,
                    comment=cg.comment,
                )
                MDF._transfer_channel_group_data(out._mdf.groups[cg_nr].channel_group, cg)
            else:
                continue

            selected_samples = typing.cast(
                Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
            )
            for signal_samples in selected_samples:
                out._mdf.extend(cg_nr, signal_samples)

                if progress and progress.stop:
                    return TERMINATED

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)
                    progress.signals.setMaximum.emit(groups_nr)

                    if progress.stop:
                        return TERMINATED

        out._transfer_metadata(self, message=f"Converted from {self._mdf.name}")
        self.configure(copy_on_get=True)

        return out

    @overload
    def cut(
        self,
        start: float | None = ...,
        stop: float | None = ...,
        whence: int = ...,
        version: Version | None = ...,
        include_ends: bool = ...,
        time_from_zero: bool = ...,
        progress: None = ...,
    ) -> MDF: ...

    @overload
    def cut(
        self,
        start: float | None = ...,
        stop: float | None = ...,
        whence: int = ...,
        version: Version | None = ...,
        include_ends: bool = ...,
        time_from_zero: bool = ...,
        progress: Callable[[int, int], None] | Any = ...,
    ) -> MDF | object: ...

    def cut(
        self,
        start: float | None = None,
        stop: float | None = None,
        whence: int = 0,
        version: Version | None = None,
        include_ends: bool = True,
        time_from_zero: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> MDF | object:
        """cut *MDF* file. *start* and *stop* limits are absolute values
        or values relative to the first timestamp depending on the *whence*
        argument.

        Parameters
        ----------
        start : float
            start time, default *None*. If *None* then the start of measurement
            is used
        stop : float
            stop time, default *None*. If *None* then the end of measurement is
            used
        whence : int
            how to search for the start and stop values

            * 0 : absolute
            * 1 : relative to first timestamp

        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', 4.20'); default *None* and in this
            case the original file version is used
        include_ends : bool
            include the *start* and *stop* timestamps after cutting the signal.
            If *start* and *stop* are found in the original timestamps, then
            the new samples will be computed using interpolation. Default *True*
        time_from_zero : bool
            start time stamps from 0s in the cut measurement

        Returns
        -------
        out : MDF
            new MDF object

        """

        if version is None:
            version = self._mdf.version
        else:
            version = validate_version_argument(version, self._mdf.default_version)

        out = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        integer_interpolation_mode = self._mdf._integer_interpolation
        float_interpolation_mode = self._mdf._float_interpolation
        out.configure(from_other=self)

        self.configure(copy_on_get=False)

        if whence == 1:
            timestamps = []
            for group in self._mdf.virtual_groups:
                master = self._mdf.get_master(group, record_offset=0, record_count=1)
                if master.size:
                    timestamps.append(master[0])

            if timestamps:
                first_timestamp = np.amin(timestamps)
            else:
                first_timestamp = 0

            if start is not None:
                start += first_timestamp
            if stop is not None:
                stop += first_timestamp

        if time_from_zero:
            if not start:
                raise ValueError("start must be provided when time_from_zero is True")
            delta = start
            t_epoch = self._mdf.header.start_time.timestamp() + delta
            out._mdf.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            delta = 0
            out._mdf.header.start_time = self._mdf.header.start_time

        groups_nr = len(self._mdf.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

        # walk through all groups and get all channels
        for i, (group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
            included_channels = self._mdf.included_channels(group_index)[group_index]
            if not included_channels:
                continue

            idx = 0
            signals = []
            for j, sigs in enumerate(self._mdf._yield_selected_signals(group_index, groups=included_channels)):
                if not sigs:
                    break
                if j == 0:
                    signals = typing.cast(list[Signal], sigs)
                    master = signals[0].timestamps
                else:
                    signal_samples = typing.cast(list[tuple[NDArray[Any], Optional[NDArray[Any]]]], sigs)
                    master = signal_samples[0][0]

                if not len(master):
                    continue

                needs_cutting = True

                # check if this fragmement is within the cut interval or
                # if the cut interval has ended
                if start is None and stop is None:
                    fragment_start = None
                    fragment_stop = None
                    start_index = 0
                    stop_index = len(master)
                    needs_cutting = False
                elif start is None:
                    fragment_start = None
                    start_index = 0
                    if master[0] > stop:
                        break
                    else:
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(master, fragment_stop, side="right")
                        if stop_index == len(master):
                            needs_cutting = False

                elif stop is None:
                    fragment_stop = None
                    if master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(master, fragment_start, side="left")
                        stop_index = len(master)
                        if start_index == 0:
                            needs_cutting = False
                else:
                    if master[0] > stop:
                        break
                    elif master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(master, fragment_start, side="left")
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(master, fragment_stop, side="right")
                        if start_index == 0 and stop_index == len(master):
                            needs_cutting = False

                # update the signal if this is not the first yield
                if j:
                    signal_samples = typing.cast(list[tuple[NDArray[Any], Optional[NDArray[Any]]]], sigs)
                    for signal, (samples, invalidation) in zip(signals, signal_samples[1:]):
                        signal.samples = samples
                        signal.timestamps = master
                        signal.invalidation_bits = invalidation

                if needs_cutting:
                    master = (
                        Signal(master, master, name="_")
                        .cut(
                            fragment_start,
                            fragment_stop,
                            include_ends,
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        .timestamps
                    )

                    if not len(master):
                        continue

                    signals = [
                        sig.cut(
                            master[0],
                            master[-1],
                            include_ends=include_ends,
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        for sig in signals
                    ]

                if time_from_zero:
                    master = master - delta
                    for sig in signals:
                        sig.timestamps = master

                if idx == 0:
                    if start:
                        start_ = f"{start}s"
                    else:
                        start_ = "start of measurement"
                    if stop:
                        stop_ = f"{stop}s"
                    else:
                        stop_ = "end of measurement"
                    cg = self._mdf.groups[group_index].channel_group
                    cg_nr = out._mdf.append(
                        signals,
                        common_timebase=True,
                        comment=cg.comment,
                    )
                    MDF._transfer_channel_group_data(out._mdf.groups[cg_nr].channel_group, cg)

                else:
                    signal_samples = [(sig.samples, sig.invalidation_bits) for sig in signals]
                    signal_samples.insert(0, (master, None))
                    out._mdf.extend(cg_nr, signal_samples)

                idx += 1

                if progress and progress.stop:
                    return TERMINATED

            # if the cut interval is not found in the measurement
            # then append a data group with 0 cycles
            if idx == 0 and signals:
                for sig in signals:
                    sig.samples = sig.samples[:0]
                    sig.timestamps = sig.timestamps[:0]
                    if sig.invalidation_bits is not None:
                        sig.invalidation_bits = sig.invalidation_bits[:0]

                if start:
                    start_ = f"{start}s"
                else:
                    start_ = "start of measurement"
                if stop:
                    stop_ = f"{stop}s"
                else:
                    stop_ = "end of measurement"
                cg = self._mdf.groups[group_index].channel_group
                cg_nr = out._mdf.append(
                    signals,
                    common_timebase=True,
                )
                MDF._transfer_channel_group_data(out._mdf.groups[cg_nr].channel_group, cg)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        print("return terminated")
                        return TERMINATED

        self.configure(copy_on_get=True)

        out._transfer_metadata(self, message=f"Cut from {start_} to {stop_}")

        return out

    def export(
        self,
        fmt: Literal["asc", "csv", "hdf5", "mat", "parquet"],
        filename: StrPathType | None = None,
        progress: Callable[[int, int], None] | Any | None = None,
        **kwargs: Unpack[_ExportKwargs],
    ) -> object | None:
        r"""export *MDF* to other formats. The *MDF* file name is used is
        available, else the *filename* argument must be provided.

        The *pandas* export option was removed. you should use the method
        *to_dataframe* instead.

        Parameters
        ----------
        fmt : string
            can be one of the following:

            * `csv` : CSV export that uses the "," delimiter. This option
              will generate a new csv file for each data group
              (<MDFNAME>_DataGroup_<cntr>.csv)

            * `hdf5` : HDF5 file output; each *MDF* data group is mapped to
              a *HDF5* group with the name 'DataGroup_<cntr>'
              (where <cntr> is the index)

            * `mat` : Matlab .mat version 4, 5 or 7.3 export. If
              *single_time_base==False* the channels will be renamed in the mat
              file to 'D<cntr>_<channel name>'. The channel group
              master will be renamed to 'DM<cntr>_<channel name>'
              ( *<cntr>* is the data group index starting from 0)

            * `parquet` : export to Apache parquet format
            * `asc`: Vector ASCII format for bus logging

                .. versionadded:: 7.3.3

        filename : string | pathlib.Path
            export file name

        \*\*kwargs

            * `single_time_base`: resample all channels to common time base,
              default *False*
            * `raster`: float time raster for resampling. Valid if
              *single_time_base* is *True*
            * `time_from_zero`: adjust time channel to start from 0
            * `use_display_names`: use display name instead of standard channel
              name, if available.
            * `empty_channels`: behaviour for channels without samples; the
              options are *skip* or *zeros*; default is *skip*
            * `format`: only valid for *mat* export; can be '4', '5' or '7.3',
              default is '5'
            * `oned_as`: only valid for *mat* export; can be 'row' or 'column'
            * `keep_arrays` : keep arrays and structure channels as well as the
              component channels. If *True* this can be very slow. If *False*
              only the component channels are saved, and their names will be
              prefixed with the parent channel.
            * `reduce_memory_usage` : bool
              reduce memory usage by converting all float columns to float32 and
              searching for minimum dtype that can reprezent the values found
              in integer columns; default *False*
            * `compression` : str
              compression to be used

              * for ``parquet`` : "GZIP", "SNAPPY" or "LZ4"
              * for ``hfd5`` : "gzip", "lzf" or "szip"
              * for ``mat`` : bool

              .. versionadded:: 8.1.0

                added LZ4 compression after changing to pyarrow

            * `time_as_date` (False) : bool
              export time as local timezone datetimee; only valid for CSV export

              .. versionadded:: 5.8.0

            * `ignore_value2text_conversions` (False) : bool
              valid only for the channels that have value to text conversions and
              if *raw=False*. If this is True then the raw numeric values will be
              used, and the conversion will not be applied.

              .. versionadded:: 5.8.0

            * raw (False) : bool
              export all channels using the raw values

              .. versionadded:: 6.0.0

            * delimiter (',') : str
              only valid for CSV: see cpython documentation for csv.Dialect.delimiter

              .. versionadded:: 6.2.0

            * doublequote (True) : bool
              only valid for CSV: see cpython documentation for csv.Dialect.doublequote

              .. versionadded:: 6.2.0

            * escapechar (None) : str
              only valid for CSV: see cpython documentation for csv.Dialect.escapechar

              .. versionadded:: 6.2.0

            * lineterminator ("\\r\\n") : str
              only valid for CSV: see cpython documentation for csv.Dialect.lineterminator

              .. versionadded:: 6.2.0

            * quotechar ('"') : str
              only valid for CSV: see cpython documentation for csv.Dialect.quotechar

              .. versionadded:: 6.2.0

            * quoting ("MINIMAL") : str
              only valid for CSV: see cpython documentation for csv.Dialect.quoting. Use the
              last part of the quoting constant name

              .. versionadded:: 6.2.0

            * add_units (False) : bool
              only valid for CSV: add the channel units on the second row of the CSV file

              .. versionadded:: 7.1.0


        """

        header_items = (
            "date",
            "time",
            "author_field",
            "department_field",
            "project_field",
            "subject_field",
        )

        if fmt != "pandas" and filename is None and self._mdf.name is None:
            message = "Must specify filename for export" "if MDF was created without a file name"
            logger.warning(message)
            return

        single_time_base = kwargs.get("single_time_base", False)
        raster = kwargs.get("raster", None)
        time_from_zero = kwargs.get("time_from_zero", True)
        use_display_names = kwargs.get("use_display_names", True)
        empty_channels = kwargs.get("empty_channels", "skip")
        format = kwargs.get("format", "5")
        oned_as = kwargs.get("oned_as", "row")
        reduce_memory_usage = kwargs.get("reduce_memory_usage", False)
        do_compression = kwargs.get("do_compression", False)
        compression = typing.cast(_ParquetCompression, kwargs.get("compression", "none").lower())
        time_as_date = kwargs.get("time_as_date", False)
        ignore_value2text_conversions = kwargs.get("ignore_value2text_conversions", False)
        raw = bool(kwargs.get("raw", False))

        if compression == "snappy":
            try:
                import snappy  # type: ignore[import-untyped] # noqa: F401
            except ImportError:
                logger.warning("snappy compressor is not installed; compression will be set to GZIP")
                compression = "gzip"

        filename = Path(filename) if filename else self._mdf.name

        if fmt == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

            except ImportError:
                logger.warning("pyarrow not found; export to parquet is unavailable")
                return None

        elif fmt == "hdf5":
            try:
                from h5py import File as HDF5  # type: ignore[import-untyped]
            except ImportError:
                logger.warning("h5py not found; export to HDF5 is unavailable")
                return None

        elif fmt == "mat":
            if format == "7.3":
                try:
                    from hdf5storage import savemat  # type: ignore[import-untyped]
                except ImportError:
                    logger.warning("hdf5storage not found; export to mat v7.3 is unavailable")
                    return None
            else:
                try:
                    from scipy.io import savemat
                except ImportError:
                    logger.warning("scipy not found; export to mat v4 and v5 is unavailable")
                    return None

        elif fmt not in ("csv", "asc"):
            raise MdfException(f"Export to {fmt} is not implemented")

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                if progress.stop:
                    return TERMINATED

        if fmt == "asc":
            return self._asc_export(filename.with_suffix(".asc"))

        if single_time_base or fmt == "parquet":
            df = self.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                use_display_names=use_display_names,
                empty_channels=empty_channels,
                reduce_memory_usage=reduce_memory_usage,
                ignore_value2text_conversions=ignore_value2text_conversions,
                raw=raw,
                numeric_1D_only=fmt == "parquet",
            )
            units: dict[Hashable, str] = {}
            comments: dict[Hashable, str] = {}
            used_names = UniqueDB()

            groups_nr = len(self._mdf.groups)
            if progress is not None:
                if callable(progress):
                    progress(0, groups_nr * 2)
                else:
                    progress.signals.setMaximum.emit(groups_nr * 2)

                    if progress.stop:
                        return TERMINATED

            for i, grp in enumerate(self._mdf.groups):
                if progress is not None and not callable(progress) and progress.stop:
                    return TERMINATED

                grp = typing.cast(
                    Union[
                        Group[v2_v3_blocks.DataGroup, v2_v3_blocks.ChannelGroup, v2_v3_blocks.Channel],
                        Group[v4_blocks.DataGroup, v4_blocks.ChannelGroup, v4_blocks.Channel],
                    ],
                    grp,
                )
                for ch in grp.channels:
                    if use_display_names:
                        channel_name = list(ch.display_names)[0] if ch.display_names else ch.name
                    else:
                        channel_name = ch.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if hasattr(ch, "unit"):
                        unit = ch.unit
                        if ch.conversion:
                            unit = unit or ch.conversion.unit
                    else:
                        unit = ""
                    comment = ch.comment

                    units[channel_name] = unit
                    comments[channel_name] = comment

                if progress is not None:
                    if callable(progress):
                        progress(i + 1, groups_nr * 2)
                    else:
                        progress.signals.setValue.emit(i + 1)

                        if progress.stop:
                            return TERMINATED

        if fmt == "hdf5":
            filename = filename.with_suffix(".hdf")

            if single_time_base:
                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if isinstance(self._mdf, (mdf_v2.MDF2, mdf_v3.MDF3)):
                        for item in header_items:
                            group.attrs[item] = self._mdf.header[item].replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    count = len(df.columns)

                    if progress is not None:
                        if callable(progress):
                            progress(0, count * 2)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(count * 2)

                            if progress.stop:
                                return TERMINATED

                    for i, channel in enumerate(df):
                        series = df[channel]
                        unit = units.get(channel, "")
                        comment = comments.get(channel, "")

                        if series.dtype.kind == "O":
                            if isinstance(series[0], np.ndarray):
                                arrays = typing.cast(Sequence[NDArray[Any]], series)
                                samples = np.vstack(arrays)
                            else:
                                continue

                        if compression:
                            dataset = group.create_dataset(channel, data=samples, compression=compression)
                        else:
                            dataset = group.create_dataset(channel, data=samples)
                        unit = unit.replace("\0", "")
                        if unit:
                            dataset.attrs["unit"] = unit
                        comment = comment.replace("\0", "")
                        if comment:
                            dataset.attrs["comment"] = comment

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, count * 2)
                            else:
                                progress.signals.setValue.emit(i + 1)

                                if progress.stop:
                                    return TERMINATED

            else:
                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if isinstance(self._mdf, (mdf_v2.MDF2, mdf_v3.MDF3)):
                        for item in header_items:
                            group.attrs[item] = self._mdf.header[item].replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    groups_nr = len(self._mdf.virtual_groups)

                    if progress is not None:
                        if callable(progress):
                            progress(0, groups_nr)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(groups_nr)

                            if progress.stop:
                                return TERMINATED

                    for i, (group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
                        included_channels = self._mdf.included_channels(group_index)[group_index]

                        if not included_channels:
                            continue

                        names = UniqueDB()
                        if progress is not None and not callable(progress) and progress.stop:
                            return TERMINATED

                        if len(virtual_group.groups) == 1:
                            comment = self._mdf.groups[virtual_group.groups[0]].channel_group.comment
                        else:
                            comment = "Virtual group i"

                        group_name = r"/" + f"ChannelGroup_{i}"
                        group = hdf.create_group(group_name)

                        group.attrs["comment"] = comment

                        master_index = self._mdf.masters_db.get(group_index, -1)

                        if master_index >= 0:
                            group.attrs["master"] = self._mdf.groups[group_index].channels[master_index].name
                            master = self._mdf.get(group.attrs["master"], group_index)
                            if reduce_memory_usage:
                                master.timestamps = downcast(master.timestamps)
                            if compression:
                                dataset = group.create_dataset(
                                    group.attrs["master"],
                                    data=master.timestamps,
                                    compression=compression,
                                )
                            else:
                                dataset = group.create_dataset(
                                    group.attrs["master"],
                                    data=master.timestamps,
                                    dtype=master.timestamps.dtype,
                                )
                            unit = master.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = master.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        channels = [
                            (None, gp_index, ch_index)
                            for gp_index, channel_indexes in included_channels.items()
                            for ch_index in channel_indexes
                        ]

                        if not channels:
                            continue

                        signals = self.select(channels, raw=raw)

                        for j, sig in enumerate(signals):
                            if use_display_names:
                                name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                name = sig.name
                            name = name.replace("\\", "_").replace("/", "_")
                            name = names.get_unique_name(name)
                            if reduce_memory_usage:
                                sig.samples = downcast(sig.samples)
                            if compression:
                                dataset = group.create_dataset(name, data=sig.samples, compression=compression)
                            else:
                                dataset = group.create_dataset(name, data=sig.samples, dtype=sig.samples.dtype)
                            unit = sig.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = sig.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, groups_nr)
                            else:
                                progress.signals.setValue.emit(i + 1)

                                if progress.stop:
                                    return TERMINATED

        elif fmt == "csv":
            fmtparams = {
                "delimiter": kwargs.get("delimiter", ",")[0],
                "doublequote": kwargs.get("doublequote", True),
                "lineterminator": kwargs.get("lineterminator", "\r\n"),
                "quotechar": kwargs.get("quotechar", '"')[0],
            }

            quoting = kwargs.get("quoting", "MINIMAL").upper()
            quoting = getattr(csv, f"QUOTE_{quoting}")

            fmtparams["quoting"] = quoting

            escapechar = kwargs.get("escapechar", '"')
            if escapechar is not None:
                escapechar = escapechar[0]

            fmtparams["escapechar"] = escapechar

            if single_time_base:
                filename = filename.with_suffix(".csv")
                message = f'Writing csv export to file "{filename}"'
                logger.info(message)

                if time_as_date:
                    index = (
                        pd.to_datetime(df.index + self._mdf.header.start_time.timestamp(), unit="s")
                        .tz_localize("UTC")
                        .tz_convert(LOCAL_TIMEZONE)
                        .astype(str)
                    )
                    df.index = index
                    df.index.name = "timestamps"

                    units["timestamps"] = ""
                else:
                    units["timestamps"] = "s"

                if hasattr(self, "can_logging_db") and self.can_logging_db:
                    dropped: dict[str, pd.Series[str]] = {}

                    for name_ in df.columns:
                        if name_.endswith("CAN_DataFrame.ID"):
                            dropped[name_] = pd.Series(
                                csv_int2hex(df[name_].astype(np.dtype("<u4")) & 0x1FFFFFFF),
                                index=df.index,
                            )

                        elif name_.endswith("CAN_DataFrame.DataBytes"):
                            dropped[name_] = pd.Series(csv_bytearray2hex(df[name_]), index=df.index)

                    df = df.drop(columns=list(dropped))
                    for name, s in dropped.items():
                        df[name] = s

                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile, **fmtparams)

                    names_row = [df.index.name, *df.columns]
                    writer.writerow(names_row)

                    if kwargs.get("add_units", False):
                        units_row = [units[name] for name in names_row]
                        writer.writerow(units_row)

                    for col in df:
                        if df[col].dtype.kind == "S":
                            for encoding, errors in (
                                ("utf-8", "strict"),
                                ("latin-1", "strict"),
                                ("utf-8", "replace"),
                                ("latin-1", "replace"),
                            ):
                                try:
                                    df[col] = df[col] = df[col].str.decode(encoding, errors)
                                    break
                                except:
                                    continue

                    vals: list[Iterable[Any]]
                    if reduce_memory_usage:
                        vals = [df.index, *(df[name] for name in df)]
                    else:
                        vals = [
                            df.index.to_list(),
                            *(df[name].to_list() for name in df),
                        ]
                    count = len(df.index)

                    if progress is not None:
                        if callable(progress):
                            progress(0, count)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(count)

                            if progress.stop:
                                return TERMINATED

                    for i, row in enumerate(zip(*vals)):
                        writer.writerow(row)

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, count)
                            else:
                                progress.signals.setValue.emit(i + 1)
                                if progress.stop:
                                    return TERMINATED

            else:
                add_units = kwargs.get("add_units", False)

                filename = filename.with_suffix(".csv")

                gp_count = len(self._mdf.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, gp_count)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(gp_count)

                        if progress.stop:
                            return TERMINATED

                for i, (group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
                    if progress is not None and not callable(progress) and progress.stop:
                        return TERMINATED

                    message = f"Exporting group {i+1} of {gp_count}"
                    logger.info(message)

                    if len(virtual_group.groups) == 1:
                        comment = self._mdf.groups[virtual_group.groups[0]].channel_group.comment
                    else:
                        comment = ""

                    if comment:
                        for char in '\n\t\r\b <>\\/:"?*|':
                            comment = comment.replace(char, "_")
                        group_csv_name = filename.parent / f"{filename.stem}.ChannelGroup_{i}_{comment}.csv"
                    else:
                        group_csv_name = filename.parent / f"{filename.stem}.ChannelGroup_{i}.csv"

                    df = self.get_group(
                        group_index,
                        raster=raster,
                        time_from_zero=time_from_zero,
                        use_display_names=use_display_names,
                        reduce_memory_usage=reduce_memory_usage,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        raw=raw,
                    )

                    if add_units:
                        units = {}
                        used_names = UniqueDB()

                        for gp_index, channel_indexes in self._mdf.included_channels(group_index)[group_index].items():
                            for ch_index in channel_indexes:
                                ch = self._mdf.groups[gp_index].channels[ch_index]

                                if use_display_names:
                                    channel_name = list(ch.display_names)[0] if ch.display_names else ch.name
                                else:
                                    channel_name = ch.name

                                channel_name = used_names.get_unique_name(channel_name)

                                if hasattr(ch, "unit"):
                                    unit = ch.unit
                                    if ch.conversion:
                                        unit = unit or ch.conversion.unit
                                else:
                                    unit = ""

                                units[channel_name] = unit
                    else:
                        units = {}

                    if time_as_date:
                        index = (
                            pd.to_datetime(df.index + self._mdf.header.start_time.timestamp(), unit="s")
                            .tz_localize("UTC")
                            .tz_convert(LOCAL_TIMEZONE)
                            .astype(str)
                        )
                        df.index = index
                        df.index.name = "timestamps"

                        units["timestamps"] = ""
                    else:
                        units["timestamps"] = "s"

                    with open(group_csv_name, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile, **fmtparams)

                        if hasattr(self, "can_logging_db") and self.can_logging_db:
                            dropped = {}

                            for name_ in df.columns:
                                if name_.endswith("CAN_DataFrame.ID"):
                                    dropped[name_] = pd.Series(
                                        csv_int2hex(df[name_] & 0x1FFFFFFF),
                                        index=df.index,
                                    )

                                elif name_.endswith("CAN_DataFrame.DataBytes"):
                                    dropped[name_] = pd.Series(csv_bytearray2hex(df[name_]), index=df.index)

                            df = df.drop(columns=list(dropped))
                            for name_, s in dropped.items():
                                df[name_] = s

                        names_row = [df.index.name, *df.columns]
                        writer.writerow(names_row)

                        if add_units:
                            units_row = [units[name] for name in names_row]
                            writer.writerow(units_row)

                        if reduce_memory_usage:
                            vals = [df.index, *(df[name] for name in df)]
                        else:
                            vals = [
                                df.index.to_list(),
                                *(df[name].to_list() for name in df),
                            ]

                        for i, row in enumerate(zip(*vals)):
                            writer.writerow(row)

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, gp_count)
                        else:
                            progress.signals.setValue.emit(i + 1)

                            if progress.stop:
                                return TERMINATED

        elif fmt == "mat":
            filename = filename.with_suffix(".mat")

            if not single_time_base:

                def decompose(samples):
                    dct = {}

                    for name in samples.dtype.names:
                        vals = samples[name]

                        if vals.dtype.names:
                            dct.update(decompose(vals))
                        else:
                            dct[name] = vals

                    return dct

                mdict: dict[str, ExtensionArray | NDArray[Any]] = {}

                master_name_template = "DGM{}_{}"
                channel_name_template = "DG{}_{}"
                used_names = UniqueDB()

                groups_nr = len(self._mdf.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr + 1)

                        if progress.stop:
                            return TERMINATED

                for i, (group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
                    if progress is not None and not callable(progress) and progress.stop:
                        return TERMINATED

                    included_channels = self._mdf.included_channels(group_index)[group_index]

                    if not included_channels:
                        continue

                    channels = [
                        (None, gp_index, ch_index)
                        for gp_index, channel_indexes in included_channels.items()
                        for ch_index in channel_indexes
                    ]

                    if not channels:
                        continue

                    signals = self.select(
                        channels,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        raw=raw,
                    )

                    master = signals[0].copy()
                    master.samples = master.timestamps

                    signals.insert(0, master)

                    for j, sig in enumerate(signals):
                        if j == 0:
                            channel_name = master_name_template.format(i, "timestamps")
                        else:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name
                            channel_name = channel_name_template.format(i, channel_name)

                        channel_name = matlab_compatible(channel_name)
                        channel_name = used_names.get_unique_name(channel_name)

                        if sig.samples.dtype.names:
                            sig.samples.dtype.names = tuple(matlab_compatible(name) for name in sig.samples.dtype.names)

                            sigs = decompose(sig.samples)

                            sigs = {
                                channel_name_template.format(i, channel_name): _v for channel_name, _v in sigs.items()
                            }

                            mdict.update(sigs)

                        else:
                            mdict[channel_name] = sig.samples

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(i + 1)

                            if progress.stop:
                                return TERMINATED

            else:
                used_names = UniqueDB()
                mdict = {}

                count = len(df.columns)

                if progress is not None:
                    if callable(progress):
                        progress(0, count)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(count)

                        if progress.stop:
                            return TERMINATED

                for i, name in enumerate(df.columns):
                    channel_name = matlab_compatible(name)
                    channel_name = used_names.get_unique_name(channel_name)

                    mdict[channel_name] = df[name].values

                    if hasattr(mdict[channel_name].dtype, "categories"):
                        mdict[channel_name] = np.array(mdict[channel_name], dtype="S")

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(i + 1)
                            progress.signals.setMaximum.emit(count)

                            if progress.stop:
                                return TERMINATED

                mdict["timestamps"] = df.index.values

            if progress is not None:
                if callable(progress):
                    progress(80, 100)
                else:
                    progress.signals.setValue.emit(0)
                    progress.signals.setMaximum.emit(100)
                    progress.signals.setValue.emit(80)

                    if progress.stop:
                        return TERMINATED

            if format == "7.3":
                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    format="7.3",
                    delete_unused_variables=False,
                    oned_as=oned_as,
                    structured_numpy_ndarray_as_struct=True,
                    store_python_metadata=False,
                )
            else:
                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    oned_as=oned_as,
                    do_compression=do_compression or compression not in ("", "none"),
                )

            if progress is not None:
                if callable(progress):
                    progress(100, 100)
                else:
                    progress.signals.setValue.emit(100)

                    if progress.stop:
                        return TERMINATED

        elif fmt == "parquet":
            filename = filename.with_suffix(".parquet")
            table = pa.table(df)
            if compression:
                pq.write_table(table, filename, compression=compression)
            else:
                pq.write_table(table, filename)

        else:
            message = 'Unsupported export type "{}". ' 'Please select "csv", "excel", "hdf5", "mat" or "pandas"'
            message.format(fmt)
            logger.warning(message)

        return None

    @overload
    def filter(self, channels: ChannelsType, version: Version | None = ..., progress: None = ...) -> MDF: ...

    @overload
    def filter(
        self, channels: ChannelsType, version: Version | None = ..., progress: Callable[[int, int], None] | Any = ...
    ) -> MDF | object: ...

    def filter(
        self,
        channels: ChannelsType,
        version: Version | None = None,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> MDF | object:
        """return new *MDF* object that contains only the channels listed in
        *channels* argument

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default *None* and in this
            case the original file version is used

        Returns
        -------
        mdf : MDF
            new *MDF* file

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> for i in range(4):
        ...     sigs = [Signal(s*(i*10+j), t, name='SIG') for j in range(1,4)]
        ...     mdf.append(sigs)
        ...
        >>> filtered = mdf.filter(['SIG', ('SIG', 3, 1), ['SIG', 2], (None, 1, 2)])
        >>> for gp_nr, ch_nr in filtered.channels_db['SIG']:
        ...     print(filtered.get(group=gp_nr, index=ch_nr))
        ...
        <Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">

        """
        if version is None:
            version = self._mdf.version
        else:
            version = validate_version_argument(version, self._mdf.default_version)

        _raise_on_multiple_occurrences = self._mdf._raise_on_multiple_occurrences
        self._mdf._raise_on_multiple_occurrences = False

        names_map: dict[tuple[int, int], str] = {}
        for item in channels:
            if isinstance(item, str):
                entry = self._mdf._validate_channel_selection(item)
                name = item
            else:
                if item[0]:
                    name = item[0]
                    if len(item) == 3:
                        _, group, index = item
                    else:
                        _, group = item
                        index = None
                    entry = self._mdf._validate_channel_selection(name, group, index)
                else:
                    continue
            names_map[entry] = name
        self._mdf._raise_on_multiple_occurrences = _raise_on_multiple_occurrences

        # group channels by group index
        gps = self._mdf.included_channels(channels=channels)

        mdf = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        mdf.configure(from_other=self)
        mdf._mdf.header.start_time = self._mdf.header.start_time

        self.configure(copy_on_get=False)

        groups_nr = len(gps)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    return TERMINATED

        for i, (group_index, groups) in enumerate(gps.items()):
            selected_signals = self._mdf._yield_selected_signals(group_index, groups=groups, version=version)

            sigs = typing.cast(Optional[list[Signal]], next(selected_signals, None))

            if not sigs:
                continue

            for sig in sigs:
                entry = sig.group_index, sig.channel_index
                if entry in names_map:
                    sig.name = names_map[entry]
            cg = self._mdf.groups[group_index].channel_group
            cg_nr = mdf.append(
                sigs,
                common_timebase=True,
                comment=cg.comment,
                acq_name=getattr(cg, "acq_name", None),
                acq_source=getattr(cg, "acq_source", None),
            )
            MDF._transfer_channel_group_data(mdf._mdf.groups[cg_nr].channel_group, cg)

            selected_samples = typing.cast(
                Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
            )

            for signal_samples in selected_samples:
                if not signal_samples:
                    break
                mdf._mdf.extend(cg_nr, signal_samples)

                if progress and progress.stop:
                    return TERMINATED

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        return TERMINATED

        self.configure(copy_on_get=True)

        mdf._transfer_metadata(self, message=f"Filtered from {self._mdf.name}")

        return mdf

    @overload
    def iter_get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: float | None = ...,
        samples_only: Literal[False] = ...,
        raw: bool = ...,
    ) -> Iterator[Signal]: ...

    @overload
    def iter_get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: float | None = ...,
        samples_only: Literal[True] = ...,
        raw: bool = ...,
    ) -> Iterator[tuple[NDArray[Any], NDArray[Any] | None]]: ...

    def iter_get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: float | None = None,
        samples_only: bool = False,
        raw: bool = False,
    ) -> Iterator[Signal | tuple[NDArray[Any], NDArray[Any] | None]]:
        """iterator over a channel

        This is usefull in case of large files with a small number of channels.

        If the *raster* keyword argument is not *None* the output is
        interpolated accordingly

        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index
        raster : float
            time raster in seconds
        samples_only : bool
            if *True* return only the channel samples as numpy array; if
                *False* return a *Signal* object
        raw : bool
            return channel samples without appling the conversion rule; default
            `False`

        """

        gp_nr, ch_nr = self._mdf._validate_channel_selection(name, group, index)

        data = self.load_data(gp_nr)

        for fragment in data:
            yield self.get(
                group=gp_nr,
                index=ch_nr,
                raster=raster,
                samples_only=samples_only,
                ignore_invalidation_bits=samples_only,
                data=fragment,
                raw=raw,
            )

    @staticmethod
    def concatenate(
        files: Sequence[StrPathType | MDF | FileLike],
        version: Version = "4.10",
        sync: bool = True,
        add_samples_origin: bool = False,
        direct_timestamp_continuation: bool = False,
        progress=None,
        **kwargs,
    ) -> MDF | object:
        """concatenates several files. The files
        must have the same internal structure (same number of groups, and same
        channels in each group).

        The order of the input files is always preserved, only the samples timestamps are influenced
        by the ``sync`` argument.

        Parameters
        ----------
        files : list | tuple
            list of *MDF* file names or *MDF*, zipfile.ZipFile, bz2.BZ2File or gzip.GzipFile
            instances

            ..versionchanged:: 6.2.0

                added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile

        version : str
            merged file version
        sync : bool
            sync the files based on the start of measurement, default *True*. The order of the
            input files is preserved, only the samples timestamps are influenced by this
            argument
        add_samples_origin : bool
            option to create a new "__samples_origin" channel that will hold
            the index of the measurement from where each timestamp originated
        direct_timestamp_continuation (False) : bool
            the time stamps from the next file will be added right after the last
            time stamp from the previous file; default False

            ..versionadded:: 6.0.0

        kwargs :

            use_display_names (False) : bool

        Examples
        --------
        >>> conc = MDF.concatenate(
            [
                'path/to/file.mf4',
                MDF(BytesIO(data)),
                MDF(zipfile.ZipFile('data.zip')),
                MDF(bz2.BZ2File('path/to/data.bz2', 'rb')),
                MDF(gzip.GzipFile('path/to/data.gzip', 'rb')),
            ],
            version='4.00',
            sync=False,
        )

        Returns
        -------
        concatenate : MDF
            new *MDF* object with concatenated channels

        Raises
        ------
        MdfException : if there are inconsistencies between the files

        """

        if not files:
            raise MdfException("No files given for merge")

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                progress.signals.setWindowTitle.emit("Concatenating measurements")

                if progress.stop:
                    return TERMINATED

        mdf_nr = len(files)
        use_display_names = kwargs.get("use_display_names", False)

        input_types = [isinstance(file, MDF) for file in files]

        versions: list[Version] = []
        if sync:
            datetimes: list[datetime] = []
            for file in files:
                if isinstance(file, MDF):
                    datetimes.append(file._mdf.header.start_time)
                    versions.append(file._mdf.version)
                else:
                    if is_file_like(file):
                        ts, version = get_measurement_timestamp_and_version(file)
                        datetimes.append(ts)
                        versions.append(version)
                    else:
                        with open(file, "rb") as bytes_io:
                            ts, version = get_measurement_timestamp_and_version(bytes_io)
                            datetimes.append(ts)
                            versions.append(version)

            try:
                oldest = min(datetimes)
            except TypeError:
                datetimes = [timestamp.astimezone(timezone.utc) for timestamp in datetimes]
                oldest = min(datetimes)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in datetimes]
            offsets = [max(0, offset) for offset in offsets]

        else:
            file = files[0]
            if isinstance(file, MDF):
                timestamp = file._mdf.header.start_time
                version = file._mdf.version
            else:
                if is_file_like(file):
                    timestamp, version = get_measurement_timestamp_and_version(file)
                else:
                    with open(file, "rb") as bytes_io:
                        timestamp, version = get_measurement_timestamp_and_version(bytes_io)

            oldest = timestamp
            versions.append(version)

            offsets = [0 for _ in files]

        included_channel_names: list[list[str]] = []
        cg_map = {}

        if add_samples_origin:
            origin_conversion: dict[str, Any] = {}
            for i, file in enumerate(files):
                origin_conversion[f"val_{i}"] = i
                origin_conversion[f"text_{i}"] = str(file._mdf.original_name if isinstance(file, MDF) else str(file))
            channel_conversion = from_dict(origin_conversion)

        for mdf_index, (offset, file) in enumerate(zip(offsets, files)):
            if not isinstance(file, MDF):
                mdf = MDF(file, use_display_names=use_display_names)
                close = True
            else:
                close = False

            if progress is not None and not callable(progress):
                progress.signals.setLabelText.emit(
                    f"Concatenating the file {mdf_index + 1} of {mdf_nr}\n{mdf._mdf.original_name}"
                )

            if mdf_index == 0:
                version = validate_version_argument(version, mdf._mdf.default_version)
                first_version = mdf._mdf.version

                kwargs = dict(mdf._mdf._kwargs)

                merged = MDF(
                    version=version,
                    **kwargs,
                )

                merged.configure(from_other=mdf)

                merged._mdf.header.start_time = oldest

            mdf.configure(copy_on_get=False)

            reorder_channel_groups = False
            cg_translations: dict[int, int] = {}

            vlsd_max_length = {}

            if mdf_index == 0:
                last_timestamps = [None for gp in mdf._mdf.virtual_groups]
                groups_nr = len(last_timestamps)
                first_mdf = mdf

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr * mdf_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr * mdf_nr)

                        if progress.stop:
                            return TERMINATED

                if isinstance(first_mdf._mdf, mdf_v4.MDF4):
                    w_mdf = first_mdf._mdf

                    vlds_channels = []

                    for _gp_idx, _gp in enumerate(w_mdf.groups):
                        for _ch_idx, _ch in enumerate(_gp.channels):
                            if _ch.channel_type == v4c.CHANNEL_TYPE_VLSD:
                                vlds_channels.append((_ch.name, _gp_idx, _ch_idx))

                                vlsd_max_length[(_ch.name, _gp_idx)] = 0

                    if vlsd_max_length:
                        for i, _file in enumerate(files):
                            if not isinstance(_file, MDF):
                                _close = True
                                _file = MDF(_file)
                            else:
                                _close = False

                            _file._mdf.determine_max_vlsd_sample_size.cache_clear()

                            for _ch_name, _gp_idx, _ch_idx in vlds_channels:
                                key = (_ch_name, _gp_idx)
                                for _second_gp_idx, _second_ch_idx in first_mdf.whereis(_ch_name):
                                    if _second_gp_idx == _gp_idx:
                                        vlsd_max_length[key] = max(
                                            vlsd_max_length[key],
                                            _file._mdf.determine_max_vlsd_sample_size(_second_gp_idx, _second_ch_idx),
                                        )
                                        break
                                else:
                                    raise MdfException(
                                        f"internal structure of file {i} is different; different channels"
                                    )

                            if _close:
                                _file._mdf.close()

            else:
                if len(mdf._mdf.virtual_groups) != groups_nr:
                    raise MdfException(
                        f"internal structure of file <{mdf._mdf.name}> is different; different channel groups count"
                    )
                else:
                    make_translation = False

                    # check if the order of the channel groups is the same
                    for i, group_index in enumerate(mdf._mdf.virtual_groups):
                        included_channels = mdf._mdf.included_channels(group_index)[group_index]
                        names = [
                            mdf._mdf.groups[gp_index].channels[ch_index].name
                            for gp_index, channels in included_channels.items()
                            for ch_index in channels
                        ]

                        if names != included_channel_names[i]:
                            if sorted(names) != sorted(included_channel_names[i]):
                                make_translation = reorder_channel_groups = True
                                break

                    # Make a channel group translation dictionary if the order is different
                    if (
                        make_translation
                        and isinstance(first_mdf._mdf, mdf_v4.MDF4)
                        and isinstance(mdf._mdf, mdf_v4.MDF4)
                    ):
                        for i, org_group in enumerate(first_mdf._mdf.groups):
                            org_group_source = org_group.channel_group.acq_source
                            for j, new_group in enumerate(mdf._mdf.groups):
                                new_group_source = new_group.channel_group.acq_source
                                if (
                                    new_group.channel_group.acq_name == org_group.channel_group.acq_name
                                    and (new_group_source and org_group_source)
                                    and new_group_source.name == org_group_source.name
                                    and new_group_source.path == org_group_source.path
                                    and new_group.channel_group.samples_byte_nr
                                    == org_group.channel_group.samples_byte_nr
                                ):
                                    new_included_channels = mdf._mdf.included_channels(j)[j]

                                    new_names = [
                                        mdf._mdf.groups[gp_index].channels[ch_index].name
                                        for gp_index, channels in new_included_channels.items()
                                        for ch_index in channels
                                    ]

                                    if sorted(new_names) == sorted(included_channel_names[i]):
                                        cg_translations[i] = j
                                        break

            for i, group_index in enumerate(mdf._mdf.virtual_groups):
                # save original group index for extension
                # replace with the translated group index
                if reorder_channel_groups:
                    origin_gp_idx = group_index
                    group_index = cg_translations[group_index]

                included_channels = mdf._mdf.included_channels(group_index)[group_index]

                if mdf_index == 0:
                    included_channel_names.append(
                        [
                            mdf._mdf.groups[gp_index].channels[ch_index].name
                            for gp_index, channels in included_channels.items()
                            for ch_index in channels
                        ]
                    )
                    different_channel_order = False
                else:
                    names = [
                        mdf._mdf.groups[gp_index].channels[ch_index].name
                        for gp_index, channels in included_channels.items()
                        for ch_index in channels
                    ]
                    different_channel_order = False
                    if names != included_channel_names[i]:
                        if sorted(names) != sorted(included_channel_names[i]):
                            raise MdfException(
                                f"internal structure of file {mdf_index} is different; different channels"
                            )
                        else:
                            original_names = included_channel_names[i]
                            different_channel_order = True
                            remap = [original_names.index(name) for name in names]

                if not included_channels:
                    continue

                last_timestamp = last_timestamps[i]
                first_timestamp = None
                original_first_timestamp = None

                mdf._mdf.vlsd_max_length.clear()
                mdf._mdf.vlsd_max_length.update(vlsd_max_length)

                selected_signals = mdf._mdf._yield_selected_signals(group_index, groups=included_channels)

                signals = typing.cast(Optional[list[Signal]], next(selected_signals, None))

                if not signals:
                    continue

                if mdf_index == 0:
                    first_signal = signals[0]
                    if len(first_signal):
                        if offset > 0:
                            timestamps = first_signal.timestamps + offset
                            for signal in signals:
                                signal.timestamps = timestamps
                        last_timestamp = first_signal.timestamps[-1]
                        first_timestamp = first_signal.timestamps[0]
                        original_first_timestamp = first_timestamp

                    if add_samples_origin:
                        signals.append(
                            Signal(
                                samples=np.ones(len(first_signal), dtype="<u2") * mdf_index,
                                timestamps=first_signal.timestamps,
                                conversion=channel_conversion,
                                name="__samples_origin",
                            )
                        )

                    cg = mdf._mdf.groups[group_index].channel_group
                    cg_nr = merged._mdf.append(
                        signals,
                        common_timebase=True,
                    )
                    MDF._transfer_channel_group_data(merged._mdf.groups[cg_nr].channel_group, cg)
                    cg_map[group_index] = cg_nr

                else:
                    if different_channel_order:
                        new_signals = signals.copy()
                        for new_index, signal in zip(remap, signals):
                            new_signals[new_index] = signal

                        signals = new_signals

                    signal_samples = [(signals[0].timestamps, None)] + [
                        (signal.samples, signal.invalidation_bits) for signal in signals
                    ]

                    master_signal = signal_samples[0][0]
                    _copied = False

                    if len(master_signal):
                        if original_first_timestamp is None:
                            original_first_timestamp = master_signal[0]
                        if offset > 0:
                            master = master_signal + offset
                            _copied = True
                        if last_timestamp is None:
                            last_timestamp = master_signal[-1]
                        else:
                            if last_timestamp >= master_signal[0] or direct_timestamp_continuation:
                                if len(master_signal) >= 2:
                                    delta = master_signal[1] - master_signal[0]
                                else:
                                    delta = 0.001
                                if _copied:
                                    master_signal -= master_signal[0]
                                else:
                                    master_signal = master_signal - master_signal[0]
                                    _copied = True
                                master_signal += last_timestamp + delta
                            last_timestamp = master_signal[-1]

                        signal_samples[0] = master_signal, None

                        if add_samples_origin:
                            signal_samples.append(
                                (
                                    np.ones(len(master_signal), dtype="<u2") * mdf_index,
                                    None,
                                )
                            )
                        cg_nr = cg_map[group_index]
                        # set the original channel group number back for extension
                        if reorder_channel_groups:
                            cg_nr = cg_map[origin_gp_idx]
                        merged._mdf.extend(cg_nr, signal_samples)

                        if first_timestamp is None:
                            first_timestamp = master_signal[0]

                selected_samples = typing.cast(
                    Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
                )

                for signal_samples in selected_samples:
                    if different_channel_order:
                        new_signal_samples = signal_samples.copy()
                        for new_index, sig in zip(remap, signal_samples[1:]):
                            new_signal_samples[new_index + 1] = sig
                        new_signal_samples[0] = signal_samples[0]

                        signal_samples = new_signal_samples

                    master = signal_samples[0][0]
                    _copied = False

                    if len(master):
                        if original_first_timestamp is None:
                            original_first_timestamp = master[0]
                        if offset > 0:
                            master = master + offset
                            _copied = True
                        if last_timestamp is None:
                            last_timestamp = master[-1]
                        else:
                            if last_timestamp >= master[0] or direct_timestamp_continuation:
                                if len(master) >= 2:
                                    delta = master[1] - master[0]
                                else:
                                    delta = 0.001
                                if _copied:
                                    master -= master[0]
                                else:
                                    master = master - master[0]
                                    _copied = True
                                master += last_timestamp + delta
                            last_timestamp = master[-1]

                        signal_samples[0] = master, None

                        if add_samples_origin:
                            signal_samples.append(
                                (
                                    np.ones(len(master), dtype="<u2") * mdf_index,
                                    None,
                                )
                            )
                        cg_nr = cg_map[group_index]
                        # set the original channel group number back for extension
                        if reorder_channel_groups:
                            cg_nr = cg_map[origin_gp_idx]
                        merged._mdf.extend(cg_nr, signal_samples)

                        if first_timestamp is None:
                            first_timestamp = master[0]

                    if progress and progress.stop:
                        return TERMINATED

                last_timestamps[i] = last_timestamp

            mdf.configure(copy_on_get=True)

            if mdf_index == 0:
                merged._transfer_metadata(mdf)

            if progress is not None:
                if callable(progress):
                    progress(i + 1 + mdf_index * groups_nr, mdf_nr * groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1 + mdf_index * groups_nr)

                    if progress.stop:
                        return TERMINATED

            if close and mdf_index:
                mdf._mdf.close()

        if isinstance(files[0], MDF):
            files[0]._mdf.close()

        if isinstance(merged._mdf, mdf_v4.MDF4):
            merged._mdf._process_bus_logging()

        return merged

    @staticmethod
    def stack(
        files: Sequence[StrPathType | MDF | FileLike],
        version: Version = "4.10",
        sync: bool = True,
        progress=None,
        **kwargs,
    ) -> MDF | object:
        """stack several files and return the stacked *MDF* object

        Parameters
        ----------
        files : list | tuple
            list of *MDF* file names or *MDF*, zipfile.ZipFile, bz2.BZ2File or gzip.GzipFile
            instances

            ..versionchanged:: 6.2.0

                added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile
        version : str
            merged file version
        sync : bool
            sync the files based on the start of measurement, default *True*

        kwargs :

            use_display_names (False) : bool

        Examples
        --------
        >>> stacked = MDF.stack(
            [
                'path/to/file.mf4',
                MDF(BytesIO(data)),
                MDF(zipfile.ZipFile('data.zip')),
                MDF(bz2.BZ2File('path/to/data.bz2', 'rb')),
                MDF(gzip.GzipFile('path/to/data.gzip', 'rb')),
            ],
            version='4.00',
            sync=False,
        )

        Returns
        -------
        stacked : MDF
            new *MDF* object with stacked channels

        """
        if not files:
            raise MdfException("No files given for stack")

        version = validate_version_argument(version, mdf_v4.MDF4.default_version)

        use_display_names = kwargs.get("use_display_names", False)

        files_nr = len(files)

        input_types = [isinstance(file, MDF) for file in files]

        if progress is not None:
            if callable(progress):
                progress(0, files_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(files_nr)

                if progress.stop:
                    return TERMINATED

        if sync:
            datetimes: list[datetime] = []
            for file in files:
                if isinstance(file, MDF):
                    datetimes.append(file._mdf.header.start_time)
                else:
                    if is_file_like(file):
                        ts, version = get_measurement_timestamp_and_version(file)
                        datetimes.append(ts)
                    else:
                        with open(file, "rb") as bytes_io:
                            ts, version = get_measurement_timestamp_and_version(bytes_io)
                            datetimes.append(ts)

            try:
                oldest = min(datetimes)
            except TypeError:
                datetimes = [timestamp.astimezone(timezone.utc) for timestamp in datetimes]
                oldest = min(datetimes)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in datetimes]

        else:
            offsets = [0 for file in files]

        for mdf_index, (offset, file) in enumerate(zip(offsets, files)):
            if not isinstance(file, MDF):
                mdf = MDF(file, use_display_names=use_display_names)
            else:
                mdf = file

            if progress is not None:
                progress.signals.setLabelText.emit(
                    f"Stacking file {mdf_index+1} of {files_nr}\n" f"{mdf._mdf.original_name.name}"
                )

            if mdf_index == 0:
                version = validate_version_argument(version, mdf._mdf.default_version)

                kwargs = dict(mdf._mdf._kwargs)

                stacked = MDF(
                    version=version,
                    **kwargs,
                )

                stacked.configure(from_other=mdf)

                if sync:
                    stacked._mdf.header.start_time = oldest
                else:
                    stacked._mdf.header.start_time = mdf._mdf.header.start_time

            mdf.configure(copy_on_get=False)

            for i, group in enumerate(mdf._mdf.virtual_groups):
                dg_cntr = None
                included_channels = mdf._mdf.included_channels(group)[group]
                if not included_channels:
                    continue

                selected_signals = mdf._mdf._yield_selected_signals(group, groups=included_channels, version=version)

                signals = typing.cast(Optional[list[Signal]], next(selected_signals, None))

                if not signals:
                    continue

                if sync:
                    timestamps = signals[0].timestamps + offset
                    for sig in signals:
                        sig.timestamps = timestamps
                cg = mdf._mdf.groups[group].channel_group
                dg_cntr = stacked._mdf.append(
                    signals,
                    common_timebase=True,
                )
                MDF._transfer_channel_group_data(stacked._mdf.groups[dg_cntr].channel_group, cg)

                if progress and progress.stop:
                    return TERMINATED

                selected_samples = typing.cast(
                    Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
                )

                for signal_samples in selected_samples:
                    master = signal_samples[0][0]
                    if sync:
                        master = master + offset
                        signal_samples[0] = master, None

                    stacked._mdf.extend(dg_cntr, signal_samples)

                if progress and progress.stop:
                    return TERMINATED

                for index in range(dg_cntr, len(stacked.groups)):
                    stacked.groups[index].channel_group.comment = (
                        f'stacked from channel group {i} of "{mdf.name.parent}"'
                    )

            if progress is not None:
                if callable(progress):
                    progress(mdf_index, files_nr)
                else:
                    progress.signals.setValue.emit(mdf_index)

                    if progress.stop:
                        return TERMINATED

            mdf.configure(copy_on_get=True)

            if mdf_index == 0:
                stacked._transfer_metadata(mdf)

            if not input_types[mdf_index]:
                mdf._mdf.close()

            if progress is not None and progress.stop:
                return TERMINATED

        if isinstance(stacked._mdf, mdf_v4.MDF4):
            stacked._mdf._process_bus_logging()

        return stacked

    def iter_channels(
        self,
        skip_master: bool = True,
        copy_master: bool = True,
        raw: bool | dict[str, bool] = False,
    ) -> Iterator[Signal]:
        """generator that yields a *Signal* for each non-master channel

        Parameters
        ----------
        skip_master : bool
            do not yield master channels; default *True*
        copy_master : bool
            copy master for each yielded channel *True*
        raw : bool
            return raw channels instead of converted; default *False*

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.


        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

        for index in self._mdf.virtual_groups:
            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self._mdf.included_channels(index)[index].items()
                for ch_index in channel_indexes
            ]

            signals = self.select(channels, copy_master=copy_master, raw=raw)

            yield from signals

    def iter_groups(
        self,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        keep_arrays: bool = False,
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        only_basenames: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """generator that yields channel groups as pandas DataFrames. If there
        are multiple occurrences for the same channel name inside a channel
        group, then a counter will be used to make the names unique
        (<original_name>_<counter>)


        Parameters
        ----------
        use_display_names : bool
            use display name instead of standard channel name, if available.

            .. versionadded:: 5.21.0

        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*

            .. versionadded:: 5.21.0

        raw (False) : bool
            the dataframe will contain the raw channel values

            .. versionadded:: 5.21.0

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.21.0

        keep_arrays (False) : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.

            .. versionadded:: 5.21.0

        empty_channels ("skip") : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*

            .. versionadded:: 5.21.0

        only_basenames (False) : bool
            use just the field names, without prefix, for structures and channel
            arrays

            .. versionadded:: 5.21.0

        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of using this argument

            .. versionadded:: 5.21.0

        """

        for i in self._mdf.virtual_groups:
            yield self.get_group(
                i,
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                only_basenames=only_basenames,
            )

    def resample(
        self,
        raster: RasterType,
        version: Version | None = None,
        time_from_zero: bool = False,
        progress=None,
    ) -> MDF | object:
        """resample all channels using the given raster. See *configure* to select
        the interpolation method for interger channels

        Parameters
        ----------
        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default *None* and
            in this case the original file version is used

        time_from_zero : bool
            start time stamps from 0s in the cut measurement

        Returns
        -------
        mdf : MDF
            new *MDF* with resampled channels

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> mdf = MDF()
        >>> sig = Signal(name='S1', samples=[1,2,3,4], timestamps=[1,2,3,4])
        >>> mdf.append(sig)
        >>> sig = Signal(name='S2', samples=[1,2,3,4], timestamps=[1.1, 3.5, 3.7, 3.9])
        >>> mdf.append(sig)
        >>> resampled = mdf.resample(raster=0.1)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
         2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        , <Signal S2:
                samples=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 3 4 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
         2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster='S2')
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[1.1 3.5 3.7 3.9]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        , <Signal S2:
                samples=[1 2 3 4]
                timestamps=[1.1 3.5 3.7 3.9]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster=[1.9, 2.0, 2.1])
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 2 2]
                timestamps=[1.9 2.  2.1]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        , <Signal S2:
                samples=[1 1 1]
                timestamps=[1.9 2.  2.1]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster='S2', time_from_zero=True)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[0.  2.4 2.6 2.8]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        , <Signal S2:
                samples=[1 2 3 4]
                timestamps=[0.  2.4 2.6 2.8]
                invalidation_bits=None
                unit=""
                conversion=None
                source=Source(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_names={}
                attachment=()>
        ]
        """

        if version is None:
            version = self._mdf.version
        else:
            version = validate_version_argument(version, self._mdf.default_version)

        mdf = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        integer_interpolation_mode = self._mdf._integer_interpolation
        float_interpolation_mode = self._mdf._float_interpolation
        mdf.configure(from_other=self)

        mdf._mdf.header.start_time = self._mdf.header.start_time

        groups_nr = len(self._mdf.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    return TERMINATED

        try:
            raster_float = float(raster)
            assert raster_float > 0
        except (TypeError, ValueError):
            if isinstance(raster, str):
                raster = self._mdf.get(raster).timestamps
            else:
                raster = np.array(raster)
        else:
            raster = master_using_raster(self._mdf, raster_float)

        if time_from_zero and len(raster):
            delta = raster[0]
            new_raster = raster - delta
            t_epoch = self._mdf.header.start_time.timestamp() + delta
            mdf._mdf.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            new_raster = None
            mdf._mdf.header.start_time = self._mdf.header.start_time

        for i, (group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self._mdf.included_channels(group_index)[group_index].items()
                for ch_index in channel_indexes
            ]

            if not channels:
                continue

            sigs = self.select(channels, raw=True)

            sigs = [
                sig.interp(
                    raster,
                    integer_interpolation_mode=integer_interpolation_mode,
                    float_interpolation_mode=float_interpolation_mode,
                )
                for sig in sigs
            ]

            if new_raster is not None:
                for sig in sigs:
                    if len(sig):
                        sig.timestamps = new_raster

            cg = self._mdf.groups[group_index].channel_group
            dg_cntr = mdf._mdf.append(
                sigs,
                common_timebase=True,
                comment=cg.comment,
            )
            MDF._transfer_channel_group_data(mdf._mdf.groups[dg_cntr].channel_group, cg)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        return TERMINATED

        mdf._transfer_metadata(self, message=f"Resampled from {self._mdf.name}")

        return mdf

    def select(
        self,
        channels: ChannelsType,
        record_offset: int = 0,
        raw: bool | dict[str, bool] = False,
        copy_master: bool = True,
        ignore_value2text_conversions: bool = False,
        record_count: int | None = None,
        validate: bool = False,
    ) -> list[Signal]:
        """retrieve the channels listed in *channels* argument as *Signal*
        objects

        .. note:: the *dataframe* argument was removed in version 5.8.0
                  use the ``to_dataframe`` method instead

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        record_offset : int
            record number offset; optimization to get the last part of signal samples
        raw : bool | dict[str, bool]
            get raw channel samples; default *False*

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.

        copy_master : bool
            option to get a new timestamps array for each selected Signal or to
            use a shared array for channels of the same channel group; default *True*
        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionchanged:: 5.8.0

        validate (False) : bool
            consider the invalidation bits

            .. versionadded:: 5.16.0

        Returns
        -------
        signals : list
            list of *Signal* objects based on the input channel list

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> for i in range(4):
        ...     sigs = [Signal(s*(i*10+j), t, name='SIG') for j in range(1,4)]
        ...     mdf.append(sigs)
        ...
        >>> # select SIG group 0 default index 1 default, SIG group 3 index 1, SIG group 2 index 1 default and channel index 2 from group 1
        ...
        >>> mdf.select(['SIG', ('SIG', 3, 1), ['SIG', 2],  (None, 1, 2)])
        [<Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        ]

        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
            raw_dict = raw
        else:
            raw_dict = None

        virtual_groups = self._mdf.included_channels(channels=channels, minimal=False, skip_master=False)

        output_signals = {}

        for virtual_group, groups in virtual_groups.items():
            cycles_nr = self._mdf.virtual_groups[virtual_group].cycles_nr
            pairs = [
                (gp_index, ch_index) for gp_index, channel_indexes in groups.items() for ch_index in channel_indexes
            ]

            if record_count is None:
                cycles = cycles_nr - record_offset
            else:
                if cycles_nr < record_count + record_offset:
                    cycles = cycles_nr - record_offset
                else:
                    cycles = record_count

            signals: list[Signal] = []

            current_pos = 0

            selected_signals = self._mdf._yield_selected_signals(
                virtual_group,
                groups=groups,
                record_offset=record_offset,
                record_count=record_count,
            )

            sigs = typing.cast(Optional[list[Signal]], next(selected_signals, None))

            if not sigs:
                continue

            next_pos = current_pos + len(sigs[0])

            master = np.empty(cycles, dtype=sigs[0].timestamps.dtype)
            master[current_pos:next_pos] = sigs[0].timestamps

            for sig in sigs:
                shape = (cycles,) + sig.samples.shape[1:]
                samples = np.empty(shape, dtype=sig.samples.dtype)
                samples[current_pos:next_pos] = sig.samples
                sig.samples = samples
                signals.append(sig)

                if sig.invalidation_bits is not None:
                    inval_samples = np.empty(cycles, dtype=sig.invalidation_bits.dtype)
                    inval_samples[current_pos:next_pos] = sig.invalidation_bits
                    sig.invalidation_bits = inval_samples

            current_pos = next_pos

            selected_samples = typing.cast(
                Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
            )

            for signal_samples in selected_samples:
                samples, _ = signal_samples[0]
                next_pos = current_pos + len(samples)
                master[current_pos:next_pos] = samples

                for signal, (samples, inval) in zip(signals, signal_samples[1:]):
                    signal.samples[current_pos:next_pos] = samples
                    if signal.invalidation_bits is not None:
                        signal.invalidation_bits[current_pos:next_pos] = inval

                current_pos = next_pos

            for signal, pair in zip(signals, pairs):
                signal.timestamps = master
                output_signals[pair] = signal

        indexes = []

        for item in channels:
            name: str | None
            if not isinstance(item, (list, tuple)):
                name = item
                group = index = None
            elif len(item) == 2:
                name, group = item
                index = None
            else:
                name, group, index = item
            indexes.append(self._mdf._validate_channel_selection(name=name, group=group, index=index))

        signals = [output_signals[pair] for pair in indexes]

        if copy_master:
            for signal in signals:
                signal.timestamps = signal.timestamps.copy()

        for signal in signals:
            if (raw_dict and not raw_dict.get(signal.name, __default__)) or (not raw_dict and not raw):
                conversion = signal.conversion
                if conversion:
                    samples = conversion.convert(
                        signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                    )
                    signal.samples = samples

                signal.raw = False
                signal.conversion = None
                if signal.samples.dtype.kind == "S":
                    signal.encoding = "utf-8" if isinstance(self._mdf, mdf_v4.MDF4) else "latin-1"

        if validate:
            signals = [sig.validate(copy=False) for sig in signals]

        for signal, channel in zip(signals, channels):
            if isinstance(channel, str):
                signal.name = channel
            else:
                name = channel[0]
                if name is not None:
                    signal.name = name

        unique = set()
        for i, signal in enumerate(signals):
            obj_id = id(signal)
            if id(signal) in unique:
                signals[i] = signal.copy()
            unique.add(obj_id)

        return signals

    @staticmethod
    def scramble(name: StrPathType, skip_attachments: bool = False, progress=None, **kwargs) -> Path | object:
        """scramble text blocks and keep original file structure

        Parameters
        ----------
        name : str | pathlib.Path
            file name
        skip_attachments : bool
            skip scrambling of attachments data if True

            .. versionadded:: 5.9.0

        Returns
        -------
        name : pathlib.Path
            scrambled file name

        """

        name = Path(name)

        mdf = MDF(name)

        if not mdf._mdf._file:
            raise RuntimeError(f"file was not openned '{name}'")

        texts: dict[int, bytes] = {}

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                if progress.stop:
                    return TERMINATED

        count = len(mdf._mdf.groups)

        if isinstance(mdf._mdf, mdf_v4.MDF4):
            try:
                stream = mdf._mdf._file

                if mdf._mdf.header.comment_addr:
                    stream.seek(mdf._mdf.header.comment_addr + 8)
                    size = UINT64_u(stream.read(8))[0] - 24
                    texts[mdf._mdf.header.comment_addr] = randomized_string(size)

                for fh in mdf.file_history:
                    addr = fh.comment_addr
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

                for ev in mdf._mdf.events:
                    for addr in (ev.comment_addr, ev.name_addr):
                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)

                for at in mdf._mdf.attachments:
                    for addr in (at.comment_addr, at.file_name_addr):
                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)
                    if not skip_attachments and at.embedded_data:
                        texts[at.address + v4c.AT_COMMON_SIZE] = randomized_string(at.embedded_size)

                for idx, gp_v4 in enumerate(mdf._mdf.groups, 1):
                    addr = gp_v4.data_group.comment_addr
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

                    cg_v4 = gp_v4.channel_group
                    for addr in (cg_v4.acq_name_addr, cg_v4.comment_addr):
                        if cg_v4.flags & v4c.FLAG_CG_BUS_EVENT:
                            continue

                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)

                        address = cg_v4.acq_source_addr
                        if address:
                            source = SourceInformation(address=address, stream=stream, mapped=False, tx_map={})
                            for addr in (
                                source.name_addr,
                                source.path_addr,
                                source.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                    for ch_v4 in gp_v4.channels:
                        for addr in (ch_v4.name_addr, ch_v4.unit_addr, ch_v4.comment_addr):
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)

                        address = ch_v4.source_addr
                        if address:
                            source = SourceInformation(address=address, stream=stream, mapped=False, tx_map={})
                            for addr in (
                                source.name_addr,
                                source.path_addr,
                                source.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                        address = ch_v4.conversion_addr
                        if address:
                            conv_v4 = ChannelConversionV4(
                                address=address,
                                stream=stream,
                                mapped=False,
                                tx_map={},
                            )
                            for addr in (
                                conv_v4.name_addr,
                                conv_v4.unit_addr,
                                conv_v4.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)
                            if conv_v4.conversion_type == v4c.CONVERSION_TYPE_ALG:
                                addr = conv_v4.formula_addr
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                            if conv_v4.referenced_blocks:
                                for key, block in conv_v4.referenced_blocks.items():
                                    if block:
                                        if isinstance(block, bytes):
                                            addr = conv_v4[key]
                                            if addr not in texts:
                                                stream.seek(addr + 8)
                                                size = len(block)
                                                texts[addr] = randomized_string(size)

                    if progress is not None:
                        if callable(progress):
                            progress(int(idx / count * 66), 100)
                        else:
                            progress.signals.setValue.emit(int(idx / count * 66))

                            if progress.stop:
                                return TERMINATED

            except:
                print(f"Error while scrambling the file: {format_exc()}.\nWill now use fallback method")
                texts = MDF._fallback_scramble_mf4(name)

            mdf._mdf.close()

            dst = name.with_suffix(".scrambled.mf4")

            copy(name, dst)

            with open(dst, "rb+") as fp:
                count = len(texts)
                chunk = max(count // 34, 1)
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    fp.seek(addr + 24)
                    fp.write(bts)
                    if index % chunk == 0:
                        if progress is not None:
                            if callable(progress):
                                progress(66 + idx, 100)
                            else:
                                progress.signals.setValue.emit(66 + idx)

                                if progress.stop:
                                    return TERMINATED

        else:
            stream = mdf._mdf._file

            if mdf._mdf.header.comment_addr:
                stream.seek(mdf._mdf.header.comment_addr + 2)
                size = UINT16_u(stream.read(2))[0] - 4
                texts[mdf._mdf.header.comment_addr + 4] = randomized_string(size)
            texts[36 + 0x40] = randomized_string(32)
            texts[68 + 0x40] = randomized_string(32)
            texts[100 + 0x40] = randomized_string(32)
            texts[132 + 0x40] = randomized_string(32)

            for idx, gp in enumerate(mdf._mdf.groups, 1):
                cg = gp.channel_group
                addr = cg.comment_addr

                if addr and addr not in texts:
                    stream.seek(addr + 2)
                    size = UINT16_u(stream.read(2))[0] - 4
                    texts[addr + 4] = randomized_string(size)

                if gp.trigger:
                    addr = gp.trigger.text_addr
                    if addr:
                        stream.seek(addr + 2)
                        size = UINT16_u(stream.read(2))[0] - 4
                        texts[addr + 4] = randomized_string(size)

                for ch in gp.channels:
                    for key in ("long_name_addr", "display_name_addr", "comment_addr"):
                        if hasattr(ch, key):
                            addr = getattr(ch, key)
                        else:
                            addr = 0
                        if addr and addr not in texts:
                            stream.seek(addr + 2)
                            size = UINT16_u(stream.read(2))[0] - 4
                            texts[addr + 4] = randomized_string(size)

                    texts[ch.address + 26] = randomized_string(32)
                    texts[ch.address + 58] = randomized_string(128)

                    address = ch.source_addr
                    if address:
                        extension = ChannelExtension(address=address, stream=stream)
                        if extension.type == v23c.SOURCE_ECU:
                            texts[extension.address + 12] = randomized_string(80)
                            texts[extension.address + 92] = randomized_string(32)
                        else:
                            texts[extension.address + 14] = randomized_string(36)
                            texts[extension.address + 50] = randomized_string(36)

                    address = ch.conversion_addr
                    if address:
                        texts[address + 22] = randomized_string(20)

                        conv = ChannelConversionV3(address=address, stream=stream)

                        if conv.conversion_type == v23c.CONVERSION_TYPE_FORMULA:
                            texts[conv.address + 36] = randomized_string(conv.block_len - 36)

                        if conv.referenced_blocks:
                            for key, block in conv.referenced_blocks.items():
                                if block:
                                    if isinstance(block, bytes):
                                        addr = conv[key]
                                        if addr and addr not in texts:
                                            stream.seek(addr + 2)
                                            size = UINT16_u(stream.read(2))[0] - 4
                                            texts[addr + 4] = randomized_string(size)

                if progress is not None:
                    if callable(progress):
                        progress(int(idx / count * 100), 100)
                    else:
                        progress.signals.setValue.emit(int(idx / count * 66))

                        if progress.stop:
                            return TERMINATED

            mdf._mdf.close()

            dst = name.with_suffix(".scrambled.mdf")

            copy(name, dst)

            with open(dst, "rb+") as fp:
                chunk = count // 34
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    fp.seek(addr)
                    fp.write(bts)
                    if chunk and index % chunk == 0:
                        if progress is not None:
                            if callable(progress):
                                progress(66 + idx, 100)
                            else:
                                progress.signals.setValue.emit(66 + idx)

                                if progress.stop:
                                    return TERMINATED

        if progress is not None:
            if callable(progress):
                progress(100, 100)
            else:
                progress.signals.setValue.emit(100)

        return dst

    @staticmethod
    def _fallback_scramble_mf4(name: StrOrBytesPathType) -> dict[int, bytes]:
        """scramble text blocks and keep original file structure

        Parameters
        ----------
        name : pathlib.Path
            file name

        Returns
        -------
        name : pathlib.Path
            scrambled file name

        """

        pattern = re.compile(
            rb"(?P<block>##(TX|MD))",
            re.DOTALL | re.MULTILINE,
        )

        texts = {}

        with open(name, "rb") as stream:
            stream.seek(0, 2)
            file_limit = stream.tell()
            stream.seek(0)

            for match in re.finditer(pattern, stream.read()):
                start = match.start()

                if file_limit - start >= 24:
                    stream.seek(start + 8)
                    (size,) = UINT64_u(stream.read(8))

                    if start + size <= file_limit:
                        texts[start + 24] = randomized_string(size - 24)

        return texts

    def get_group(
        self,
        index: int,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        keep_arrays: bool = False,
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        only_basenames: bool = False,
    ) -> pd.DataFrame:
        """get channel group as pandas DataFrames. If there are multiple
        occurrences for the same channel name, then a counter will be used to
        make the names unique (<original_name>_<counter>)

        Parameters
        ----------
        index : int
            channel group index
        use_display_names : bool
            use display name instead of standard channel name, if available.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool | dict[str, bool]
            the dataframe will contain the raw channel values

            .. versionadded:: 5.7.0

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        keep_arrays (False) : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.

            .. versionadded:: 5.8.0

        time_from_zero : bool
            adjust time channel to start from 0; default *True*

        empty_channels ("skip") : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*

            .. versionadded:: 5.8.0

        only_basenames (False) : bool
            use just the field names, without prefix, for structures and channel
            arrays

            .. versionadded:: 5.13.0

        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of using this argument

        Returns
        -------
        df : pandas.DataFrame

        """

        channels = [
            (None, gp_index, ch_index)
            for gp_index, channel_indexes in self._mdf.included_channels(index)[index].items()
            for ch_index in channel_indexes
        ]

        return self.to_dataframe(
            channels=channels,
            raster=raster,
            time_from_zero=time_from_zero,
            empty_channels=empty_channels,
            keep_arrays=keep_arrays,
            use_display_names=use_display_names,
            time_as_date=time_as_date,
            reduce_memory_usage=reduce_memory_usage,
            raw=raw,
            ignore_value2text_conversions=ignore_value2text_conversions,
            only_basenames=only_basenames,
        )

    def iter_to_dataframe(
        self,
        channels: ChannelsType | None = None,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        keep_arrays: bool = False,
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        use_interpolation: bool = True,
        only_basenames: bool = False,
        chunk_ram_size: int = 200 * 1024 * 1024,
        interpolate_outwards_with_nan: bool = False,
        numeric_1D_only: bool = False,
        progress=None,
    ) -> Iterator[pd.DataFrame]:
        """generator that yields pandas DataFrame's that should not exceed
        200MB of RAM

        .. versionadded:: 5.15.0

        Parameters
        ----------
        channels : list
            list of items to be filtered (default None); each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of using this argument

        time_from_zero : bool
            adjust time channel to start from 0; default *True*
        empty_channels : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*
        use_display_names : bool
            use display name instead of standard channel name, if available.
        keep_arrays : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.
        time_as_date : bool
            the dataframe index will contain the datetime timestamps
            according to the measurement start time; default *False*. If
            *True* then the argument ``time_from_zero`` will be ignored.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool
            the columns will contain the raw values

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.
        use_interpolation (True) : bool
            option to perform interpolations when multiple timestamp raster are
            present. If *False* then dataframe columns will be automatically
            filled with NaN's were the dataframe index values are not found in
            the current column's timestamps
        only_basenames (False) : bool
            use jsut the field names, without prefix, for structures and channel
            arrays
        interpolate_outwards_with_nan : bool
            use NaN values for the samples that lie outside of the original
            signal's timestamps
        chunk_ram_size : int
            desired data frame RAM usage in bytes; default 200 MB
        numeric_1D_only (False) : bool
            only keep the 1D-columns that have numeric values

            .. versionadded:: 7.0.0


        Returns
        -------
        dataframe : pandas.DataFrame
            yields pandas DataFrame's that should not exceed 200MB of RAM

        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
            raw_dict = raw
        else:
            raw_dict = None

        if channels:
            mdf = self.filter(channels)

            result = mdf.iter_to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                chunk_ram_size=chunk_ram_size,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
                numeric_1D_only=numeric_1D_only,
            )

            for df in result:
                yield df

            mdf._mdf.close()
        else:
            # channels is None

            self._mdf._set_temporary_master(None)

            masters = {index: self._mdf.get_master(index) for index in self._mdf.virtual_groups}

            if raster is not None:
                try:
                    raster_float = float(raster)
                    assert raster_float > 0
                except (TypeError, ValueError):
                    if isinstance(raster, str):
                        master = self.get(raster, raw=True, ignore_invalidation_bits=True).timestamps
                    else:
                        master = np.array(raster)
                else:
                    master = master_using_raster(self._mdf, raster_float)
            else:
                if masters:
                    master = reduce(np.union1d, masters.values())
                else:
                    master = np.array([], dtype="<f4")

            master_ = master
            if time_from_zero and len(master_):
                master_ -= master_[0]
            channel_count = sum(len(gp.channels) - 1 for gp in self._mdf.groups) + 1
            # approximation with all float64 dtype
            itemsize = channel_count * 8
            # use 200MB DataFrame chunks
            chunk_count = chunk_ram_size // itemsize or 1

            chunks, r = divmod(len(master), chunk_count)
            if r:
                chunks += 1

            for i in range(chunks):
                master = master_[chunk_count * i : chunk_count * (i + 1)]
                start = master[0]
                end = master[-1]

                data: dict[str, pd.Series] = {}
                self._mdf._set_temporary_master(None)

                used_names = UniqueDB()
                used_names.get_unique_name("timestamps")

                groups_nr = len(self._mdf.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr)

                for group_index, virtual_group in self._mdf.virtual_groups.items():
                    group_cycles = virtual_group.cycles_nr
                    if group_cycles == 0 and empty_channels == "skip":
                        continue

                    record_offset = max(np.searchsorted(masters[group_index], start).flatten()[0] - 1, 0)
                    stop = np.searchsorted(masters[group_index], end).flatten()[0]
                    record_count = min(stop - record_offset + 1, group_cycles)

                    channels = [
                        (None, gp_index, ch_index)
                        for gp_index, channel_indexes in self._mdf.included_channels(group_index)[group_index].items()
                        for ch_index in channel_indexes
                    ]
                    signals = self.select(
                        channels,
                        raw=True,
                        copy_master=False,
                        record_offset=record_offset,
                        record_count=record_count,
                        validate=False,
                    )

                    if not signals:
                        continue

                    group_master = signals[0].timestamps

                    for sig in signals:
                        if len(sig) == 0:
                            if empty_channels == "zeros":
                                sig.samples = np.zeros(
                                    len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                                    dtype=sig.samples.dtype,
                                )
                                sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                    for signal in signals:
                        if (raw_dict and not raw_dict.get(signal.name, __default__)) or (not raw_dict and not raw):
                            conversion = signal.conversion
                            if conversion:
                                samples = conversion.convert(
                                    signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                                )
                                signal.samples = samples

                            signal.raw = False
                            signal.conversion = None
                            if signal.samples.dtype.kind == "S":
                                signal.encoding = "utf-8" if isinstance(self._mdf, mdf_v4.MDF4) else "latin-1"

                    for s_index, sig in enumerate(signals):
                        sig = sig.validate(copy=False)

                        if len(sig) == 0:
                            if empty_channels == "zeros":
                                sig.samples = np.zeros(
                                    len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                                    dtype=sig.samples.dtype,
                                )
                                sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                        signals[s_index] = sig

                    if use_interpolation:
                        same_master = np.array_equal(master, group_master)

                        if not same_master and interpolate_outwards_with_nan:
                            idx = np.argwhere((master >= group_master[0]) & (master <= group_master[-1])).flatten()

                        cycles = len(group_master)

                        signals = [
                            (
                                signal.interp(
                                    master,
                                    integer_interpolation_mode=self._integer_interpolation,
                                    float_interpolation_mode=self._float_interpolation,
                                )
                                if not same_master or len(signal) != cycles
                                else signal
                            )
                            for signal in signals
                        ]

                        if not same_master and interpolate_outwards_with_nan:
                            for sig in signals:
                                sig.timestamps = sig.timestamps[idx]
                                sig.samples = sig.samples[idx]

                        group_master = master

                    signals = [sig for sig in signals if len(sig)]

                    if signals:
                        diffs = np.diff(group_master, prepend=-np.inf) > 0

                        if group_master.dtype.byteorder not in target_byte_order:
                            group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

                        if np.all(diffs):
                            index = pd.Index(group_master, tupleize_cols=False)

                        else:
                            idx = np.argwhere(diffs).flatten()
                            group_master = group_master[idx]

                            index = pd.Index(group_master, tupleize_cols=False)

                            for sig in signals:
                                sig.samples = sig.samples[idx]
                                sig.timestamps = sig.timestamps[idx]

                    size = len(index)
                    for sig in signals:
                        if sig.timestamps.dtype.byteorder not in target_byte_order:
                            sig.timestamps = sig.timestamps.byteswap().view(sig.timestamps.dtype.newbyteorder())

                        sig_index = index if len(sig) == size else pd.Index(sig.timestamps, tupleize_cols=False)

                        # byte arrays
                        if len(sig.samples.shape) > 1:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name

                            channel_name = used_names.get_unique_name(channel_name)

                            if sig.samples.dtype.byteorder not in target_byte_order:
                                sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                            data[channel_name] = pd.Series(
                                list(sig.samples),
                                index=sig_index,
                            )

                        # arrays and structures
                        elif sig.samples.dtype.names:
                            for name, series in components(
                                sig.samples,
                                sig.name,
                                used_names,
                                master=sig_index,
                                only_basenames=only_basenames,
                            ):
                                data[name] = series

                        # scalars
                        else:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name

                            channel_name = used_names.get_unique_name(channel_name)

                            if reduce_memory_usage and sig.samples.dtype.kind in "SU":
                                unique = np.unique(sig.samples)

                                if sig.samples.dtype.byteorder not in target_byte_order:
                                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                                if len(sig.samples) / len(unique) >= 2:
                                    data[channel_name] = pd.Series(
                                        sig.samples,
                                        index=sig_index,
                                        dtype="category",
                                    )
                                else:
                                    data[channel_name] = pd.Series(
                                        sig.samples,
                                        index=sig_index,
                                    )
                            else:
                                if reduce_memory_usage:
                                    sig.samples = downcast(sig.samples)

                                if sig.samples.dtype.byteorder not in target_byte_order:
                                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                                data[channel_name] = pd.Series(
                                    sig.samples,
                                    index=sig_index,
                                )

                    if progress is not None:
                        if callable(progress):
                            progress(group_index + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(group_index + 1)

                strings, nonstrings = {}, {}

                for col, series in data.items():
                    if series.dtype.kind == "S":
                        strings[col] = series
                    else:
                        nonstrings[col] = series

                if numeric_1D_only:
                    nonstrings = {col: series for col, series in nonstrings.items() if series.dtype.kind in "uif"}
                    strings = {}

                df = pd.DataFrame(nonstrings, index=master)

                if strings:
                    df_strings = pd.DataFrame(strings, index=master)
                    df = pd.concat([df, df_strings], axis=1)

                df.index.name = "timestamps"

                if time_as_date:
                    delta = pd.to_timedelta(df.index, unit="s")

                    new_index = self._mdf.header.start_time + delta
                    df.set_index(new_index, inplace=True)

                yield df

    @overload
    def to_dataframe(
        self,
        channels: ChannelsType | None = ...,
        raster: RasterType | None = ...,
        time_from_zero: bool = ...,
        empty_channels: EmptyChannelsType = ...,
        keep_arrays: bool = ...,
        use_display_names: bool = ...,
        time_as_date: bool = ...,
        reduce_memory_usage: bool = ...,
        raw: bool | dict[str, bool] = ...,
        ignore_value2text_conversions: bool = ...,
        use_interpolation: bool = ...,
        only_basenames: bool = ...,
        interpolate_outwards_with_nan: bool = ...,
        numeric_1D_only: bool = ...,
        progress: None = ...,
    ) -> pd.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        channels: ChannelsType | None = ...,
        raster: RasterType | None = ...,
        time_from_zero: bool = ...,
        empty_channels: EmptyChannelsType = ...,
        keep_arrays: bool = ...,
        use_display_names: bool = ...,
        time_as_date: bool = ...,
        reduce_memory_usage: bool = ...,
        raw: bool | dict[str, bool] = ...,
        ignore_value2text_conversions: bool = ...,
        use_interpolation: bool = ...,
        only_basenames: bool = ...,
        interpolate_outwards_with_nan: bool = ...,
        numeric_1D_only: bool = ...,
        *,
        progress: Callable[[int, int], None] | Any,
    ) -> pd.DataFrame | object: ...

    def to_dataframe(
        self,
        channels: ChannelsType | None = None,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        keep_arrays: bool = False,
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        use_interpolation: bool = True,
        only_basenames: bool = False,
        interpolate_outwards_with_nan: bool = False,
        numeric_1D_only: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> pd.DataFrame | object:
        """generate pandas DataFrame

        Parameters
        ----------
        channels : list
            list of items to be filtered (default None); each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of using this argument

        time_from_zero : bool
            adjust time channel to start from 0; default *True*
        empty_channels : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*
        use_display_names : bool
            use display name instead of standard channel name, if available.
        keep_arrays : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.
        time_as_date : bool
            the dataframe index will contain the datetime timestamps
            according to the measurement start time; default *False*. If
            *True* then the argument ``time_from_zero`` will be ignored.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool
            the columns will contain the raw values

            .. versionadded:: 5.7.0

            .. versionchanged:: 8.0.0

                provide individual raw mode based on a dict. If the parameters is given
                as dict then it must contain the key ``__default__`` with the default raw value. The dict keys
                are the channel names and the values are the boolean raw values for each channel.

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        use_interpolation (True) : bool
            option to perform interpolations when multiple timestamp raster are
            present. If *False* then dataframe columns will be automatically
            filled with NaN's were the dataframe index values are not found in
            the current column's timestamps

            .. versionadded:: 5.11.0

        only_basenames (False) : bool
            use just the field names, without prefix, for structures and channel
            arrays

            .. versionadded:: 5.13.0

        interpolate_outwards_with_nan : bool
            use NaN values for the samples that lie outside of the original
            signal's timestamps

            .. versionadded:: 5.15.0

        Returns
        -------
        dataframe : pandas.DataFrame

        """
        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
            raw_dict = raw
        else:
            raw_dict = None

        if channels is not None:
            mdf = self.filter(channels)

            result = mdf.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
                numeric_1D_only=numeric_1D_only,
            )

            mdf._mdf.close()
            return result

        target_byte_order = "<=" if sys.byteorder == "little" else ">="

        data = {}

        self._mdf._set_temporary_master(None)

        if raster is not None:
            try:
                raster_float = float(raster)
                assert raster_float > 0
            except (TypeError, ValueError):
                if isinstance(raster, str):
                    master = self._mdf.get(raster).timestamps
                else:
                    master = np.array(raster)
            else:
                master = master_using_raster(self._mdf, raster_float)
        else:
            masters = {index: self._mdf.get_master(index) for index in self._mdf.virtual_groups}

            if masters:
                master = reduce(np.union1d, masters.values())
            else:
                master = np.array([], dtype=np.dtype("<f4"))

            del masters

        idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
        master = master[idx]

        used_names = UniqueDB()
        used_names.get_unique_name("timestamps")

        groups_nr = len(self._mdf.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    return TERMINATED

        for group_index, (virtual_group_index, virtual_group) in enumerate(self._mdf.virtual_groups.items()):
            if virtual_group.cycles_nr == 0 and empty_channels == "skip":
                continue

            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self._mdf.included_channels(virtual_group_index)[
                    virtual_group_index
                ].items()
                for ch_index in channel_indexes
                if ch_index != self._mdf.masters_db.get(gp_index, None)
            ]

            signals = self.select(channels, raw=True, copy_master=False, validate=False)

            if not signals:
                continue

            group_master = signals[0].timestamps

            for sig in signals:
                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

            for signal in signals:
                if (raw_dict and not raw_dict.get(signal.name, __default__)) or (not raw_dict and not raw):
                    conversion = signal.conversion
                    if conversion:
                        samples = conversion.convert(
                            signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                        )
                        signal.samples = samples

                    signal.raw = False
                    signal.conversion = None
                    if signal.samples.dtype.kind == "S":
                        signal.encoding = "utf-8" if isinstance(self._mdf, mdf_v4.MDF4) else "latin-1"

            for s_index, sig in enumerate(signals):
                sig = sig.validate(copy=False)

                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                signals[s_index] = sig

            if use_interpolation:
                same_master = np.array_equal(master, group_master)

                if not same_master and interpolate_outwards_with_nan:
                    idx = np.argwhere((master >= group_master[0]) & (master <= group_master[-1])).flatten()

                cycles = len(group_master)

                signals = [
                    (
                        signal.interp(
                            master,
                            integer_interpolation_mode=self._mdf._integer_interpolation,
                            float_interpolation_mode=self._mdf._float_interpolation,
                        )
                        if not same_master or len(signal) != cycles
                        else signal
                    )
                    for signal in signals
                ]

                if not same_master and interpolate_outwards_with_nan:
                    for sig in signals:
                        sig.timestamps = sig.timestamps[idx]
                        sig.samples = sig.samples[idx]

                group_master = master

            if any(len(sig) for sig in signals):
                signals = [sig for sig in signals if len(sig)]

            if group_master.dtype.byteorder not in target_byte_order:
                group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

            if signals:
                diffs = np.diff(group_master, prepend=-np.inf) > 0
                if np.all(diffs):
                    index = pd.Index(group_master, tupleize_cols=False)

                else:
                    idx = np.argwhere(diffs).flatten()
                    group_master = group_master[idx]

                    index = pd.Index(group_master, tupleize_cols=False)

                    for sig in signals:
                        sig.samples = sig.samples[idx]
                        sig.timestamps = sig.timestamps[idx]
            else:
                index = pd.Index(group_master, tupleize_cols=False)

            size = len(index)
            for sig in signals:
                if sig.timestamps.dtype.byteorder not in target_byte_order:
                    sig.timestamps = sig.timestamps.byteswap().view(sig.timestamps.dtype.newbyteorder())

                sig_index = index if len(sig) == size else pd.Index(sig.timestamps, tupleize_cols=False)

                # byte arrays
                if len(sig.samples.shape) > 1:
                    if use_display_names:
                        channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if sig.samples.dtype.byteorder not in target_byte_order:
                        sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                    data[channel_name] = pd.Series(
                        list(sig.samples),
                        index=sig_index,
                    )

                # arrays and structures
                elif sig.samples.dtype.names:
                    for name, series in components(
                        sig.samples,
                        sig.name,
                        used_names,
                        master=sig_index,
                        only_basenames=only_basenames,
                    ):
                        data[name] = series

                # scalars
                else:
                    if use_display_names:
                        channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if reduce_memory_usage and sig.samples.dtype.kind not in "SU":
                        if sig.samples.size > 0:
                            sig.samples = downcast(sig.samples)

                    if sig.samples.dtype.byteorder not in target_byte_order:
                        sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                    data[channel_name] = pd.Series(sig.samples, index=sig_index)

            if progress is not None:
                if callable(progress):
                    progress(group_index + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(group_index + 1)

                    if progress.stop:
                        return TERMINATED

        strings, nonstrings = {}, {}

        for col, series in data.items():
            if series.dtype.kind == "S":
                strings[col] = series
            else:
                nonstrings[col] = series

        if numeric_1D_only:
            nonstrings = {col: series for col, series in nonstrings.items() if series.dtype.kind in "uif"}
            strings = {}

        df = pd.DataFrame(nonstrings, index=master)

        if strings:
            df_strings = pd.DataFrame(strings, index=master)
            df = pd.concat([df, df_strings], axis=1)

        df.index.name = "timestamps"

        if time_as_date:
            delta = pd.to_timedelta(df.index, unit="s")

            new_index = self._mdf.header.start_time + delta
            df.set_index(new_index, inplace=True)

        elif time_from_zero and len(master):
            df.set_index(df.index - df.index[0], inplace=True)

        return df

    def extract_bus_logging(
        self,
        database_files: dict[BusType, Iterable[DbcFileType]],
        version: Version | None = None,
        ignore_invalid_signals: bool | None = None,
        consolidated_j1939: bool | None = None,
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress=None,
    ) -> MDF | object:
        """extract all possible CAN signal using the provided databases.

        Changed in version 6.0.0 from `extract_can_logging`

        Parameters
        ----------
        database_files : dict
            each key will contain an iterable of database files for that bus type. The
            supported bus types are "CAN", "LIN". The iterables will contain the
            (databases, valid bus) pairs. The database can be a  str, pathlib.Path or canamtrix.CanMatrix object.
            The valid bus is an integer specifying for which bus channel the database
            can be applied; 0 means any bus channel.

            .. versionchanged:: 6.0.0 added canmatrix.CanMatrix type

            .. versionchanged:: 6.3.0 added bus channel fileter

        version (None) : str
            output file version
        ignore_invalid_signals (None) : bool | None
            ignore signals that have all samples equal to their maximum value

            .. versionadded:: 5.7.0

            .. deprecated:: 7.0.2
                this argument is no longer used and will be removed in the future

        consolidated_j1939 (None) : bool | None
            handle PGNs from all the messages as a single instance

            .. versionadded:: 5.7.0

            .. deprecated:: 7.2.0
                this argument is no longer used and will be removed in the future. The PGNs are saved
                separately according to the source address

        ignore_value2text_conversion (True): bool
            ignore value to text conversions

            .. versionadded:: 5.23.0

        prefix ("") : str
            prefix that will be added to the channel group names and signal names in
            the output file

            .. versionadded:: 6.3.0


        Returns
        -------
        mdf : MDF
            new MDF file that contains the succesfully extracted signals

        Examples
        --------
        >>> "extrac CAN and LIN bus logging"
        >>> mdf = asammdf.MDF(r'bus_logging.mf4')
        >>> databases = {
        ...     "CAN": [("file1.dbc", 0), ("file2.arxml", 2)],
        ...     "LIN": [("file3.dbc", 0)],
        ... }
        >>> extracted = mdf.extract_bus_logging(database_files=database_files)
        >>> ...
        >>> "extrac just LIN bus logging"
        >>> mdf = asammdf.MDF(r'bus_logging.mf4')
        >>> databases = {
        ...     "LIN": [("file3.dbc", 0)],
        ... }
        >>> extracted = mdf.extract_bus_logging(database_files=database_files)

        """
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("The method `extract_bus_logging` is only available for MDF4 files")

        if ignore_invalid_signals is not None:
            warn(
                "The argument `ignore_invalid_signals` from the method `extract_bus_logging` is no longer used and will be removed in the future",
                stacklevel=1,
            )

        if consolidated_j1939 is not None:
            warn(
                "The argument `consolidated_j1939` from the method `extract_bus_logging` is no longer used and will be removed in the future",
                stacklevel=1,
            )

        if version is None:
            version = self._mdf.version
        else:
            version = validate_version_argument(version, self._mdf.default_version)

        out = MDF(
            version=version,
            password=self._mdf.password,
            use_display_names=True,
        )

        if isinstance(out._mdf, mdf_v4.MDF4):
            out_mdf = out._mdf
        else:
            raise MdfException("The method `extract_bus_logging` is only available for MDF4 files")

        out_mdf.header.start_time = self._mdf.header.start_time

        self._mdf.last_call_info = {}

        if database_files.get("CAN", None):
            to_keep_or_terminated = self._extract_can_logging(
                out_mdf,
                database_files["CAN"],
                ignore_value2text_conversion,
                prefix,
                progress=progress,
            )

            if isinstance(to_keep_or_terminated, list):
                if to_keep_or_terminated:
                    tmp = out.filter(to_keep_or_terminated, out_mdf.version)
                    out_mdf.close()
                    out = tmp

                    if not out_mdf.groups:
                        logger.warning(
                            f'No CAN signals could be extracted from "{self._mdf.name}". The'
                            "output file will be empty."
                        )
                else:
                    terminated = None
            else:
                terminated = to_keep_or_terminated

        if database_files.get("LIN", None):
            terminated = self._extract_lin_logging(
                out_mdf,
                database_files["LIN"],
                ignore_value2text_conversion,
                prefix,
                progress=progress,
            )

        return terminated if terminated else out

    def _extract_can_logging(
        self,
        output_file: mdf_v4.MDF4,
        dbc_files: Iterable[DbcFileType],
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress=None,
    ) -> list[tuple[None, int, int]] | object | None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("the method '_extract_lin_logging' is only available for MDF4")

        out = output_file

        max_flags: list[list[list[bool | np.bool_]]] = []

        valid_dbc_files: list[tuple[CanMatrix, StrPathType, int]] = []
        unique_name = UniqueDB()
        for dbc_name, bus_channel in dbc_files:
            if isinstance(dbc_name, CanMatrix):
                valid_dbc_files.append(
                    (
                        dbc_name,
                        unique_name.get_unique_name("UserProvidedCanMatrix"),
                        bus_channel,
                    )
                )
            else:
                dbc = load_can_database(Path(dbc_name))
                if dbc is None:
                    continue
                else:
                    valid_dbc_files.append((dbc, dbc_name, bus_channel))

        count = sum(
            1
            for group in self._mdf.groups
            if (
                group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                and group.channel_group.acq_source
                and group.channel_group.acq_source.bus_type == v4c.BUS_TYPE_CAN
            )
        )
        count *= len(valid_dbc_files)

        if progress is not None:
            if callable(progress):
                progress(0, count)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(count)

                if progress.stop:
                    return TERMINATED

        cntr = 0

        total_unique_ids: set[tuple[int, bool]] = set()
        found_ids: defaultdict[StrPathType, set[tuple[tuple[int, int, bool], str]]] = defaultdict(set)
        not_found_ids: defaultdict[StrPathType, list[tuple[tuple[int, bool] | int, str]]] = defaultdict(list)
        unknown_ids_dict: defaultdict[int | tuple[int, bool], list[bool]] = defaultdict(list)

        for dbc, dbc_name, bus_channel in valid_dbc_files:
            messages: dict[tuple[int, bool], Frame] = {
                (message.arbitration_id.id, message.arbitration_id.extended): message for message in dbc
            }

            protocol_type: str = dbc.attributes.get("ProtocolType", "")
            global_is_j1939 = protocol_type.lower() == "j1939"
            not_extended = [msg for msg in dbc if not msg.arbitration_id.extended]
            if global_is_j1939 and not_extended:
                logger.warning(
                    f"Not all j1939 messages in <{dbc_name}> seem to use extended addressing. Disabling global j1939 flag..."
                )
                for msg in not_extended:
                    logger.warning(f"  {msg} with id {msg.arbitration_id}")
                global_is_j1939 = False  # Relax req on j1939 adressing

            j1939_messages = {
                (
                    message.arbitration_id.pgn,
                    message.arbitration_id.j1939_source,
                ): message
                for message in dbc
                if message.is_j1939 or global_is_j1939
            }

            current_not_found: set[tuple[tuple[int, bool] | int, str]] = {
                (
                    (
                        (message.arbitration_id.id, message.arbitration_id.extended)
                        if not message.is_j1939 and not global_is_j1939
                        else message.arbitration_id.pgn
                    ),
                    message.name,
                )
                for msg_id, message in messages.items()
            }

            msg_map = {}

            for i, group in enumerate(self._mdf.groups):
                if (
                    not group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                    or (group.channel_group.acq_source and group.channel_group.acq_source.bus_type != v4c.BUS_TYPE_CAN)
                    or not "CAN_DataFrame" in [ch.name for ch in group.channels]
                ):
                    continue

                self._mdf._prepare_record(group)
                data = self._mdf._load_data(group, optimize_read=False)

                for fragment in data:
                    self._mdf._set_temporary_master(None)
                    self._mdf._set_temporary_master(self._mdf.get_master(i, data=fragment))

                    bus_ids = self._mdf.get(
                        "CAN_DataFrame.BusChannel",
                        group=i,
                        data=fragment,
                    ).samples.astype("<u1")

                    msg_ids = self._mdf.get("CAN_DataFrame.ID", group=i, data=fragment).astype("<u4")
                    try:
                        msg_ide = self._mdf.get("CAN_DataFrame.IDE", group=i, data=fragment).astype("<u1")
                    except:
                        msg_ide = (msg_ids & 0x80000000) >> 31

                    msg_ids &= 0x1FFFFFFF

                    data_bytes = self._mdf.get(
                        "CAN_DataFrame.DataBytes",
                        group=i,
                        data=fragment,
                    ).samples

                    buses = np.unique(bus_ids)

                    for bus in buses:
                        if bus_channel and bus != bus_channel:
                            continue

                        idx = np.argwhere(bus_ids == bus).ravel()
                        bus_t = msg_ids.timestamps[idx]
                        bus_msg_ids = msg_ids.samples[idx]
                        bus_msg_ide = msg_ide[idx]
                        bus_data_bytes = data_bytes[idx]

                        tmp_pgn = bus_msg_ids >> 8
                        ps = tmp_pgn & 0xFF
                        pf = (bus_msg_ids >> 16) & 0xFF
                        _pgn = tmp_pgn & 0x3FF00
                        j1939_msg_pgns = np.where(pf >= 240, _pgn + ps, _pgn)
                        j9193_msg_sa = bus_msg_ids & 0xFF

                        unique_ids: set[tuple[int, bool]] = set(zip(bus_msg_ids.tolist(), bus_msg_ide.samples.tolist()))

                        total_unique_ids = total_unique_ids | set(unique_ids)

                        for msg_id, is_extended in sorted(unique_ids):
                            message = messages.get((msg_id, is_extended), None)

                            if message is None:
                                tmp_pgn = msg_id >> 8
                                ps = tmp_pgn & 0xFF
                                pf = (msg_id >> 16) & 0xFF
                                _pgn = tmp_pgn & 0x3FF00
                                msg_pgn = _pgn + ps if pf >= 240 else _pgn

                                for (_pgn, _sa), _msg in j1939_messages.items():
                                    if _pgn == msg_pgn:
                                        message = _msg
                                        break
                                else:
                                    unknown_ids_dict[msg_id].append(True)
                                    continue

                            is_j1939 = message.is_j1939 or global_is_j1939
                            if is_j1939:
                                source_address = msg_id & 0xFF
                                pgn_number: int = message.arbitration_id.pgn
                                key = (pgn_number, source_address, True)
                                found_ids[dbc_name].add((key, message.name))

                                try:
                                    current_not_found.remove((pgn_number, message.name))
                                except KeyError:
                                    pass

                            else:
                                key = msg_id, bool(is_extended), False

                                found_ids[dbc_name].add((key, message.name))
                                try:
                                    current_not_found.remove(((msg_id, is_extended), message.name))
                                except KeyError:
                                    pass

                            unknown_ids_dict[(msg_id, is_extended)].append(False)

                            if is_j1939:
                                idx = np.argwhere(
                                    (j1939_msg_pgns == pgn_number) & (j9193_msg_sa == source_address)
                                ).ravel()
                            else:
                                idx = np.argwhere(
                                    (bus_msg_ids == msg_id) & (bus_msg_ide.samples == is_extended)
                                ).ravel()

                            payload = bus_data_bytes[idx]
                            t = bus_t[idx]

                            try:
                                extracted_signals = bus_logging_utils.extract_mux(
                                    payload,
                                    message,
                                    msg_id,
                                    bus,
                                    t,
                                    original_message_id=source_address if is_j1939 else None,
                                    ignore_value2text_conversion=ignore_value2text_conversion,
                                    is_j1939=is_j1939,
                                    is_extended=is_extended,
                                    raw=True,
                                )
                            except:
                                print(format_exc())
                                raise

                            for entry, signals in extracted_signals.items():
                                if len(next(iter(signals.values()))["samples"]) == 0:
                                    continue

                                if entry not in msg_map:
                                    sigs = []

                                    index = len(out.groups)
                                    msg_map[entry] = index

                                    for name_, signal in signals.items():
                                        signal_name = f"{prefix}{signal['name']}"
                                        sig = Signal(
                                            samples=signal["samples"],
                                            timestamps=signal["t"],
                                            name=signal_name,
                                            comment=signal["comment"],
                                            unit=signal["unit"],
                                            invalidation_bits=signal["invalidation_bits"],
                                            display_names={
                                                f"CAN{bus}.{message.name}.{signal_name}": "bus",
                                                f"{message.name}.{signal_name}": "message",
                                            },
                                            raw=True,
                                            conversion=signal["conversion"],
                                        )

                                        sigs.append(sig)

                                    if is_j1939:
                                        if prefix:
                                            comment = f"{prefix}: CAN{bus} ID=0x{msg_id:X} {message} PGN=0x{pgn_number:X} SA=0x{source_address:X}"
                                        else:
                                            comment = f"CAN{bus} ID=0x{msg_id:X} {message} PGN=0x{pgn_number:X} SA=0x{source_address:X}"
                                        acq_name = f"SourceAddress = 0x{source_address}"
                                    else:
                                        if prefix:
                                            acq_name = (
                                                f"{prefix}: CAN{bus} message ID=0x{msg_id:X} EXT={bool(is_extended)}"
                                            )
                                            comment = f'{prefix}: CAN{bus} - message "{message}" 0x{msg_id:X} EXT={bool(is_extended)}'
                                        else:
                                            acq_name = f"CAN{bus} message ID=0x{msg_id:X} EXT={bool(is_extended)}"
                                            comment = (
                                                f"CAN{bus} - message {message} 0x{msg_id:X} EXT={bool(is_extended)}"
                                            )

                                    acq_source = Source(
                                        name=acq_name,
                                        path=f"CAN{int(bus)}.CAN_DataFrame.ID=0x{message.arbitration_id.id:X} EXT={bool(is_extended)}",
                                        comment=f"""\
<SIcomment>
    <TX>CAN{bus} data frame 0x{message.arbitration_id.id:X} EXT={bool(is_extended)} - {message.name}</TX>
    <bus name="CAN{int(bus)}"/>
    <common_properties>
        <e name="ChannelNo" type="integer">{int(bus)}</e>
    </common_properties>
</SIcomment>""",
                                        source_type=v4c.SOURCE_BUS,
                                        bus_type=v4c.BUS_TYPE_CAN,
                                    )

                                    for sig in sigs:
                                        sig.source = acq_source

                                    cg_nr = out.append(
                                        sigs,
                                        acq_name=acq_name,
                                        acq_source=acq_source,
                                        comment=comment,
                                        common_timebase=True,
                                    )

                                    out.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

                                    if is_j1939:
                                        max_flags.append([[False]])
                                        for ch_index, sig in enumerate(sigs, 1):
                                            max_flags[cg_nr].append([np.all(sig.invalidation_bits)])
                                    else:
                                        max_flags.append([[False]] * (len(sigs) + 1))

                                else:
                                    index = msg_map[entry]

                                    signal_samples: list[tuple[NDArray[Any], NDArray[Any] | None]] = []

                                    for name_, signal in signals.items():
                                        signal_samples.append(
                                            (
                                                signal["samples"],
                                                signal["invalidation_bits"],
                                            )
                                        )

                                        t = signal["t"]

                                    if is_j1939:
                                        signal_samples_and_invalidation_bits = typing.cast(
                                            list[tuple[NDArray[Any], NDArray[Any]]], signal_samples
                                        )
                                        for ch_index, (_, invalidation_bits) in enumerate(
                                            signal_samples_and_invalidation_bits, 1
                                        ):
                                            max_flags[index][ch_index].append(np.all(invalidation_bits))

                                    signal_samples.insert(0, (t, None))

                                    out.extend(index, signal_samples)
                    self._mdf._set_temporary_master(None)

                cntr += 1
                if progress is not None:
                    if callable(progress):
                        progress(cntr, count)
                    else:
                        progress.signals.setValue.emit(cntr)

                        if progress.stop:
                            return TERMINATED

            if current_not_found:
                not_found_ids[dbc_name] = list(current_not_found)

        unknown_ids = {msg_id for msg_id, not_found in unknown_ids_dict.items() if all(not_found)}

        self._mdf.last_call_info["CAN"] = {
            "dbc_files": dbc_files,
            "total_unique_ids": total_unique_ids,
            "unknown_id_count": len(unknown_ids),
            "not_found_ids": not_found_ids,
            "found_ids": found_ids,
            "unknown_ids": unknown_ids,
        }

        to_keep = []
        all_channels = []

        for i, group in enumerate(out.groups):
            for j, channel in enumerate(group.channels[1:], 1):
                if not all(max_flags[i][j]):
                    to_keep.append((None, i, j))
                all_channels.append((None, i, j))

        if to_keep != all_channels:
            return to_keep

        return None

    def _extract_lin_logging(
        self,
        output_file: mdf_v4.MDF4,
        dbc_files: Iterable[DbcFileType],
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress=None,
    ) -> object | None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("the method '_extract_lin_logging' is only available for MDF4")

        out = output_file

        valid_dbc_files: list[tuple[CanMatrix, StrPathType, int]] = []
        unique_name = UniqueDB()
        for dbc_name, bus_channel in dbc_files:
            if isinstance(dbc_name, CanMatrix):
                valid_dbc_files.append(
                    (
                        dbc_name,
                        unique_name.get_unique_name("UserProvidedCanMatrix"),
                        bus_channel,
                    )
                )
            else:
                dbc = load_can_database(Path(dbc_name))
                if dbc is None:
                    continue
                else:
                    valid_dbc_files.append((dbc, dbc_name, bus_channel))

        count = sum(
            1
            for group in self._mdf.groups
            if (
                group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                and group.channel_group.acq_source
                and group.channel_group.acq_source.bus_type == v4c.BUS_TYPE_LIN
            )
        )
        count *= len(valid_dbc_files)

        if progress is not None:
            if callable(progress):
                progress(0, count)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(count)

                if progress.stop:
                    return TERMINATED

        cntr = 0

        total_unique_ids: set[tuple[int, ...]] = set()
        found_ids: defaultdict[StrPathType, set[tuple[tuple[int, bool, bool], str]]] = defaultdict(set)
        not_found_ids: defaultdict[StrPathType, list[tuple[int, str]]] = defaultdict(list)
        unknown_ids_dict: defaultdict[int, list[bool]] = defaultdict(list)

        for dbc, dbc_name, bus_channel in valid_dbc_files:
            messages: dict[int, Frame] = {message.arbitration_id.id: message for message in dbc}

            current_not_found_ids = {(msg_id, message.name) for msg_id, message in messages.items()}

            msg_map = {}

            for i, group in enumerate(self._mdf.groups):
                if (
                    not group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                    or (group.channel_group.acq_source and group.channel_group.acq_source.bus_type != v4c.BUS_TYPE_LIN)
                    or not "LIN_Frame" in [ch.name for ch in group.channels]
                ):
                    continue

                self._mdf._prepare_record(group)  # type: ignore[arg-type]
                data = self._mdf._load_data(group, optimize_read=False)  # type: ignore[arg-type]

                for fragment in data:
                    self._mdf._set_temporary_master(None)
                    self._mdf._set_temporary_master(self._mdf.get_master(i, data=fragment))

                    msg_ids = self._mdf.get("LIN_Frame.ID", group=i, data=fragment).astype("<u4") & 0x1FFFFFFF

                    original_ids = msg_ids.samples.copy()

                    data_bytes = self._mdf.get(
                        "LIN_Frame.DataBytes",
                        group=i,
                        data=fragment,
                    ).samples

                    try:
                        bus_ids = self._mdf.get(
                            "LIN_Frame.BusChannel",
                            group=i,
                            data=fragment,
                        ).samples.astype("<u1")
                    except:
                        bus_ids = np.ones(len(original_ids), dtype="u1")

                    bus_t = msg_ids.timestamps
                    bus_msg_ids = msg_ids.samples
                    bus_data_bytes = data_bytes
                    original_msg_ids = original_ids

                    unique_ids = np.unique(np.rec.fromarrays([bus_msg_ids, bus_msg_ids]))

                    total_unique_ids = total_unique_ids | {tuple(int(e) for e in f) for f in unique_ids}

                    buses = np.unique(bus_ids)

                    for bus in buses:
                        if bus_channel and bus != bus_channel:
                            continue

                        for msg_id_record in sorted(unique_ids.tolist()):
                            msg_id = int(msg_id_record[0])
                            original_msg_id = int(msg_id_record[1])
                            message = messages.get(msg_id, None)
                            if message is None:
                                unknown_ids_dict[msg_id].append(True)
                                continue

                            found_ids[dbc_name].add(((msg_id, False, False), message.name))
                            try:
                                current_not_found_ids.remove((msg_id, message.name))
                            except KeyError:
                                pass

                            unknown_ids_dict[msg_id].append(False)

                            idx = np.argwhere(bus_msg_ids == msg_id).ravel()
                            payload = bus_data_bytes[idx]
                            t = bus_t[idx]

                            extracted_signals = bus_logging_utils.extract_mux(
                                payload,
                                message,
                                msg_id,
                                bus,
                                t,
                                original_message_id=None,
                                ignore_value2text_conversion=ignore_value2text_conversion,
                                raw=True,
                            )

                            for entry, signals in extracted_signals.items():
                                if len(next(iter(signals.values()))["samples"]) == 0:
                                    continue
                                if entry not in msg_map:
                                    sigs: list[Signal] = []

                                    index = len(out.groups)
                                    msg_map[entry] = index

                                    for name_, signal in signals.items():
                                        signal_name = f"{prefix}{signal['name']}"
                                        sig = Signal(
                                            samples=signal["samples"],
                                            timestamps=signal["t"],
                                            name=signal_name,
                                            comment=signal["comment"],
                                            unit=signal["unit"],
                                            invalidation_bits=signal["invalidation_bits"],
                                            display_names={
                                                f"LIN{bus}.{message.name}.{signal_name}": "bus",
                                                f"{message.name}.{signal_name}": "message",
                                            },
                                            raw=True,
                                            conversion=signal["conversion"],
                                        )

                                        sigs.append(sig)

                                    if prefix:
                                        acq_name = f"{prefix}: from LIN{bus} message ID=0x{msg_id:X}"
                                    else:
                                        acq_name = f"from LIN{bus} message ID=0x{msg_id:X}"

                                    acq_source = Source(
                                        name=acq_name,
                                        path=f"LIN{int(bus)}.LIN_Frame.ID=0x{message.arbitration_id.id:X}",
                                        comment=f"""\
<SIcomment>
    <TX>LIN{bus} data frame 0x{message.arbitration_id.id:X} - {message.name}</TX>
    <bus name="LIN{int(bus)}"/>
    <common_properties>
        <e name="ChannelNo" type="integer">{int(bus)}</e>
    </common_properties>
</SIcomment>""",
                                        source_type=v4c.SOURCE_BUS,
                                        bus_type=v4c.BUS_TYPE_LIN,
                                    )

                                    for sig in sigs:
                                        sig.source = acq_source

                                    cg_nr = out.append(
                                        sigs,
                                        acq_name=acq_name,
                                        acq_source=acq_source,
                                        comment=f"from LIN{bus} - message {message} 0x{msg_id:X}",
                                        common_timebase=True,
                                    )

                                    out.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

                                else:
                                    index = msg_map[entry]

                                    signal_samples: list[tuple[NDArray[Any], NDArray[Any] | None]] = []

                                    for name_, signal in signals.items():
                                        signal_samples.append(
                                            (
                                                signal["samples"],
                                                signal["invalidation_bits"],
                                            )
                                        )

                                        t = signal["t"]

                                    signal_samples.insert(0, (t, None))

                                    out.extend(index, signal_samples)
                    self._mdf._set_temporary_master(None)

                cntr += 1
                if progress is not None:
                    if callable(progress):
                        progress(cntr, count)
                    else:
                        progress.signals.setValue.emit(cntr)

                        if progress.stop:
                            return TERMINATED

            if current_not_found_ids:
                not_found_ids[dbc_name] = list(current_not_found_ids)

        unknown_ids = {msg_id for msg_id, not_found in unknown_ids_dict.items() if all(not_found)}

        self._mdf.last_call_info["LIN"] = {
            "dbc_files": dbc_files,
            "total_unique_ids": total_unique_ids,
            "unknown_id_count": len(unknown_ids),
            "not_found_ids": not_found_ids,
            "found_ids": found_ids,
            "unknown_ids": unknown_ids,
        }

        if not out.groups:
            logger.warning(
                f'No LIN signals could be extracted from "{self._mdf.name}". The' "output file will be empty."
            )

        return None

    @property
    def start_time(self) -> datetime:
        """getter and setter the measurement start timestamp

        Returns
        -------
        timestamp : datetime.datetime
            start timestamp

        """

        return self._mdf.header.start_time

    @start_time.setter
    def start_time(self, timestamp: datetime) -> None:
        self._mdf.header.start_time = timestamp

    def save(
        self,
        dst: WritableBufferType | StrPathType,
        overwrite: bool = False,
        compression: CompressionType = 0,
        progress: Callable[[int, int], None] | Any | None = None,
        add_history_block: bool = True,
    ) -> Path | object:
        if isinstance(dst, (os.PathLike, str)):
            return self._mdf.save(
                dst,
                overwrite=overwrite,
                compression=compression,
                progress=progress,
                add_history_block=add_history_block,
            )
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.save(
                dst,
                overwrite=overwrite,
                compression=compression,
                progress=progress,
                add_history_block=add_history_block,
            )
        raise MdfException("The method `save` for MDF2 and MDF3 files does not support writing to a buffer")

    def cleanup_timestamps(
        self,
        minimum: float,
        maximum: float,
        exp_min: int = -15,
        exp_max: int = 15,
        version: Version | None = None,
        progress=None,
    ) -> MDF | object:
        """convert *MDF* to other version

        .. versionadded:: 5.22.0

        Parameters
        ----------
        minimum : float
            minimum plausible time stamp
        maximum : float
            maximum plausible time stamp
        exp_min (-15) : int
            minimum plausible exponent used for the time stamps float values
        exp_max (15) : int
            maximum plausible exponent used for the time stamps float values
        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default the same as
            the input file

        Returns
        -------
        out : MDF
            new *MDF* object

        """

        if version is None:
            version = self._mdf.version
        else:
            version = validate_version_argument(version, self._mdf.default_version)

        out = MDF(version=version)

        out._mdf.header.start_time = self._mdf.header.start_time

        groups_nr = len(self._mdf.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    return TERMINATED

        cg_nr = None

        self.configure(copy_on_get=False)

        # walk through all groups and get all channels
        for i, virtual_group in enumerate(self._mdf.virtual_groups):
            selected_signals = self._mdf._yield_selected_signals(virtual_group, version=version)
            sigs = typing.cast(Optional[list[Signal]], next(selected_signals, None))
            if sigs:
                t = sigs[0].timestamps
                if len(t):
                    all_ok, indexes = plausible_timestamps(t, minimum, maximum, exp_min, exp_max)
                    if not all_ok:
                        t = t[indexes]
                        if len(t):
                            for sig in sigs:
                                sig.samples = sig.samples[indexes]
                                sig.timestamps = t
                                if sig.invalidation_bits is not None:
                                    sig.invalidation_bits = sig.invalidation_bits[indexes]
                cg = self._mdf.groups[virtual_group].channel_group
                cg_nr = out.append(
                    sigs,
                    acq_name=getattr(cg, "acq_name", None),
                    acq_source=getattr(cg, "acq_source", None),
                    comment=f"Timestamps cleaned up and converted from {self._mdf.version} to {version}",
                    common_timebase=True,
                )
            else:
                continue
            selected_samples = typing.cast(
                Iterator[list[tuple[NDArray[Any], Optional[NDArray[Any]]]]], selected_signals
            )
            for signal_samples in selected_samples:
                t, _ = signal_samples[0]
                if len(t):
                    all_ok, indexes = plausible_timestamps(t, minimum, maximum, exp_min, exp_max)
                    if not all_ok:
                        t = t[indexes]
                        if len(t):
                            for i, (samples, invalidation_bits) in enumerate(signal_samples):
                                if invalidation_bits is not None:
                                    invalidation_bits = invalidation_bits[indexes]
                                samples = samples[indexes]

                                signal_samples[i] = (samples, invalidation_bits)

                out._mdf.extend(cg_nr, signal_samples)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        return TERMINATED

        out._transfer_metadata(self)
        self.configure(copy_on_get=True)

        return out

    def whereis(
        self,
        channel: str,
        source_name: str | None = None,
        source_path: str | None = None,
        acq_name: str | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """get occurrences of channel name in the file

        Parameters
        ----------
        channel : str
            channel name string
        source_name : str, optional
            filter occurrences on source name, by default None
        source_path : str, optional
            filter occurrences on source path, by default None
        acq_name : str, optional
            filter occurrences on channel group acquisition name, by default None

            .. versionadded:: 6.0.0

        Returns
        -------
        tuple[tuple[int, int], ...]
            (gp_idx, cn_idx) pairs

        Examples
        --------
        >>> mdf = MDF(file_name)
        >>> mdf.whereis('VehicleSpeed') # "VehicleSpeed" exists in the file
        ((1, 2), (2, 4))
        >>> mdf.whereis('VehicleSPD') # "VehicleSPD" doesn't exist in the file
        ()
        """
        occurrences = tuple(
            self._mdf._filter_occurrences(
                self._mdf.channels_db.get(channel, []),
                source_name=source_name,
                source_path=source_path,
                acq_name=acq_name,
            )
        )
        return occurrences

    def search(
        self,
        pattern: str,
        mode: Literal["plain", "regex", "wildcard"] | SearchMode = SearchMode.plain,
        case_insensitive: bool = False,
    ) -> list[str]:
        """search channels

        .. versionadded:: 7.0.0

        Parameters
        ----------
        pattern : str
            search pattern
        mode : Literal["plain", "regex", "wildcard"] or SearchMode, optional
            search mode, by default SearchMode.plain

                * `plain` : normal name search
                * `regex` : regular expression based search
                * `wildcard` : wildcard based search
        case_insensitive : bool, optional
            case sensitivity for the channel name search, by default False

        Returns
        -------
        list[str]
            name of the channels

        Raises
        ------
        ValueError
            unsupported search mode

        Examples
        --------
        >>> mdf = MDF(file_name)
        >>> mdf.search('*veh*speed*', case_insensitive=True, mode='wildcard') # case insensitive wildcard based search
        ['vehicleAverageSpeed', 'vehicleInstantSpeed', 'targetVehicleAverageSpeed', 'targetVehicleInstantSpeed']
        >>> mdf.search('^vehicle.*Speed$', case_insensitive=False, mode='regex') # case sensitive regex based search
        ['vehicleAverageSpeed', 'vehicleInstantSpeed']
        """
        search_mode = SearchMode(mode)

        if search_mode is SearchMode.plain:
            if case_insensitive:
                pattern = pattern.casefold()
                channels = [name for name in self._mdf.channels_db if pattern in name.casefold()]
            else:
                channels = [name for name in self._mdf.channels_db if pattern in name]
        elif search_mode is SearchMode.regex:
            flags = re.IGNORECASE if case_insensitive else 0
            compiled_pattern = re.compile(pattern, flags=flags)
            channels = [name for name in self._mdf.channels_db if compiled_pattern.search(name)]
        elif search_mode is SearchMode.wildcard:
            wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
            pattern = pattern.replace("*", wildcard)
            pattern = re.escape(pattern)
            pattern = pattern.replace(wildcard, ".*")

            flags = re.IGNORECASE if case_insensitive else 0

            compiled_pattern = re.compile(pattern, flags=flags)

            channels = [name for name in self._mdf.channels_db if compiled_pattern.search(name)]

        else:
            raise ValueError(f"unsupported mode {search_mode}")

        return channels

    def _asc_export(self, file_name):
        if isinstance(self._mdf, (mdf_v2.MDF2, mdf_v3.MDF3)):
            return

        groups_count = len(self._mdf.groups)

        dfs = []

        for index in range(groups_count):
            group = self._mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_CAN:
                    if "CAN_DataFrame" in names:
                        data = self._mdf.get("CAN_DataFrame", index)  # , raw=True)

                    elif "CAN_RemoteFrame" in names:
                        data = self._mdf.get("CAN_RemoteFrame", index, raw=True)

                    elif "CAN_ErrorFrame" in names:
                        data = self._mdf.get("CAN_ErrorFrame", index, raw=True)

                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "type": np.full(count, "CAN", dtype="O"),
                        "Bus": np.zeros(count, dtype="u1"),
                        "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                        "IDE": np.zeros(count, dtype="u1"),
                        "Direction": np.full(count, "Rx", dtype="O"),
                        "Event Type": np.full(count, "CAN Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "ESI": np.zeros(count, dtype="u1"),
                        "EDL": np.zeros(count, dtype="u1"),
                        "BRS": np.zeros(count, dtype="u1"),
                        "DLC": np.zeros(count, dtype="u1"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                        "Name": np.full(count, "", dtype="O"),
                    }

                    for string in v4c.CAN_ERROR_TYPES.values():
                        sys.intern(string)

                    frame_map = None
                    if data.attachment and data.attachment[0]:
                        dbc = load_can_database(data.attachment[1], data.attachment[0])
                        if dbc:
                            frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                            for name in frame_map.values():
                                sys.intern(name)

                    if data.name == "CAN_DataFrame":
                        columns["Bus"] = data["CAN_DataFrame.BusChannel"].astype("u1")

                        vals = data["CAN_DataFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        columns["DLC"] = data["CAN_DataFrame.DLC"].astype("u1")
                        columns["Data Length"] = data["CAN_DataFrame.DataLength"].astype("u1")

                        vals = csv_bytearray2hex(
                            pd.Series(list(data["CAN_DataFrame.DataBytes"])),
                            columns["Data Length"],
                        )
                        columns["Data Bytes"] = vals

                        if "CAN_DataFrame.Dir" in names:
                            if data["CAN_DataFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize() for v in data["CAN_DataFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_DataFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_DataFrame.ESI" in names:
                            columns["ESI"] = data["CAN_DataFrame.ESI"].astype("u1")

                        if "CAN_DataFrame.EDL" in names:
                            columns["EDL"] = data["CAN_DataFrame.EDL"].astype("u1")

                        if "CAN_DataFrame.BRS" in names:
                            columns["BRS"] = data["CAN_DataFrame.BRS"].astype("u1")

                        if "CAN_DataFrame.IDE" in names:
                            columns["IDE"] = data["CAN_DataFrame.IDE"].astype("u1")

                    elif data.name == "CAN_RemoteFrame":
                        columns["Bus"] = data["CAN_RemoteFrame.BusChannel"].astype("u1")

                        vals = data["CAN_RemoteFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        columns["DLC"] = data["CAN_RemoteFrame.DLC"].astype("u1")
                        columns["Data Length"] = data["CAN_RemoteFrame.DataLength"].astype("u1")
                        columns["Event Type"] = "Remote Frame"

                        if "CAN_RemoteFrame.Dir" in names:
                            if data["CAN_RemoteFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize() for v in data["CAN_RemoteFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_RemoteFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_RemoteFrame.IDE" in names:
                            columns["IDE"] = data["CAN_RemoteFrame.IDE"].astype("u1")

                    elif data.name == "CAN_ErrorFrame":
                        names = set(data.samples.dtype.names)

                        if "CAN_ErrorFrame.BusChannel" in names:
                            columns["Bus"] = data["CAN_ErrorFrame.BusChannel"].astype("u1")

                        if "CAN_ErrorFrame.ID" in names:
                            vals = data["CAN_ErrorFrame.ID"].astype("u4") & 0x1FFFFFFF
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_ErrorFrame.DLC" in names:
                            columns["DLC"] = data["CAN_ErrorFrame.DLC"].astype("u1")

                        if "CAN_ErrorFrame.DataLength" in names:
                            columns["Data Length"] = data["CAN_ErrorFrame.DataLength"].astype("u1")

                        columns["Event Type"] = "Error Frame"

                        if "CAN_ErrorFrame.ErrorType" in names:
                            vals = data["CAN_ErrorFrame.ErrorType"].astype("u1").tolist()
                            vals = [v4c.CAN_ERROR_TYPES.get(err, "Other error") for err in vals]

                            columns["Details"] = vals

                        if "CAN_ErrorFrame.Dir" in names:
                            if data["CAN_ErrorFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize() for v in data["CAN_ErrorFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_ErrorFrame.Dir"].astype("u1").tolist()
                                ]

                    dfs.append(pd.DataFrame(columns, index=df_index))

                elif source and source.bus_type == v4c.BUS_TYPE_FLEXRAY:
                    if "FLX_Frame" in names:
                        data = self._mdf.get("FLX_Frame", index, raw=True)

                    elif "FLX_NullFrame" in names:
                        data = self._mdf.get("FLX_NullFrame", index, raw=True)

                    elif "FLX_StartCycle" in names:
                        data = self._mdf.get("FLX_StartCycle", index, raw=True)

                    elif "FLX_Status" in names:
                        data = self._mdf.get("FLX_Status", index, raw=True)
                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "type": np.full(count, "FLEXRAY", dtype="O"),
                        "Bus": np.zeros(count, dtype="u1"),
                        "ID": np.full(count, 0xFFFF, dtype="u2"),
                        "ControllerFlags": np.zeros(count, dtype="u2"),
                        "FrameFlags": np.zeros(count, dtype="u4"),
                        "Direction": np.full(count, "Rx", dtype="O"),
                        "Cycle": np.full(count, 0xFF, dtype="u1"),
                        "Event Type": np.full(count, "FlexRay Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Payload Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                        "Header CRC": np.full(count, 0xFFFF, dtype="u2"),
                    }

                    if data.name == "FLX_Frame":
                        columns["Bus"] = data["FLX_Frame.FlxChannel"].astype("u1")
                        columns["ID"] = data["FLX_Frame.ID"].astype("u2")
                        columns["Cycle"] = data["FLX_Frame.Cycle"].astype("u1")
                        columns["Data Length"] = data["FLX_Frame.DataLength"].astype("u1")
                        columns["Payload Length"] = data["FLX_Frame.PayloadLength"].astype("u1") * 2

                        vals = csv_bytearray2hex(
                            pd.Series(list(data["FLX_Frame.DataBytes"])),
                            columns["Data Length"],
                        )
                        columns["Data Bytes"] = vals

                        columns["Header CRC"] = data["FLX_Frame.HeaderCRC"].astype("u2")

                        if "FLX_Frame.Dir" in names:
                            if data["FLX_Frame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize() for v in data["FLX_Frame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["FLX_Frame.Dir"].astype("u1").tolist()
                                ]

                        if "FLX_Frame.ControllerFlags" in names:
                            columns["ControllerFlags"] = np.frombuffer(
                                data["FLX_Frame.ControllerFlags"].tobytes(), dtype="<u2"
                            )
                        if "FLX_Frame.FrameFlags" in names:
                            columns["FrameFlags"] = np.frombuffer(data["FLX_Frame.FrameFlags"].tobytes(), dtype="<u4")

                    elif data.name == "FLX_NullFrame":
                        columns["Bus"] = data["FLX_NullFrame.FlxChannel"].astype("u1")
                        columns["ID"] = data["FLX_NullFrame.ID"].astype("u2")
                        columns["Cycle"] = data["FLX_NullFrame.Cycle"].astype("u1")

                        columns["Event Type"] = "FlexRay NullFrame"
                        columns["Header CRC"] = data["FLX_NullFrame.HeaderCRC"].astype("u2")

                        if "FLX_NullFrame.Dir" in names:
                            if data["FLX_NullFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize() for v in data["FLX_NullFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["FLX_NullFrame.Dir"].astype("u1").tolist()
                                ]

                    elif data.name == "FLX_StartCycle":
                        columns["Cycle"] = data["FLX_StartCycle.Cycle"].astype("u1")
                        columns["Event Type"] = "FlexRay StartCycle"

                    elif data.name == "FLX_Status":
                        vals = data["FLX_Status.StatusType"].astype("u1")
                        columns["Details"] = vals.astype("U").astype("O")

                        columns["Event Type"] = "FlexRay Status"

                    dfs.append(pd.DataFrame(columns, index=df_index))

        if dfs:
            signals = pd.concat(dfs).sort_index()

            index = pd.Index(range(len(signals)))
            signals.set_index(index, inplace=True)
        else:
            signals = pd.DataFrame()

        with open(file_name, "w") as asc:
            start = self.start_time.strftime("%a %b %d %I:%M:%S.%f %p %Y")
            asc.write(f"date {start}\n")
            asc.write("base hex  timestamps absolute\n")
            asc.write("no internal events logged\n")

            for row in signals.to_dict("records"):
                if row["type"] == "CAN":
                    if row["Event Type"] == "CAN Frame":
                        if row["EDL"]:
                            data = row["Data Bytes"].lower()
                            id = f"{row['ID']:x}"

                            if row["IDE"]:
                                id = f"{id}x"

                            bus = row["Bus"]
                            dir = row["Direction"]
                            t = row["timestamps"]

                            flags = 1 << 12
                            brs = row["BRS"]
                            if brs:
                                flags |= 1 << 13

                            esi = row["ESI"]
                            if esi:
                                flags |= 1 << 14

                            name = row["Name"]
                            dlc = row["DLC"]
                            data_length = row["Data Length"]

                            asc.write(
                                f"   {t: 9.6f} CANFD {bus:>3} {dir:<4} {id:>8}  {name:>32} {brs} {esi} {dlc:x} {data_length:>2} {data}        0    0 {flags:>8x}        0        0        0        0        0\n"
                            )

                        else:
                            dlc = row["DLC"]
                            data = row["Data Bytes"]
                            id = f"{row['ID']:x}"

                            if row["IDE"]:
                                id = f"{id}x"

                            bus = row["Bus"]
                            dir = row["Direction"]
                            t = row["timestamps"]

                            asc.write(f"{t: 9.6f} {bus}  {id:<15} {dir:<4} d {dlc:x} {data}\n")

                    elif row["Event Type"] == "Error Frame":
                        asc.write(f"   {row['timestamps']: 9.6f} {row['Bus']} ErrorFrame\n")

                    elif row["Event Type"] == "Remote Frame":
                        dlc = row["DLC"]
                        id = f"{row['ID']:x}"

                        if row["IDE"]:
                            id = f"{id}x"

                        bus = row["Bus"]
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(f"   {t: 9.6f} {bus}  {id:<15} {dir:<4} r {dlc:x}\n")

                elif row["type"] == "FLEXRAY":
                    if row["Event Type"] == "FlexRay Frame":
                        frame_flags = f'{row["FrameFlags"]:x}'
                        controller_flags = f'{row["ControllerFlags"]:x}'
                        data = row["Data Bytes"]
                        header_crc = f'{row["Header CRC"]:x}'
                        data_length = f'{row["Data Length"]:x}'
                        payload_length = f'{row["Payload Length"]:x}'
                        bus = f'{row["Bus"] + 1:x}'
                        slot = f'{row["ID"]:x}'
                        cycle = f'{row["Cycle"]:x}'
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(
                            f"   {t: 9.6f} Fr RMSG  0 0 1 {bus} {slot} {cycle} {dir} 0 {frame_flags} 5  {controller_flags}  {header_crc} x {payload_length} {data_length} {data} 0  0  0\n"
                        )

                    elif row["Event Type"] == "FlexRay NullFrame":
                        frame_flags = f'{row["FrameFlags"]:x}'
                        controller_flags = f'{row["ControllerFlags"]:x}'
                        header_crc = f'{row["Header CRC"]:x}'
                        payload_length = f'{row["Payload Length"]:x}'
                        bus = f'{row["Bus"] + 1:x}'
                        slot = f'{row["ID"]:x}'
                        cycle = f'{row["Cycle"]:x}'
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(
                            f"   {t: 9.6f} Fr RMSG  0 0 1 {bus} {slot} {cycle} {dir} 0 {frame_flags} 5  {controller_flags}  {header_crc} x {payload_length} 0 0  0  0\n"
                        )


if __name__ == "__main__":
    pass
