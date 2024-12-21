"""
ASAM MDF version 4 file format module
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from io import StringIO
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Required, TypedDict

from ..types import ChannelType, DbcFileType, MDF_v2_v3_v4, StrPathType
from . import v2_v3_blocks, v4_blocks
from .utils import DataBlockInfo, EMPTY_TUPLE, MdfException, SignalDataBlockInfo

logger = logging.getLogger("asammdf")

__all__ = ["MDF_Common"]


class MdfKwargs(TypedDict, total=False):
    temporary_folder: Optional[Union[str, PathLike[str]]]
    raise_on_multiple_occurrences: bool
    use_display_names: bool
    fill_0_for_missing_computation_channels: bool
    remove_source_from_channel_names: bool
    password: Optional[str]
    progress: Union[Callable[[int, int], None], Any]
    callback: Union[Callable[[int, int], None], Any]


class CommonKwargs(MdfKwargs, total=False):
    original_name: Required[Optional[Union[str, Path]]]
    __internal__: bool


class CanBusInfo(TypedDict):
    dbc_files: Iterable[DbcFileType]
    total_unique_ids: set[tuple[int, bool]]
    unknown_id_count: int
    not_found_ids: defaultdict[StrPathType, list[tuple[tuple[int, bool] | int, str]]]
    found_ids: defaultdict[StrPathType, set[tuple[tuple[int, int, bool], str]]]
    unknown_ids: set[int | tuple[int, bool]]


class LinBusInfo(TypedDict):
    dbc_files: Iterable[DbcFileType]
    total_unique_ids: set[tuple[int, ...]]
    unknown_id_count: int
    not_found_ids: defaultdict[StrPathType, list[tuple[int, str]]]
    found_ids: defaultdict[StrPathType, set[tuple[tuple[int, bool, bool], str]]]
    unknown_ids: set[int]


class BusInfo(TypedDict, total=False):
    CAN: CanBusInfo
    LIN: LinBusInfo


_DG = TypeVar("_DG", v2_v3_blocks.DataGroup, v4_blocks.DataGroup)
_CG = TypeVar("_CG", v2_v3_blocks.ChannelGroup, v4_blocks.ChannelGroup)
_CN = TypeVar("_CN", v2_v3_blocks.Channel, v4_blocks.Channel)


class Group(Generic[_DG, _CG, _CN]):
    __slots__ = (
        "channel_dependencies",
        "channel_group",
        "channels",
        "data_blocks",
        "data_blocks_info_generator",
        "data_group",
        "data_location",
        "index",
        "read_split_count",
        "record",
        "record_size",
        "signal_data",
        "signal_types",
        "single_channel_dtype",
        "sorted",
        "string_dtypes",
        "trigger",
        "uses_ld",
        "uuid",
    )

    def __init__(self, data_group: _DG) -> None:
        self.data_group = data_group
        self.channel_group: _CG
        self.channels: list[_CN] = []
        self.channel_dependencies = []
        self.signal_data = []
        self.record: list[Optional[tuple[np.dtype[Any], int, int, int]]] = []
        self.record_size: dict[int, int] = {}
        self.trigger: Optional[v2_v3_blocks.TriggerBlock] = None
        self.sorted: bool
        self.string_dtypes: list[str] = []
        self.data_blocks = []
        self.signal_types: list[int]
        self.single_channel_dtype = None
        self.uses_ld = False
        self.read_split_count = 0
        self.data_blocks_info_generator = iter(EMPTY_TUPLE)
        self.uuid = ""
        self.data_location: int
        self.index = 0

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def set_blocks_info(self, info: list[DataBlockInfo]) -> None:
        self.data_blocks = info

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def clear(self) -> None:
        self.data_blocks.clear()
        self.channels.clear()
        self.channel_dependencies.clear()
        self.signal_data.clear()
        self.data_blocks_info_generator = None

    def get_data_blocks(self) -> Iterator[DataBlockInfo]:
        yield from self.data_blocks

        while True:
            try:
                info = next(self.data_blocks_info_generator)
                self.data_blocks.append(info)
                yield info
            except StopIteration:
                break

    def get_signal_data_blocks(self, index: int) -> Iterator[SignalDataBlockInfo]:
        signal_data = self.signal_data[index]
        if signal_data is not None:
            signal_data, signal_generator = signal_data
            yield from signal_data

            while True:
                try:
                    info = next(signal_generator)
                    signal_data.append(info)
                    yield info
                except StopIteration:
                    break


def debug_channel(
    mdf: MDF_v2_v3_v4,
    group: Group,
    channel: ChannelType,
    dependency: list[tuple[int, int]],
    file: StringIO | None = None,
) -> None:
    """use this to print debug information in case of errors

    Parameters
    ----------
    mdf : MDF
        source MDF object
    group : dict
        group
    channel : Channel
        channel object
    dependency : ChannelDependency
        channel dependency object

    """
    print("MDF", "=" * 76, file=file)
    print("name:", mdf.name, file=file)
    print("version:", mdf.version, file=file)
    print("read fragment size:", mdf._read_fragment_size, file=file)
    print("write fragment size:", mdf._write_fragment_size, file=file)
    print()

    record = mdf._prepare_record(group)
    print("GROUP", "=" * 74, file=file)
    print("sorted:", group["sorted"], file=file)
    print("data location:", group["data_location"], file=file)
    print("data blocks:", group.data_blocks, file=file)
    print("dependencies", group["channel_dependencies"], file=file)
    print("record:", record, file=file)
    print(file=file)

    cg = group["channel_group"]
    print("CHANNEL GROUP", "=" * 66, file=file)
    print(cg, cg.cycles_nr, cg.samples_byte_nr, cg.invalidation_bytes_nr, file=file)
    print(file=file)

    print("CHANNEL", "=" * 72, file=file)
    print(channel, file=file)
    print(file=file)

    print("CHANNEL ARRAY", "=" * 66, file=file)
    print(dependency, file=file)
    print(file=file)


class MDF_Common:
    """common methods for MDF objects"""

    def _set_temporary_master(self, master: NDArray[Any] | None) -> None:
        self._master = master

    # @lru_cache(maxsize=1024)
    def _validate_channel_selection(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> tuple[int, int]:
        """Gets channel comment.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurrences for this channel then the
            *group* and *index* arguments can be used to select a specific
            group.
            * if there are multiple occurrences for this channel and either the
            *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
        number (keyword argument *index*). Use *info* method for group and
        channel numbers


        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        group_index, channel_index : (int, int)
            selected channel's group and channel index

        """

        if name is None:
            if group is None or index is None:
                message = "Invalid arguments for channel selection: " 'must give "name" or, "group" and "index"'
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if ch_nr >= 0:
                    try:
                        grp = self.groups[gp_nr]
                    except IndexError:
                        raise MdfException("Group index out of range") from None

                    try:
                        grp.channels[ch_nr]
                    except IndexError:
                        raise MdfException(f"Channel index out of range: {(name, group, index)}") from None
        else:
            if name not in self.channels_db:
                raise MdfException(f'Channel "{name}" not found')
            else:
                if group is None:
                    entries = self.channels_db[name]
                    if len(entries) > 1:
                        if self._raise_on_multiple_occurrences:
                            message = (
                                f'Multiple occurrences for channel "{name}": {entries}. '
                                'Provide both "group" and "index" arguments'
                                " to select another data group"
                            )
                            logger.exception(message)
                            raise MdfException(message)
                        else:
                            message = (
                                f'Multiple occurrences for channel "{name}": {entries}. '
                                "Returning the first occurrence since the MDF object was "
                                "configured to not raise an exception in this case."
                            )
                            logger.warning(message)
                            gp_nr, ch_nr = entries[0]
                    else:
                        gp_nr, ch_nr = entries[0]

                else:
                    if index is not None and index < 0:
                        gp_nr = group
                        ch_nr = index
                    else:
                        if index is None:
                            entries = tuple((gp_nr, ch_nr) for gp_nr, ch_nr in self.channels_db[name] if gp_nr == group)
                            count = len(entries)

                            if count == 1:
                                gp_nr, ch_nr = entries[0]

                            elif count == 0:
                                message = f'Channel "{name}" not found in group {group}'
                                raise MdfException(message)

                            else:
                                if self._raise_on_multiple_occurrences:
                                    message = (
                                        f'Multiple occurrences for channel "{name}": {entries}. '
                                        'Provide both "group" and "index" arguments'
                                        " to select another data group"
                                    )
                                    logger.exception(message)
                                    raise MdfException(message)
                                else:
                                    message = (
                                        f'Multiple occurrences for channel "{name}": {entries}. '
                                        "Returning the first occurrence since the MDF object was "
                                        "configured to not raise an exception in this case."
                                    )
                                    logger.warning(message)
                                    gp_nr, ch_nr = entries[0]
                        else:
                            if (group, index) in self.channels_db[name]:
                                ch_nr = index
                                gp_nr = group
                            else:
                                message = f'Channel "{name}" not found in group {group} at index {index}'
                                raise MdfException(message)

        return gp_nr, ch_nr
