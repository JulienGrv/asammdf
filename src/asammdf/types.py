from collections.abc import Sequence
from io import BufferedRandom, BufferedWriter, BytesIO
from mmap import mmap
from os import PathLike
from typing import Any, Optional, TYPE_CHECKING, Union

from canmatrix import CanMatrix
from numpy.typing import NDArray
from typing_extensions import Literal

if TYPE_CHECKING:
    from .blocks import v2_v3_blocks, v4_blocks
    from .blocks.source_utils import Source


StrPathType = Union[str, "PathLike[str]"]
StrOrBytesPathType = Union[str, bytes, "PathLike[str]", "PathLike[bytes]"]
WritableBufferType = Union[BufferedRandom, BufferedWriter, BytesIO, mmap]

# asammdf specific types

BusType = Literal["CAN", "LIN"]
ChannelConversionType = Union["v2_v3_blocks.ChannelConversion", "v4_blocks.ChannelConversion"]
ChannelGroupType = Union["v2_v3_blocks.ChannelGroup", "v4_blocks.ChannelGroup"]
ChannelsType = Sequence[Union[str, tuple[Optional[str], int, int], tuple[str, int]]]
CompressionType = Literal[0, 1, 2]
DbcFileType = tuple[Union[StrPathType, CanMatrix], int]
EmptyChannelsType = Literal["skip", "zeros"]
FloatInterpolationModeType = Literal[0, 1]
IntInterpolationModeType = Literal[0, 1, 2]
RasterType = Union[float, str, NDArray[Any]]
SourceType = Union["v2_v3_blocks.ChannelExtension", "v4_blocks.SourceInformation", "Source"]
SyncType = Literal[0, 1, 2, 3, 4]
