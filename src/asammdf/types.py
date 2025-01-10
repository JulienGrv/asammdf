from collections.abc import Sequence
from io import BufferedRandom, BufferedWriter, BytesIO
from mmap import mmap
from os import PathLike
from typing import Any, Optional, TYPE_CHECKING, Union

from canmatrix import CanMatrix
from numpy.typing import NDArray
from typing_extensions import Literal

if TYPE_CHECKING:
    from .blocks import v2_v3_blocks as v3b
    from .blocks import v4_blocks as v4b
    from .blocks.source_utils import Source


StrPathType = Union[str, "PathLike[str]"]
StrOrBytesPathType = Union[str, bytes, "PathLike[str]", "PathLike[bytes]"]
WritableBufferType = Union[BufferedRandom, BufferedWriter, BytesIO, mmap]

# asammdf specific types

BusType = Literal["CAN", "LIN"]
ChannelConversionType = Union["v3b.ChannelConversion", "v4b.ChannelConversion"]
ChannelGroupType = Union["v3b.ChannelGroup", "v4b.ChannelGroup"]
ChannelsType = Sequence[Union[str, tuple[Optional[str], int, int], tuple[str, int]]]
CompressionType = Literal[0, 1, 2]
DbcFileType = tuple[Union[StrPathType, CanMatrix], int]
EmptyChannelsType = Literal["skip", "zeros"]
FloatInterpolationModeType = Literal[0, 1]
IntInterpolationModeType = Literal[0, 1, 2]
RasterType = Union[float, str, NDArray[Any]]
SourceType = Union["v3b.ChannelExtension", "v4b.SourceInformation", "Source"]
SyncType = Literal[0, 1, 2, 3, 4]
