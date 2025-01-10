""" ASAM MDF version 2 file format module """

from os import PathLike
from typing import Optional, Union

from .mdf_v3 import MDF3, Version
from .utils import FileLike, MdfException, validate_version_argument

__all__ = ["MDF2"]


# MDF versions 2 and 3 share the same implementation
class MDF2(MDF3):
    """shared implementation for mdf version 2 and 3"""

    default_version = "2.14"

    def __init__(
        self,
        name: Optional[Union[str, PathLike[str], FileLike]] = None,
        version: Version = default_version,
        **kwargs,
    ) -> None:
        version = validate_version_argument(version, hint=self.default_version)

        if not kwargs.get("__internal__", False):
            raise MdfException("Always use the MDF class; do not use the class MDF2 directly")

        super().__init__(name, version, **kwargs)


if __name__ == "__main__":
    pass
