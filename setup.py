import os

import numpy
from setuptools import Extension, setup

NUMPY_INCLUDE = numpy.get_include()


def _has_prebuilt_windows_extensions() -> bool:
    """Return whether both Cython extensions already exist for this Python."""
    if os.name != "nt":
        return False
    extension_dir = os.path.join(os.path.dirname(__file__), "qlib", "data", "_libs")
    return all(
        any(name.startswith(f"{module}.") and name.endswith(".pyd") for name in os.listdir(extension_dir))
        for module in ("rolling", "expanding")
    )


EXT_MODULES = [] if _has_prebuilt_windows_extensions() else [
    Extension(
        "qlib.data._libs.rolling",
        ["qlib/data/_libs/rolling.pyx"],
        language="c++",
        include_dirs=[NUMPY_INCLUDE],
    ),
    Extension(
        "qlib.data._libs.expanding",
        ["qlib/data/_libs/expanding.pyx"],
        language="c++",
        include_dirs=[NUMPY_INCLUDE],
    ),
]


setup(
    ext_modules=EXT_MODULES,
)
