import os
from pathlib import Path

from setuptools import find_packages, setup

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "0"
_PATCH_VERSION = "1"

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as fp:
    install_requires = fp.read().split("\n")


def glob_fix(package_name, glob):
    package_path = Path(os.path.join(os.path.dirname(__file__), package_name)).resolve()
    return [str(path.relative_to(package_path)) for path in package_path.glob(glob)]


setup(
    name="grax",
    description="Graph Networks with Jax",
    url="https://github.com/jackd/grax",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={"grax": glob_fix("grax", "**/*.gin")},
    install_requires=install_requires,
    zip_safe=True,
    python_requires=">=3.6",
    version=".".join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION]),
)
