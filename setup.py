from setuptools import find_packages
from setuptools import setup


setup(
    name="torch_stft",
    version="0.0.1",
    author="Wangyou Zhang",
    author_email="wyz-97@sjtu.edu.cn",
    description="PyTorch-based implementations of STFT and iSTFT",
    url="https://github.com/Emrys365/torch_stft",
    license="Apache-2.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["torch>=1.0.0"],
    python_requires=">=3.6",
    tests_require=["numpy", "pytest", "soundfile"],
)
