from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ramtorch",
    version="0.2.1.dev0",
    author="Lodestone",
    author_email="lodestone.rock@gmail.com",
    description="RAM is All You Need",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/lodestone-rock/RamTorch",
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=["torch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
