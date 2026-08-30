import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="winning",
    version="1.3.0",
    description="Dealing with races, correlated or not",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/winning",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    packages=["winning", "winning.classic", "winning.factor", "winning.methods", "winning.bench", "winning.thurstone", "winning.probit", "winning.ratings"],
    test_suite='pytest',
    tests_require=['pytest','pandas','scipy>=1.7.3','randomcov'],
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={"test": ["pytest", "pandas", "matplotlib"],
                    "benchmarks": ["pandas"],
                    # compiled kernels (rust/fastrace); pure python without
                    "fast": ["fastrace"]},
    entry_points={
        "console_scripts": [
            "winning=winning.__main__:main",
        ]
    },
)
