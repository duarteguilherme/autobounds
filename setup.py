import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autobound-gjduarte", # Replace with your own username
    version="0.0.2",
    author="Guilherme Duarte, Dean Knox, Jonathan Mummolo",
    author_email="gjardimduarte@gmail.com",
    description="AUTOBOUND -- software for calculating causal bounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
