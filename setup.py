import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spikee",
    version="1.0.0",
    author="Dominik Dold",
    author_email="dodo.science@web.de",
    description="spike-based graph embedding model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.7',
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'torch >= 1.6.0',
          'jupyter'
      ]
)
