# Spike-based embeddings for multi-relational graph data

This package contains the initial SpikE implementation with some examples.
To install the package, use

`pip install -e .`

after which the package can be imported in Python using

`import spikee`

## Data

The data folder includes the countries dataset (https://github.com/ZhenfengLei/KGDatasets/tree/master/Countries) and graph data generated from an industrial automation demonstrator.
New datasets have to be split into a training, validation and test file (train.txt, valid.txt, test.txt), with each file containing named triples

`<subject1>\t<hasRelation>\t<object1>` .

Before using the dataset, preprocessing is required (this has been taken from LibKGE https://github.com/uma-pi1/kge)

`cd data`

`python preprocess_default.py <your_folder>` .

## Experiments

Jupyter notebooks with example experiments illustrating the model. We show three experiments here:

- **Countries dataset:** Learning spike-based embeddings for the Countries_S1 dataset. The countries dataset encodes countries and continents as nodes and contains the two relations 'neighbor' and 'locatedin'.
- **Industrial automation 1:** Learning graph data generated from an industrial automation demonstrator.
- **Industrial automation 2:** A simple anomaly evaluation task derived from the industrial automation data.
