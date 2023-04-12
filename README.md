# Spike-based embeddings for multi-relational graph data

Code for learning spike-based knowledge graph embeddings described in [SpikE: spike-based embeddings for multi-relational graph data (IJCNN 2021)](https://arxiv.org/abs/2104.13398) and extended to spike-based (relational) graph convolutional networks in [Learning through structure: towards deep neuromorphic knowledge graph embeddings (ICNC 2021)](https://arxiv.org/abs/2109.10376). See also [Neuro-symbolic computing with spiking neural networks](https://arxiv.org/abs/2208.02576) for a summary of this work and follow-up work.

## Introduction

We present a model that enables learning of spike-based representations of abstract concepts and their relationships, which can subsequently be used to reason in the underlying semantic space. We approach this problem from the perspective of knowledge graphs (KGs). KGs are a widely used, rich data structure that enables a symbolic description of abstract concepts and how they relate to each other. In general, a KG is summarized as a list of triple statements (s,p,o) stating that a relationship p holds between two entities s and o. We propose a spike-based algorithm where entities in a KG are represented by single spike times of neuron populations and relations as spike time differences between populations. The obtained spike-based representations can be used to perform inference on the semantic space spanned by the KG (symbolic reasoning), e.g., to evaluate unknown facts. Learning such spike-based embeddings only requires knowledge about spike times and spike time differences, compatible with recently proposed frameworks for training spiking neural networks.

## Code

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

## Citation

If you use parts of this repository for your own work or build upon the presented spike-based graph embedding method, please cite

```
@inproceedings{dold2021spike,
  title={Spike: Spike-based embeddings for multi-relational graph data},
  author={Dold, Dominik and Garrido, Josep Soler},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

and

```
@inproceedings{chian2021learning,
  title={Learning through structure: towards deep neuromorphic knowledge graph embeddings},
  author={Chian, Victor Caceres and Hildebrandt, Marcel and Runkler, Thomas and Dold, Dominik},
  booktitle={2021 International Conference on Neuromorphic Computing (ICNC)},
  pages={61--70},
  year={2021},
  organization={IEEE}
}
```
