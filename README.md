 Machine Learning Engineer Nanodegree

### Castone Project
## Telstra Network Disruptions

### 1. Project Overview

Information and Communications Technology (ICT), including the Internet and smartphones, is playing an important role in our modern society. Most of our daily activities are supported by the swift and diverse communication through the networks. Therefore, reliable network management is crucial. When the network suffers any disruption or fault, the company providing the network access needs to fix it as soon as possible. However, as the networks grow larger and more complex along with the development of our communication environment, solving network fault becomes more difficult. To satisfy customers’ expectation in this challenging situation, telecommunications companies started to turn their attention to the latest big data and machine learning techniques. They expect to predict future faults by making use of data from their networks and avoid serious fault or disruption before it actually affects customers.

As a solution of this problem, I propose a model that predicts the level of fault severity at a time and a location from network log data. To build the predictive model, I use a dataset provided originally by Telstra, Australia's largest telecommunications company. In Telstra's dataset, fault severity is indicated by three levels (0 means no fault, 1 means only a few, and 2 means many). Thus the model is supposed to perform multi-class classification task where each data point that represents a certain location-and-time is classified into one of three classes of fault severity.

In this project, I examine three Supervised Learning algorithms including Deep Neural Networks, with two different data encodings. I also apply two types of tequniques to refine the model.

The detail of this project is thoroughly described in the project report `Telstra_project_report.pdf`.


### 2. Dataset

The dataset I used in this project consists of 6 csv files shown below:

 - `train.csv`: the main dataset for fault severity with respect to a location and time
 - `event_type.csv`: event type number related to the main dataset
 - `log_feature.csv`: feature number and volume extracted from log files
 - `resource_type.csv`: type of resource related to the main dataset
 - `severity_type.csv`: severity type of a warning message coming from the log
 - `test.csv`: the test dataset


These data files were obtained from Kaggle's past compation page:
 "[Telstra Network Disruptions](https://www.kaggle.com/c/telstra-recruiting-network)". They were originally provided by Telstra.


### 3. Libraries and Packages
This project requires **Python 3** and **Jupyter Notebook**. Other libraries and packages that are necessary to run this project are listed below:
- numpy (1.14.3)
- Pandas (0.23.0)
- sklearn (0.19.1)
- Keras (2.2.2)
- Tensorflow (1.10.0)
- matplotlib (2.2.2)
- seaborn (0.9.0)
- [lightgbm (2.2.2)](https://github.com/Microsoft/LightGBM/tree/master/python-package) : for LightGBM model




### 4. Contents of this repository

#### Documents
- `Telstra_project_report.pdf`: Project report

- `proposal.pdf`: proposal

#### Project Files
- `Telstra_project_dataexploration.ipynb`: jupyter notebook for data exprolation and visualization (`.html` is also available)

- `Telstra_project_onehot.ipynb`: jupyter notebook to train models with data encoded by one-hot encoding  (`.html` is also available)

- `Telstra_project_numeric.ipynb`: jupyter notebook to train models with data encoded by numeric encoding  (`.html` is also available)

- `telstra_helper.py`: Python file containing helper functions for this project

#### Data Files
- `data/*.csv`: CSV files used in this project as input data.

#### Output Files
These are output files from the project.
- `dnn_model_ohehot_best.hdf5`
- `result_numeric.csv`
- `result_onehot.csv`
- `figures/*`

### 5. Note

This project has been done as a part of Machine Learning Engineer Nanodegree program, at [Udacity](https://www.udacity.com/).
