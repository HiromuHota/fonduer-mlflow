# Overview

This project aims at managing the ops side of Fonduer-based applications.
A Fonduer-based app lifecycle has two phases: developement and operations.
Development starts from mention/candidate extraction, labeling functions development, to train a discriminative model using exisiting documents.
Operations includes deployment of the model that extracts knowledge from new documents.
Jupyter Notebook might be good for the development phase but never be good for the operations phase.

# MLflow Projects

## Prepare

Convert a Jupyter Notebook to a Python script.

```
$ jupyter nbconvert --to script some.ipynb
$ sed -i -e "s/get_ipython().run_line_magic('matplotlib', 'inline')/import matplotlib\nmatplotlib.use('Agg')/" some.py
```

Download data.

```
$ ./download_data.sh
```

Deploy a PostgreSQL if you don't have one.

```
$ docker run --name postgres -e POSTGRES_USER=ubuntu -d -p 5432:5432 postgres
```

Create an anaconda environment and activate it.

```
$ conda create env -f conda.yaml
$ conda activate fonduer-mlflow
```

Install spacy English model.

```
(fonduer-mlflow) $ python -m spacy download en
```

## Train

```
(fonduer-mlflow) $ mlflow run ./ --no-conda
```

## Test

```
(fonduer-mlflow) $ mlflow run -e test ./ --no-conda
```

# TODO

- MLflow Models