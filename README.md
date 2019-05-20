# Overview

This project aims at managing the lifecycle of a Fonduer-based application.
Roughly a Fonduer-based app lifecycle has two phases: training and serving.
Training, aka development, starts from mention/candidate extraction, labeling functions development, to train a generative/discriminative model using traing data, and test the model with test data.
Serving includes deployment of the model that serves to extract knowledge from new data.
Jupyter Notebook might be good for development but never be good for serving the model.

# MLflow Project

## Prepare

Download data.

```
$ ./download_data.sh
```

Deploy a PostgreSQL if you don't have one.

```
$ docker run --name postgres -e POSTGRES_USER=ubuntu -d -p 5432:5432 postgres
```

Create a database.

```
$ createdb pob_presidents
```

Create an anaconda environment and activate it.

```
$ conda env create -f conda.yaml
$ conda activate fonduer-mlflow
```

Install spacy English model.

```
(fonduer-mlflow) $ pip install git+https://github.com/HazyResearch/fonduer.git
(fonduer-mlflow) $ python -m spacy download en
```

## Train

```
(fonduer-mlflow) $ mlflow run ./ --no-conda -P conn_string=postgresql://localhost:5432/pob_presidents
```

# MLflow Model

## Deploys the model as a local REST API server

```
$ mlflow pyfunc serve -m fonduer_model
```


```
$ curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["filename"], "data":["data/new/Woodrow_Wilson.html"]}' http://127.0.0.1:5000/invocations
```

# Acknowlegements

Most of the initial codes were derived from the wiki tutorial of [fonduer-tutorials](https://github.com/HazyResearch/fonduer-tutorials).
The Jupyter Notebook was converted to a Python script as follows:

```
$ jupyter nbconvert --to script some.ipynb
$ sed -i -e "s/get_ipython().run_line_magic('matplotlib', 'inline')/import matplotlib\nmatplotlib.use('Agg')/" some.py
```

# TODO

- Split the training phase into smaller phases: parsing, mention extraction, candidate extraction, etc.
- How to package a model
