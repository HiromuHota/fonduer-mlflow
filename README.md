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

The trained Fonduer model will be saved at `fonduer_model`.

Let's see what's inside

```bash
(fonduer-mlflow) $ tree fonduer_model
fonduer_model
├── MLmodel
├── best_model.pt
├── code
├── feature_keys.pkl
└── fonduer_model.pkl
```

In order for this model to work, some dependent python files need to be saved at `fonduer_model/code`

```bash
$ cp fonduerconfig.py fonduer_model/code/
$ cp fonduer_model.py fonduer_model/code/
$ cp my_fonduer_model.py fonduer_model/code/
```

Now you'll get

```bash
(fonduer-mlflow) $ tree fonduer_model
fonduer_model
├── MLmodel
├── best_model.pt
├── code
│   ├── fonduer_model.py
│   ├── fonduerconfig.py
│   └── my_fonduer_model.py
├── feature_keys.pkl
└── fonduer_model.pkl
```

# MLflow Model

## Deploys the model as a local REST API server

```
$ mlflow models serve -m fonduer_model -w 1
```


```
$ curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["path"], "data":["data/new/Woodrow_Wilson.html"]}' http://127.0.0.1:5000/invocations
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
