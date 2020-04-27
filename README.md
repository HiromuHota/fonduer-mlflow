# Overview

This project aims at managing the lifecycle of a Fonduer-based application.
Roughly a Fonduer-based app lifecycle has three phases: development, training, and serving.

| Phase | Framework / Interface
| --- | --- |
| Development | Jupyter Notebook / Web GUI |
| Training | MLflow Project / CLI |
| Serving | MLflow Model / Rest API |

In the development phase, a developer writes Python codes in that a parser, mention/candidate extractors, labeling functions, and a classifier are defined.
Once they are defined, a model can be trained using a training document set.
A trained model will be deployed and will serve to extract knowledge from a new document.

Jupyter Notebook might be good for development but not always good for training and serving.
This project uses MLflow in the training phase for reproducibility (of training) and in the serving phase for packageability (of a trained model).

Contributions to the Fonduer project include

- Defined a Fonduer model: what it includes, which parts are common/different for different apps.
- Created a custom MLflow model for Fonduer, which can be used to package a trained Fonduer model, deploy it, and let it serve.

# Prerequisites

- MLflow (v1.1.0 or higher)
- Anaconda or Miniconda
- Docker (optional)

# Development

`fonduer_model.py` defines `FonduerModel` that is a custom MLflow model (see [here](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models) for details) for Fonduer.
A developer is supposed to create a class, say `MyFonduerModel`, that inherits `FonduerModel` and implements `_classify(self, doc: Document) -> DataFrame`.

Also, a developer is supposed to create `fonduer_subclasses.py` and `fonduer_lfs.py`, each of which defines mention/candidate subclasses and labeling functions, respectively.

# Training

## Prepare

Download data.

```
$ ./download_data.sh
```

Deploy a PostgreSQL if you don't have one.

```
$ docker run --name postgres -e POSTGRES_USER=`whoami` -d -p 5432:5432 postgres
```

Create a database.

```
$ docker exec postgres createdb -U `whoami` pob_presidents
```

## Train a model

```
$ mlflow run ./ -P conn_string=postgresql://localhost:5432/pob_presidents
```

## Check the trained model

A trained Fonduer model will be saved at `./fonduer_model` with the following contents.

```bash
$ tree fonduer_model
fonduer_model
├── MLmodel
├── code
│   ├── fonduer_model.py
│   ├── fonduer_subclasses.py
│   └── my_fonduer_model.py
├── conda.yaml
└── model.pkl
```

This `fonduer_model` folder, conforming to the MLflow Model, is portable and can be deployed anywhere.

Note that the trained model can also be found under `./mlruns/<experiment-id>/<run-id>/artifacts`.

# Serving

There are a few ways to deploy a MLflow-compatible model (see [here](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models) for details).
Let me show you one of the ways.

## Deploys the model as a local REST API server

```
$ mlflow models serve -m fonduer_model -w 1
```

or alternatively,

```
$ mlflow models serve -m runs:/<run-id>/fonduer_model -w 1
```

If you send the following request to the API endpoint (`http://127.0.0.1:5000/invocations` in this case)

```
$ curl -X POST -H "Content-Type:application/json; format=pandas-split" \
  --data '{"columns":["path"], "data":["data/new/Al_Gore.html"]}' \
  http://127.0.0.1:5000/invocations
```

You will get a response like below:

```json
[
    {
        "Presidentname": "Al Gore",
        "Placeofbirth": "Washington"
    }
]
```

# Docker (experimental)

MLflow should be v1.8.0 or higher (mlflow/mlflow#2691, mlflow/mlflow#2699).

Build a Docker image

```
$ mlflow models build-docker -m fonduer_model -n fonduer_model
```

Deploy

```
$ docker run -p 5000:8080 -v "$(pwd)"/data:/opt/mlflow/data fonduer_model
```

# Acknowledgement

Most of the initial codes were derived from the wiki tutorial of [fonduer-tutorials](https://github.com/HazyResearch/fonduer-tutorials).
The Jupyter Notebook was converted to a Python script as follows:

```
$ jupyter nbconvert --to script some.ipynb
$ sed -i -e "s/get_ipython().run_line_magic('matplotlib', 'inline')/import matplotlib\nmatplotlib.use('Agg')/" some.py
```