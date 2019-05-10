def _load_model(model_file):
    from predict import FonduerModel
    DB = "pob_presidents"
    conn_string = 'postgresql://localhost:5432/' + DB
    model = FonduerModel(conn_string)
    return model

def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _FonduerWrapper(_load_model(path, **kwargs))


class _FonduerWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, fonduer_model):
        self.fonduer_model = fonduer_model

    def predict(self, dataframe):
        predicted = self.fonduer_model.predict(dataframe)
        return predicted
