import logging
import os
import pickle
from typing import Iterable, List, Optional

import torch
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.pyfunc.model import PythonModelContext
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from pandas import DataFrame
from sqlalchemy.orm import Session

from fonduer import Meta, init_logging
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.candidates import MentionExtractor, CandidateExtractor
from fonduer.features import Featurizer
from fonduer.learning import LogisticRegression
from fonduer.learning.classifier import Classifier
from fonduer.supervision import Labeler
from metal.label_model import LabelModel

logger = logging.getLogger(__name__)

CONN_STRING = "conn_string"
MODEL_TYPE = "model_type"
PARALLEL = "parallel"


class FonduerModel(pyfunc.PythonModel):
    """
    A custom MLflow model for Fonduer.
    """

    def _get_doc_preprocessor(self, path: str) -> Iterable[Document]:
        raise NotImplementedError()

    def _get_parser(self, session: Session) -> Parser:
        raise NotImplementedError()

    def _get_mention_extractor(self, session: Session) -> MentionExtractor:
        raise NotImplementedError()

    def _get_candidate_extractor(self, session: Session) -> CandidateExtractor:
        raise NotImplementedError()

    def _classify(self) -> DataFrame:
        raise NotImplementedError()

    def load_context(self, context: PythonModelContext) -> None:
        # Configure logging for Fonduer
        init_logging(log_dir="logs")
        logger.info("loading context")

        pyfunc_conf = _get_flavor_configuration(
            model_path=self.model_path, flavor_name=pyfunc.FLAVOR_NAME
        )
        conn_string = pyfunc_conf.get(CONN_STRING, None)
        if conn_string is None:
            raise RuntimeError("conn_string is missing from MLmodel file.")
        self.parallel = pyfunc_conf.get(PARALLEL, 1)
        session = Meta.init(conn_string).Session()

        logger.info("Getting parser")
        self.corpus_parser = self._get_parser(session)
        logger.info("Getting mention extractor")
        self.mention_extractor = self._get_mention_extractor(session)
        logger.info("Getting candidate extractor")
        self.candidate_extractor = self._get_candidate_extractor(session)
        candidate_classes = self.candidate_extractor.candidate_classes

        self.model_type = pyfunc_conf.get(MODEL_TYPE, "discriminative")
        if self.model_type == "discriminative":
            self.featurizer = Featurizer(session, candidate_classes)
            with open(os.path.join(self.model_path, "feature_keys.pkl"), "rb") as f:
                key_names = pickle.load(f)
            self.featurizer.drop_keys(key_names)
            self.featurizer.upsert_keys(key_names)

            disc_model = LogisticRegression()

            # Workaround to https://github.com/HazyResearch/fonduer/issues/208
            checkpoint = torch.load(os.path.join(self.model_path, "best_model.pt"))
            disc_model.settings = checkpoint["config"]
            disc_model.cardinality = checkpoint["cardinality"]
            disc_model._build_model()

            disc_model.load(model_file="best_model.pt", save_dir=self.model_path)
            self.disc_model = disc_model
        else:
            self.labeler = Labeler(session, candidate_classes)
            with open(os.path.join(self.model_path, "labeler_keys.pkl"), "rb") as f:
                key_names = pickle.load(f)
            self.labeler.drop_keys(key_names)
            self.labeler.upsert_keys(key_names)

            self.gen_models = [
                LabelModel.load(os.path.join(self.model_path, _.__name__ + ".pkl"))
                for _ in candidate_classes
            ]

    def predict(self, context: PythonModelContext, model_input: DataFrame) -> DataFrame:
        df = DataFrame()
        for index, row in model_input.iterrows():
            df = df.append(self._process(row["path"]))
        return df

    def _process(self, path: str) -> DataFrame:
        """
        Takes a file/directory path and returns values extracted from the file or files in that directory.

        :param path: a file/directory path.
        """
        if not os.path.exists(path):
            raise RuntimeError("path should be a file/directory path")
        # Parse docs
        doc_preprocessor = self._get_doc_preprocessor(path)
        # clear=False otherwise gets stuck.
        self.corpus_parser.apply(
            doc_preprocessor, clear=False, parallelism=self.parallel, pdf_path=path
        )
        logger.info(f"Parsing {path}")
        test_docs = self.corpus_parser.get_last_documents()

        logger.info(f"Extracting mentions from {path}")
        self.mention_extractor.apply(test_docs, clear=False, parallelism=self.parallel)

        logger.info(f"Extracting candidates from {path}")
        self.candidate_extractor.apply(
            test_docs, split=2, clear=True, parallelism=self.parallel
        )

        logger.info(f"Classifying candidates from {path}")
        df = self._classify()
        return df


def _load_pyfunc(model_path: str):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    with open(os.path.join(model_path, "fonduer_model.pkl"), "rb") as f:
        fonduer_model = pickle.load(f)
    fonduer_model.model_path = model_path
    context = PythonModelContext(artifacts=None)
    fonduer_model.load_context(context=context)
    return _FonduerWrapper(fonduer_model, context)


def save_model(
    fonduer_model: FonduerModel,
    model_path: str,
    conn_string: str,
    code_paths: Optional[List[str]] = None,
    parallel: Optional[int] = 1,
    model_type: Optional[str] = "discriminative",
    labeler: Optional[Labeler] = None,
    gen_models: Optional[List[LabelModel]] = None,
    featurizer: Optional[Featurizer] = None,
    disc_model: Optional[Classifier] = None,
) -> None:
    """Save a custom MLflow model to a path on the local file system.

    :param fonduer_model: the model to be saved.
    :param model_path: the path on the local file system.
    :param conn_string: the connection string.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories containing file dependencies). These files are prepended to the system path when the model is loaded.
    :param parallel: the number of parallelism.
    :param model_type: the model type, either "discriminative" or "generative", defaults to "discriminative".
    :param labeler: a labeler, defaults to None.
    :param gen_models: a list of generative models, defaults to None.
    :param featurizer: a featurizer, defaults to None.
    :param disc_model: a discriminative model, defaults to None.
    """
    os.makedirs(model_path)
    model_code_path = os.path.join(model_path, pyfunc.CODE)
    os.makedirs(model_code_path)

    with open(os.path.join(model_path, "fonduer_model.pkl"), "wb") as f:
        pickle.dump(fonduer_model, f)
    if model_type == "discriminative":
        key_names = [key.name for key in featurizer.get_keys()]
        with open(os.path.join(model_path, "feature_keys.pkl"), "wb") as f:
            pickle.dump(key_names, f)
        disc_model.save(model_file="best_model.pt", save_dir=model_path)
    else:
        for candidate_class, gen_model in zip(labeler.candidate_classes, gen_models):
            gen_model.save(os.path.join(model_path, candidate_class.__name__ + ".pkl"))

        key_names = [key.name for key in labeler.get_keys()]
        with open(os.path.join(model_path, "labeler_keys.pkl"), "wb") as f:
            pickle.dump(key_names, f)

    _copy_file_or_tree(src=__file__, dst=model_code_path)
    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=model_code_path)

    mlflow_model = Model()
    mlflow_model.add_flavor(
        pyfunc.FLAVOR_NAME,
        code=pyfunc.CODE,
        loader_module="fonduer_model",
        conn_string=conn_string,
        parallel=parallel,
        model_type=model_type,
    )
    mlflow_model.save(os.path.join(model_path, "MLmodel"))


class _FonduerWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(
        self, fonduer_model: FonduerModel, context: PythonModelContext
    ) -> None:
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        """
        self.fonduer_model = fonduer_model
        self.context = context

    def predict(self, dataframe: DataFrame) -> DataFrame:
        predicted = self.fonduer_model.predict(self.context, dataframe)
        return predicted
