import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

from mlflow import pyfunc
from mlflow.models import Model
from mlflow.pyfunc.model import PythonModelContext
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from pandas import DataFrame
import torch

import emmental
from emmental.model import EmmentalModel
from fonduer import Meta, init_logging
from fonduer.parser.parser import ParserUDF
from fonduer.parser.preprocessors import DocPreprocessor
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.features.featurizer import Featurizer, FeaturizerUDF
from fonduer.supervision.labeler import Labeler, LabelerUDF
from snorkel.labeling import LabelModel

logger = logging.getLogger(__name__)

MODEL_TYPE = "model_type"
PARALLEL = "parallel"


class FonduerModel(pyfunc.PythonModel):
    """
    A custom MLflow model for Fonduer.
    """

    def _get_doc_preprocessor(self, path: str) -> DocPreprocessor:
        raise NotImplementedError()

    def _get_parser(self) -> ParserUDF:
        raise NotImplementedError()

    def _get_mention_extractor(self) -> MentionExtractorUDF:
        raise NotImplementedError()

    def _get_candidate_extractor(self) -> CandidateExtractorUDF:
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
        self.parallel = pyfunc_conf.get(PARALLEL, 1)
        emmental.init(Meta.log_path)

        logger.info("Getting parser")
        self.corpus_parser = self._get_parser()
        logger.info("Getting mention extractor")
        self.mention_extractor = self._get_mention_extractor()
        logger.info("Getting candidate extractor")
        self.candidate_extractor = self._get_candidate_extractor()
        candidate_classes = self.candidate_extractor.candidate_classes

        self.model_type = pyfunc_conf.get(MODEL_TYPE, "discriminative")
        if self.model_type == "discriminative":
            self.featurizer = FeaturizerUDF(candidate_classes)
            with open(os.path.join(self.model_path, "feature_keys.pkl"), "rb") as f:
                key_names = pickle.load(f)
            self.featurizer.drop_keys(key_names)
            self.featurizer.upsert_keys(key_names)

            self.disc_model = torch.load(os.path.join(self.model_path, "disc_model.pkl"))

            with open(os.path.join(self.model_path, "word2id.pkl"), "rb") as f:
                self.word2id = pickle.load(f)

        else:
            self.labeler = LabelerUDF(candidate_classes)
            with open(os.path.join(self.model_path, "labeler_keys.pkl"), "rb") as f:
                key_names = pickle.load(f)

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


def log_model(
    fonduer_model: FonduerModel,
    artifact_path: str,
    code_paths: Optional[List[str]] = None,
    parallel: Optional[int] = 1,
    model_type: Optional[str] = "discriminative",
    labeler: Optional[Labeler] = None,
    gen_models: Optional[List[LabelModel]] = None,
    featurizer: Optional[Featurizer] = None,
    disc_model: Optional[EmmentalModel] = None,
    word2id: Optional[Dict] = None,
) -> None:
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        fonduer_model=fonduer_model,
        code_paths=code_paths,
        parallel=parallel,
        model_type=model_type,
        labeler=labeler,
        gen_models=gen_models,
        featurizer=featurizer,
        disc_model=disc_model,
        word2id=word2id,
    )


def save_model(
    fonduer_model: FonduerModel,
    path: str,
    mlflow_model: Model = Model(),
    code_paths: Optional[List[str]] = None,
    parallel: Optional[int] = 1,
    model_type: Optional[str] = "discriminative",
    labeler: Optional[Labeler] = None,
    gen_models: Optional[List[LabelModel]] = None,
    featurizer: Optional[Featurizer] = None,
    disc_model: Optional[EmmentalModel] = None,
    word2id: Optional[Dict] = None,
) -> None:
    """Save a custom MLflow model to a path on the local file system.

    :param fonduer_model: the model to be saved.
    :param path: the path on the local file system.
    :param conn_string: the connection string.
    :param mlflow_model: model configuration.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories containing file dependencies). These files are prepended to the system path when the model is loaded.
    :param parallel: the number of parallelism.
    :param model_type: the model type, either "discriminative" or "generative", defaults to "discriminative".
    :param labeler: a labeler, defaults to None.
    :param gen_models: a list of generative models, defaults to None.
    :param featurizer: a featurizer, defaults to None.
    :param disc_model: a discriminative model, defaults to None.
    :param word2id: a word embedding map.
    """
    os.makedirs(path)
    model_code_path = os.path.join(path, pyfunc.CODE)
    os.makedirs(model_code_path)

    with open(os.path.join(path, "fonduer_model.pkl"), "wb") as f:
        pickle.dump(fonduer_model, f)
    if model_type == "discriminative":
        key_names = [key.name for key in featurizer.get_keys()]
        with open(os.path.join(path, "feature_keys.pkl"), "wb") as f:
            pickle.dump(key_names, f)

        torch.save(disc_model, os.path.join(path, "disc_model.pkl"))

        with open(os.path.join(path, "word2id.pkl"), "wb") as f:
            pickle.dump(word2id, f)
    else:
        for candidate_class, gen_model in zip(labeler.candidate_classes, gen_models):
            gen_model.save(os.path.join(path, candidate_class.__name__ + ".pkl"))

        key_names = [key.name for key in labeler.get_keys()]
        with open(os.path.join(path, "labeler_keys.pkl"), "wb") as f:
            pickle.dump(key_names, f)

    _copy_file_or_tree(src=__file__, dst=model_code_path)
    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=model_code_path)

    mlflow_model.add_flavor(
        pyfunc.FLAVOR_NAME,
        code=pyfunc.CODE,
        loader_module=__name__,
        parallel=parallel,
        model_type=model_type,
    )
    mlflow_model.save(os.path.join(path, "MLmodel"))


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
