import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from pandas import DataFrame
from scipy.sparse import csr_matrix
import torch

import emmental
from emmental.model import EmmentalModel
from fonduer import Meta, init_logging
from fonduer.parser import Parser
from fonduer.parser.parser import ParserUDF
from fonduer.parser.models import Document
from fonduer.parser.preprocessors import DocPreprocessor
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.features.feature_extractors import FeatureExtractor
from fonduer.features.featurizer import Featurizer, FeaturizerUDF
from fonduer.supervision.labeler import Labeler, LabelerUDF
from fonduer.utils.utils_udf import unshift_label_matrix
from snorkel.labeling.model import LabelModel


logger = logging.getLogger(__name__)

MODEL_TYPE = "model_type"


class FonduerModel(pyfunc.PythonModel):
    """
    A custom MLflow model for Fonduer.
    """
    def _classify(self, doc: Document) -> DataFrame:
        raise NotImplementedError()

    def predict(self, model_input: DataFrame) -> DataFrame:
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
        doc = next(self.preprocessor._parse_file(path, os.path.basename(path)))

        logger.info(f"Parsing {path}")
        doc = self.corpus_parser.apply(doc, pdf_path=path)

        logger.info(f"Extracting mentions from {path}")
        doc = self.mention_extractor.apply(doc)

        logger.info(f"Extracting candidates from {path}")
        doc = self.candidate_extractor.apply(doc, split=2)

        logger.info(f"Classifying candidates from {path}")
        df = self._classify(doc)
        return df


def _load_pyfunc(model_path: str):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    model = pickle.load(open(os.path.join(model_path, "model.pkl"), "rb"))
    fonduer_model = model["fonduer_model"]
    fonduer_model.preprocessor = model["preprosessor"]
    fonduer_model.corpus_parser = ParserUDF(**model["parser"])
    fonduer_model.mention_extractor = MentionExtractorUDF(**model["mention_extractor"])
    fonduer_model.candidate_extractor = CandidateExtractorUDF(**model["candidate_extractor"])

    # Configure logging for Fonduer
    init_logging(log_dir="logs")

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    candidate_classes = fonduer_model.candidate_extractor.candidate_classes

    fonduer_model.model_type = pyfunc_conf.get(MODEL_TYPE, "discriminative")
    if fonduer_model.model_type == "discriminative":
        emmental.init()
        fonduer_model.featurizer = FeaturizerUDF(candidate_classes, FeatureExtractor())
        fonduer_model.key_names = model["feature_keys"]
        fonduer_model.word2id = model["word2id"]

        fonduer_model.disc_model = torch.load(os.path.join(model_path, "disc_model.pkl"))
    else:
        fonduer_model.labeler = LabelerUDF(candidate_classes)
        fonduer_model.key_names = model["labeler_keys"]

        fonduer_model.gen_models = []
        for _ in candidate_classes:
            gen_model = LabelModel()
            gen_model.load(os.path.join(model_path, _.__name__ + ".pkl"))
            fonduer_model.gen_models.append(gen_model)
    return _FonduerWrapper(fonduer_model)


def log_model(
    fonduer_model: FonduerModel,
    artifact_path: str,
    preprocessor: DocPreprocessor,
    parser: Parser,
    mention_extractor: MentionExtractor,
    candidate_extractor: CandidateExtractor,
    code_paths: Optional[List[str]] = None,
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
        preprocessor=preprocessor,
        parser=parser,
        mention_extractor=mention_extractor,
        candidate_extractor=candidate_extractor,
        code_paths=code_paths,
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
    preprocessor: DocPreprocessor,
    parser: Parser,
    mention_extractor: MentionExtractor,
    candidate_extractor: CandidateExtractor,
    mlflow_model: Model = Model(),
    code_paths: Optional[List[str]] = None,
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

    # Note that ParserUDF, MentionExtractorUDF, CandidateExtractorUDF themselves are not picklable.
    model = {
        "fonduer_model": fonduer_model,
        "preprosessor": preprocessor,
        "parser": parser.udf_init_kwargs,
        "mention_extractor": mention_extractor.udf_init_kwargs,
        "candidate_extractor": candidate_extractor.udf_init_kwargs,
    }
    if model_type == "discriminative":
        key_names = [key.name for key in featurizer.get_keys()]
        model["feature_keys"] = key_names
        model["word2id"] = word2id

        torch.save(disc_model, os.path.join(path, "disc_model.pkl"))
    else:
        key_names = [key.name for key in labeler.get_keys()]
        model["labeler_keys"] = key_names

        for candidate_class, gen_model in zip(labeler.candidate_classes, gen_models):
            gen_model.save(os.path.join(path, candidate_class.__name__ + ".pkl"))

    pickle.dump(model, open(os.path.join(path, "model.pkl"), "wb"))

    _copy_file_or_tree(src=__file__, dst=model_code_path)
    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=model_code_path)

    mlflow_model.add_flavor(
        pyfunc.FLAVOR_NAME,
        code=pyfunc.CODE,
        loader_module=__name__,
        model_type=model_type,
    )
    mlflow_model.save(os.path.join(path, "MLmodel"))


class _FonduerWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(
        self, fonduer_model: FonduerModel
    ) -> None:
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        """
        self.fonduer_model = fonduer_model

    def predict(self, dataframe: DataFrame) -> DataFrame:
        predicted = self.fonduer_model.predict(dataframe)
        return predicted


def F_matrix(features: List[Dict[str, Any]], key_names: List[str]) -> csr_matrix:
    # Convert features (keys_map) into a sparse matrix
    keys_map = {}
    for (i, k) in enumerate(key_names):
        keys_map[k] = i

    indptr = [0]
    indices = []
    data = []
    for feature in features:
        for cand_key, cand_value in zip(feature["keys"], feature["values"]):
            if cand_key in key_names:
                indices.append(keys_map[cand_key])
                data.append(cand_value)
        indptr.append(len(indices))
    F = csr_matrix(
        (data, indices, indptr),
        shape=(len(features), len(key_names))
        )
    return F


def L_matrix(labels: List[Dict[str, Any]], key_names: List[str]) -> np.ndarray:
    return unshift_label_matrix(F_matrix(labels, key_names))
