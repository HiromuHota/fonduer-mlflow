import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.pyfunc.model import PythonModelContext
from mlflow.utils.model_utils import _get_flavor_configuration

from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.features import Featurizer
from fonduer.learning import LogisticRegression
from fonduer.parser import Parser
from fonduer.parser.preprocessors import HTMLDocPreprocessor

CONN_STRING = "conn_string"

ABSTAIN = 0
FALSE = 1
TRUE = 2

# Configure logging for Fonduer
init_logging(log_dir="logs")
logger = logging.getLogger(__name__)

PARALLEL = 4 # assuming a quad-core machine


class FonduerModel(pyfunc.PythonModel):

    def __init__(self, model_path):
        pyfunc_conf = _get_flavor_configuration(model_path=model_path,
                                                flavor_name=pyfunc.FLAVOR_NAME)
        conn_string = pyfunc_conf.get(CONN_STRING, None)
        if conn_string is None:
            raise RuntimeError("conn_string is missing from MLmodel file.")
        session = Meta.init(conn_string).Session()
        from fonduerconfig import matchers, mention_classes, mention_spaces, candidate_classes  # isort:skip

        self.corpus_parser = Parser(session, structural=True, lingual=True)
        self.mention_extractor = MentionExtractor(
            session,
            mention_classes, mention_spaces, matchers
        )
        self.candidate_extractor = CandidateExtractor(session, candidate_classes)

        self.featurizer = Featurizer(session, candidate_classes)
        with open(os.path.join(model_path, 'feature_keys.pkl'), 'rb') as f:
            key_names = pickle.load(f)
        self.featurizer.drop_keys(key_names)
        self.featurizer.upsert_keys(key_names)

        disc_model = LogisticRegression()

        # Workaround to https://github.com/HazyResearch/fonduer/issues/208
        checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
        disc_model.settings = checkpoint["config"]
        disc_model.cardinality = checkpoint["cardinality"]
        disc_model._build_model()

        disc_model.load(model_file="best_model.pt", save_dir=model_path)
        self.disc_model = disc_model

    def load_context(self, context):
        logger.info("loading context")

    def predict(self, context, model_input):
        df = pd.DataFrame()
        for index, row in model_input.iterrows():
            df = df.append(self._process(row['filename']))
        return df

    def _process(self, filename):
        # Parse docs
        docs_path = filename
        doc_preprocessor = HTMLDocPreprocessor(docs_path)
        # clear=False otherwise gets stuck.
        self.corpus_parser.apply(doc_preprocessor, clear=False, parallelism=PARALLEL)
        test_docs = self.corpus_parser.get_last_documents()

        self.mention_extractor.apply(test_docs, clear=False, parallelism=PARALLEL)

        # Candidate
        self.candidate_extractor.apply(test_docs, split=2, clear=True, parallelism=PARALLEL)
        test_cands = self.candidate_extractor.get_candidates(split=2)

        # Featurization
        self.featurizer.apply(test_docs, clear=False)
        F_test = self.featurizer.get_feature_matrices(test_cands)

        test_score = self.disc_model.predict((test_cands[0], F_test[0]), b=0.6, pos_label=TRUE)
        true_preds = [test_cands[0][_] for _ in np.nditer(np.where(test_score == TRUE))]

        df = pd.DataFrame()
        for entity_relation in FonduerModel.get_unique_entity_relations(true_preds):
            df = df.append(
                pd.DataFrame([entity_relation],
                columns=[m.__name__ for m in self.mention_extractor.mention_classes]
                )
            )
        return df

    @staticmethod
    def get_entity_relation(candidate):
        return tuple(([m.context.get_span() for m in candidate.get_mentions()]))

    @staticmethod
    def get_unique_entity_relations(candidates):
        unique_entity_relation = set()
        for candidate in candidates:
            entity_relation = FonduerModel.get_entity_relation(candidate)
            unique_entity_relation.add(entity_relation)
        return unique_entity_relation


def _load_pyfunc(model_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    fonduer_model = FonduerModel(model_path)
    context = PythonModelContext(artifacts=None)
    fonduer_model.load_context(context=context)
    return _FonduerWrapper(fonduer_model, context)


def save_model(model_path, featurizer, disc_model, conn_string):
    os.makedirs(model_path)
    model_code_path = os.path.join(model_path, pyfunc.CODE)
    os.makedirs(model_code_path)

    shutil.copy("fonduerconfig.py", model_code_path)
    shutil.copy("fonduer_model.py", model_code_path)
    key_names = [key.name for key in featurizer.get_keys()]
    with open(os.path.join(model_path, 'feature_keys.pkl'), 'wb') as f:
        pickle.dump(key_names, f)
    disc_model.save(model_file="best_model.pt", save_dir=model_path)

    mlflow_model = Model()
    mlflow_model.add_flavor(
        pyfunc.FLAVOR_NAME,
        code=pyfunc.CODE,
        loader_module="fonduer_model",
        conn_string=conn_string,
    )
    mlflow_model.save(os.path.join(model_path, "MLmodel"))


class _FonduerWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, fonduer_model, context):
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        """
        self.fonduer_model = fonduer_model
        self.context = context

    def predict(self, dataframe):
        predicted = self.fonduer_model.predict(self.context, dataframe)
        return predicted
