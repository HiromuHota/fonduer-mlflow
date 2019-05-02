import logging

import numpy as np

import torch
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import candidate_subclass
from fonduer.features import Featurizer
from fonduer.learning import LogisticRegression
from fonduer.parser import Parser
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from mentionconfig import *

ABSTAIN = 0
FALSE = 1
TRUE = 2

# Configure logging for Fonduer
init_logging(log_dir="logs")
logger = logging.getLogger(__name__)

PARALLEL = 4 # assuming a quad-core machine
ATTRIBUTE = "pob_presidents"
conn_string = 'postgresql://localhost:5432/' + ATTRIBUTE
session = Meta.init(conn_string).Session()

# Parse docs
logger.info("parsing...")
docs_path = "data/new/"
doc_preprocessor = HTMLDocPreprocessor(docs_path)
corpus_parser = Parser(session, structural=True, lingual=True)
# clear=False otherwise gets stuck.
corpus_parser.apply(doc_preprocessor, clear=False, parallelism=PARALLEL)
test_docs = corpus_parser.get_last_documents()


mention_extractor = MentionExtractor(
    session,
    [Presidentname, Placeofbirth],
    [presname_ngrams, placeofbirth_ngrams],
    [president_name_matcher, place_of_birth_matcher],
)
mention_extractor.apply(test_docs, clear=False, parallelism=PARALLEL)

# Candidate
PresidentnamePlaceofbirth = candidate_subclass(
    "PresidentnamePlaceofbirth", [Presidentname, Placeofbirth]
)
candidate_extractor = CandidateExtractor(session, [PresidentnamePlaceofbirth])
candidate_extractor.apply(test_docs, split=2, clear=False, parallelism=PARALLEL)
test_cands = candidate_extractor.get_candidates(split=2)

# Featurization
featurizer = Featurizer(session, [PresidentnamePlaceofbirth])
featurizer.apply(test_docs, clear=False)
F_test = featurizer.get_feature_matrices(test_cands)

disc_model = LogisticRegression()

# Workaround to https://github.com/HazyResearch/fonduer/issues/208
checkpoint = torch.load("./best_model.pt")
disc_model.settings = checkpoint["config"]
disc_model.cardinality = checkpoint["cardinality"]
disc_model._build_model()

disc_model.load(model_file="best_model.pt", save_dir="./")

test_score = disc_model.predict((test_cands[0], F_test[0]), b=0.6, pos_label=TRUE)
true_preds = [test_cands[0][_] for _ in np.nditer(np.where(test_score == TRUE))]

for true_pred in true_preds:
    print(', '.join([m.context.get_span() for m in true_pred.get_mentions()]))
