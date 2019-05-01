from fonduer.learning import LogisticRegression

from fonduer.candidates import CandidateExtractor
from fonduer.candidates.models import candidate_subclass
from fonduer.candidates.models import mention_subclass
from fonduer.features import Featurizer
from fonduer import Meta, init_logging
from wiki_table_utils import entity_level_f1
from fonduer.parser.models import Document
import numpy as np
import torch

ABSTAIN = 0
FALSE = 1
TRUE = 2

# Configure logging for Fonduer
init_logging(log_dir="logs")

PARALLEL = 4 # assuming a quad-core machine
ATTRIBUTE = "pob_presidents"
conn_string = 'postgresql://localhost:5432/' + ATTRIBUTE
session = Meta.init(conn_string).Session()

# Mention
Presidentname = mention_subclass("Presidentname")
Placeofbirth = mention_subclass("Placeofbirth")


# Candidate
PresidentnamePlaceofbirth = candidate_subclass(
    "PresidentnamePlaceofbirth", [Presidentname, Placeofbirth]
)
candidate_extractor = CandidateExtractor(session, [PresidentnamePlaceofbirth])
test_cands = candidate_extractor.get_candidates(split=2)

# Featurization
featurizer = Featurizer(session, [PresidentnamePlaceofbirth])
F_test = featurizer.get_feature_matrices(test_cands)

disc_model = LogisticRegression()

# Workaround to https://github.com/HazyResearch/fonduer/issues/208
checkpoint = torch.load("./best_model.pt")
disc_model.settings = checkpoint["config"]
disc_model.cardinality = checkpoint["cardinality"]
disc_model._build_model()

disc_model.load(model_file="best_model.pt", save_dir="./")

test_score = disc_model.predict((test_cands[0], F_test[0]), b=0.6, pos_label=TRUE)
true_pred = [test_cands[0][_] for _ in np.nditer(np.where(test_score == TRUE))]

gold_file = "data/president_tutorial_gold.csv"

# TODO: we only need test docs
test_docs = session.query(Document).order_by(Document.name).all()
(TP, FP, FN) = entity_level_f1(true_pred, gold_file, test_docs)

