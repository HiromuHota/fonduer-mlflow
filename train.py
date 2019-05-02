#!/usr/bin/env python
# coding: utf-8

PARALLEL = 4 # assuming a quad-core machine
ATTRIBUTE = "pob_presidents"
conn_string = 'postgresql://localhost:5432/' + ATTRIBUTE

from fonduer import Meta, init_logging

# Configure logging for Fonduer
init_logging(log_dir="logs")

session = Meta.init(conn_string).Session()

from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser import Parser

docs_path = "data/train/"
doc_preprocessor = HTMLDocPreprocessor(docs_path)


corpus_parser = Parser(session, structural=True, lingual=True)
corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer.parser.models import Document, Sentence

print(f"Documents: {session.query(Document).count()}")
print(f"Sentences: {session.query(Sentence).count()}")

train_docs = session.query(Document).order_by(Document.name).all()

# Mention

from mentionconfig import *
from fonduer.candidates import MentionExtractor
mention_extractor = MentionExtractor(
    session,
    [Presidentname, Placeofbirth],
    [presname_ngrams, placeofbirth_ngrams],
    [president_name_matcher, place_of_birth_matcher],
)

from fonduer.candidates.models import Mention

mention_extractor.apply(train_docs, parallelism=PARALLEL)
num_names = session.query(Presidentname).count()
num_pobs = session.query(Placeofbirth).count()
print(
    f"Total Mentions: {session.query(Mention).count()} ({num_names} names, {num_pobs} places of birth)"
)

from fonduer.candidates.models import candidate_subclass

PresidentnamePlaceofbirth = candidate_subclass(
    "PresidentnamePlaceofbirth", [Presidentname, Placeofbirth]
)

from fonduer.candidates import CandidateExtractor


candidate_extractor = CandidateExtractor(session, [PresidentnamePlaceofbirth])

for i, docs in enumerate([train_docs]):
    candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)
    print(
        f"Number of Candidates in split={i}: {session.query(PresidentnamePlaceofbirth).filter(PresidentnamePlaceofbirth.split == i).count()}"
    )

train_cands = candidate_extractor.get_candidates(split=0)

from fonduer.features import Featurizer

featurizer = Featurizer(session, [PresidentnamePlaceofbirth])
featurizer.apply(split=0, train=True, parallelism=PARALLEL)
F_train = featurizer.get_feature_matrices(train_cands)

from wiki_table_utils import load_president_gold_labels

gold_file = "data/president_tutorial_gold.csv"
load_president_gold_labels(
    session, PresidentnamePlaceofbirth, gold_file, annotator_name="gold"
)

from lfconfig import president_name_pob_lfs, TRUE

from fonduer.supervision import Labeler

labeler = Labeler(session, [PresidentnamePlaceofbirth])
labeler.apply(split=0, lfs=[president_name_pob_lfs], train=True, parallelism=PARALLEL)
L_train = labeler.get_label_matrices(train_cands)

L_gold_train = labeler.get_gold_labels(train_cands, annotator="gold")

from metal import analysis

analysis.lf_summary(
    L_train[0],
    lf_names=labeler.get_keys(),
    Y=L_gold_train[0].todense().reshape(-1).tolist()[0],
)

from metal.label_model import LabelModel

gen_model = LabelModel(k=2)
gen_model.train_model(L_train[0], n_epochs=500, print_every=100)

train_marginals = gen_model.predict_proba(L_train[0])

from fonduer.learning import LogisticRegression

disc_model = LogisticRegression()
disc_model.train((train_cands[0], F_train[0]), train_marginals, n_epochs=10, lr=0.001)


disc_model.save(model_file="best_model.pt", save_dir="./")
