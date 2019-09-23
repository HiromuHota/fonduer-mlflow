#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='Fonduer')
parser.add_argument('--conn_string', help='conn string')
args = parser.parse_args()

PARALLEL = 4 # assuming a quad-core machine
conn_string = args.conn_string
print(conn_string)

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

from fonduerconfig import mention_classes, mention_spaces, matchers, candidate_classes
from fonduer.candidates import MentionExtractor
mention_extractor = MentionExtractor(
    session,
    mention_classes, mention_spaces, matchers
)

from fonduer.candidates.models import Mention

mention_extractor.apply(train_docs, parallelism=PARALLEL)
#num_names = session.query(Presidentname).count()
#num_pobs = session.query(Placeofbirth).count()
print(
    f"Total Mentions: {session.query(Mention).count()}"
)

from fonduer.candidates import CandidateExtractor


candidate_extractor = CandidateExtractor(session, candidate_classes)
candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
train_cands = candidate_extractor.get_candidates(split=0)
print(
    f"Number of Candidates: {len(train_cands[0])}"
)

from fonduer.features import Featurizer
import pickle

featurizer = Featurizer(session, candidate_classes)
featurizer.apply(split=0, train=True, parallelism=PARALLEL)
F_train = featurizer.get_feature_matrices(train_cands)

from wiki_table_utils import load_president_gold_labels

gold_file = "data/president_tutorial_gold.csv"
load_president_gold_labels(
    session, candidate_classes, gold_file, annotator_name="gold"
)

from lfconfig import president_name_pob_lfs, TRUE

from fonduer.supervision import Labeler

labeler = Labeler(session, candidate_classes)
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

from my_fonduer_model import MyFonduerModel
model = MyFonduerModel()

import fonduer_model
fonduer_model.save_model(
    fonduer_model=model,
    model_path="fonduer_model",
    conn_string=conn_string,
    code_paths=[
        "fonduerconfig.py",
        "my_fonduer_model.py",
    ],
    featurizer=featurizer,
    disc_model=disc_model,
)
