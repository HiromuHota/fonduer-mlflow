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

from fonduer_subclasses import mention_classes, mention_spaces, matchers, candidate_classes
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

featurizer = Featurizer(session, candidate_classes)
featurizer.apply(split=0, train=True, parallelism=PARALLEL)
F_train = featurizer.get_feature_matrices(train_cands)

from wiki_table_utils import gold
from fonduer.supervision import Labeler
from fonduer.supervision.models import GoldLabel

labeler = Labeler(session, candidate_classes)
labeler.apply(docs=train_docs, lfs=[[gold]], table=GoldLabel, train=True)

from fonduer_lfs import president_name_pob_lfs

labeler.apply(split=0, lfs=[president_name_pob_lfs], train=True, parallelism=PARALLEL)
L_train = labeler.get_label_matrices(train_cands)

L_gold_train = labeler.get_gold_labels(train_cands, annotator="gold")

from snorkel.labeling.model import LabelModel

label_model = LabelModel(verbose=False)
label_model.fit(L_train[0], n_epochs=500)

train_marginals = label_model.predict_proba(L_train[0])

ATTRIBUTE = "wiki"

import numpy as np
import emmental
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.modules.embedding_module import EmbeddingModule
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task
from fonduer.learning.utils import collect_word_counter
# Collect word counter
word_counter = collect_word_counter(train_cands)

emmental.init(Meta.log_path)

# Generate word embedding module
arity = 2
# Geneate special tokens
specials = []
for i in range(arity):
    specials += [f"~~[[{i}", f"{i}]]~~"]

emb_layer = EmbeddingModule(
    word_counter=word_counter, word_dim=300, specials=specials
)

diffs = train_marginals.max(axis=1) - train_marginals.min(axis=1)
train_idxs = np.where(diffs > 1e-6)[0]

train_dataloader = EmmentalDataLoader(
    task_to_label_dict={ATTRIBUTE: "labels"},
    dataset=FonduerDataset(
        ATTRIBUTE,
        train_cands[0],
        F_train[0],
        emb_layer.word2id,
        train_marginals,
        train_idxs,
    ),
    split="train",
    batch_size=100,
    shuffle=True,
)

tasks = create_task(
    ATTRIBUTE, 2, F_train[0].shape[1], 2, emb_layer, model="LogisticRegression"
)

emmental_model = EmmentalModel()

for task in tasks:
    emmental_model.add_task(task)

emmental_learner = EmmentalLearner()
emmental_learner.learn(emmental_model, [train_dataloader])

from my_fonduer_model import MyFonduerModel
model = MyFonduerModel()
code_paths = [
    "fonduer_subclasses.py",
    "fonduer_model.py",  # will no longer needed once HazyResearch/fonduer#407 is merged.
    "my_fonduer_model.py",
]

import fonduer_model

# Discrinative model
fonduer_model.save_model(
    model,
    "fonduer_emmental_model",
    code_paths=code_paths,
    preprocessor=doc_preprocessor,
    parser=corpus_parser,
    mention_extractor=mention_extractor,
    candidate_extractor=candidate_extractor,
    featurizer=featurizer,
    emmental_model=emmental_model,
    word2id=emb_layer.word2id,
)

fonduer_model.log_model(
    model,
    "fonduer_emmental_model",
    code_paths=code_paths,
    preprocessor=doc_preprocessor,
    parser=corpus_parser,
    mention_extractor=mention_extractor,
    candidate_extractor=candidate_extractor,
    featurizer=featurizer,
    emmental_model=emmental_model,
    word2id=emb_layer.word2id,
)

# Generative model
fonduer_model.save_model(
    model,
    "fonduer_label_model",
    code_paths=code_paths,
    preprocessor=doc_preprocessor,
    parser=corpus_parser,
    mention_extractor=mention_extractor,
    candidate_extractor=candidate_extractor,
    model_type="label",
    labeler=labeler,
    lfs=[president_name_pob_lfs],
    label_models=[label_model],
)

fonduer_model.log_model(
    model,
    "fonduer_label_model",
    code_paths=code_paths,
    preprocessor=doc_preprocessor,
    parser=corpus_parser,
    mention_extractor=mention_extractor,
    candidate_extractor=candidate_extractor,
    model_type="label",
    labeler=labeler,
    lfs=[president_name_pob_lfs],
    label_models=[label_model],
)
