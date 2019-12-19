from typing import Iterable, Set, Tuple
from pandas import DataFrame
import numpy as np

from sqlalchemy.orm import Session
from emmental.data import EmmentalDataLoader
from fonduer.parser import Parser
from fonduer.parser.preprocessors import DocPreprocessor, HTMLDocPreprocessor
from fonduer.candidates import MentionExtractor, CandidateExtractor
from fonduer.candidates.models import Candidate
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.utils import collect_word_counter

from fonduer_model import FonduerModel
from fonduerconfig import matchers, mention_classes, mention_spaces, candidate_classes


def get_entity_relation(candidate: Candidate) -> Tuple:
    return tuple(([m.context.get_span() for m in candidate.get_mentions()]))


def get_unique_entity_relations(candidates: Iterable[Candidate]) -> Set[Candidate]:
    unique_entity_relation = set()
    for candidate in candidates:
        entity_relation = get_entity_relation(candidate)
        unique_entity_relation.add(entity_relation)
    return unique_entity_relation


ABSTAIN = -1
FALSE = 0
TRUE = 1


class MyFonduerModel(FonduerModel):
    def _get_doc_preprocessor(self, path: str) -> DocPreprocessor:
        return HTMLDocPreprocessor(path)

    def _get_parser(self, session: Session) -> Parser:
        return Parser(session, structural=True, lingual=True)

    def _get_mention_extractor(self, session: Session) -> MentionExtractor:
        return MentionExtractor(
            session,
            mention_classes, mention_spaces, matchers
        )

    def _get_candidate_extractor(self, session: Session) -> CandidateExtractor:
        return CandidateExtractor(session, candidate_classes)

    def _classify(self) -> DataFrame:
        test_docs = self.corpus_parser.get_last_documents()
        test_cands = self.candidate_extractor.get_candidates(split=2)

        # Featurization
        self.featurizer.apply(test_docs, clear=False)
        F_test = self.featurizer.get_feature_matrices(test_cands)

        # Dataloader for test
        ATTRIBUTE = "wiki"
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands[0], F_test[0], self.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        test_preds = self.disc_model.predict(test_dataloader, return_preds=True)
        positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
        true_preds = [test_cands[0][_] for _ in positive[0]]

        df = DataFrame()
        for entity_relation in get_unique_entity_relations(true_preds):
            df = df.append(
                DataFrame([entity_relation],
                columns=[m.__name__ for m in self.mention_extractor.mention_classes]
                )
            )
        return df
