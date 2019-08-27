from typing import Iterable, Set, Tuple
from pandas import DataFrame
import numpy as np

from sqlalchemy.orm import Session
from fonduer.parser import Parser
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.models import Document
from fonduer.candidates import MentionExtractor, CandidateExtractor
from fonduer.candidates.models import Candidate

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


ABSTAIN = 0
FALSE = 1
TRUE = 2


class MyFonduerModel(FonduerModel):
    def _get_doc_preprocessor(self, path: str) -> Iterable[Document]:
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

        test_score = self.disc_model.predict((test_cands[0], F_test[0]), b=0.6, pos_label=TRUE)
        true_preds = [test_cands[0][_] for _ in np.nditer(np.where(test_score == TRUE))]

        df = DataFrame()
        for entity_relation in get_unique_entity_relations(true_preds):
            df = df.append(
                DataFrame([entity_relation],
                columns=[m.__name__ for m in self.mention_extractor.mention_classes]
                )
            )
        return df
