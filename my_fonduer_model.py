import itertools
from typing import Iterable, Set, Tuple
from pandas import DataFrame
import numpy as np
from scipy.sparse import csr_matrix

from emmental.data import EmmentalDataLoader
from fonduer.parser.parser import ParserUDF
from fonduer.parser.preprocessors import DocPreprocessor, HTMLDocPreprocessor
from fonduer.parser.models import Document
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.candidates.models import Candidate
from fonduer.learning.dataset import FonduerDataset

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

    def _get_parser(self) -> ParserUDF:
        return ParserUDF(
            structural=True,
            blacklist=["style", "script"],
            flatten=["span", "br"],
            language="en",
            lingual=True,
            lingual_parser=None,
            strip=True,
            replacements=[("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")],
            tabular=True,
            visual=False,
            vizlink=None,
            pdf_path=None,
        )

    def _get_mention_extractor(self) -> MentionExtractorUDF:
        return MentionExtractorUDF(
            mention_classes, mention_spaces, matchers
        )

    def _get_candidate_extractor(self) -> CandidateExtractorUDF:
        return CandidateExtractorUDF(
            candidate_classes,
            [None] * len(candidate_classes),
            False, False, True
        )

    def _classify(self, doc: Document) -> DataFrame:
        test_cands = [getattr(doc, candidate_class.__tablename__ + "s") for candidate_class in candidate_classes]

        keys_map = {}
        for (i, k) in enumerate(self.key_names):
            keys_map[k] = i

        # Featurization
        features_list = self.featurizer.apply(doc)
        features = itertools.chain.from_iterable(features_list)

        # Convert features into a sparse matrix
        F_test = []
        indptr = [0]
        indices = []
        data = []
        for feature in features:
            for cand_key, cand_value in zip(feature["keys"], feature["values"]):
                if cand_key in self.key_names:
                    indices.append(keys_map[cand_key])
                    data.append(cand_value)
            indptr.append(len(indices))
        F_test.append(
            csr_matrix((data, indices, indptr), shape=(len(test_cands[0]), len(self.key_names)))
        )

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
