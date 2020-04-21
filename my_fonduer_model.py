from typing import Iterable, Set, Tuple
from pandas import DataFrame
import numpy as np

from emmental.data import EmmentalDataLoader
from fonduer.parser.models import Document
from fonduer.candidates.models import Candidate
from fonduer.learning.dataset import FonduerDataset

from fonduer_model import FonduerModel, F_matrix
from fonduer_subclasses import mention_classes, candidate_classes


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
    def _classify(self, doc: Document) -> DataFrame:
        # Only one candidate class is defined.
        candidate_class = self.candidate_extractor.candidate_classes[0]
        test_cands = getattr(doc, candidate_class.__tablename__ + "s")

        # Featurization
        features_list = self.featurizer.apply(doc)

        # Convert features into a sparse matrix
        F_test = F_matrix(features_list[0], self.key_names)

        # Dataloader for test
        ATTRIBUTE = "wiki"
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={ATTRIBUTE: "labels"},
            dataset=FonduerDataset(
                ATTRIBUTE, test_cands, F_test, self.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        test_preds = self.disc_model.predict(test_dataloader, return_preds=True)
        positive = np.where(np.array(test_preds["probs"][ATTRIBUTE])[:, TRUE] > 0.6)
        true_preds = [test_cands[_] for _ in positive[0]]

        df = DataFrame()
        for entity_relation in get_unique_entity_relations(true_preds):
            df = df.append(
                DataFrame([entity_relation],
                columns=[m.__name__ for m in candidate_class.mentions]
                )
            )
        return df
