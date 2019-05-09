"""
This file defines:
- Mention subclasses
- Mention spaces
- Mention matchers
"""

from fonduer.candidates.models import mention_subclass

Presidentname = mention_subclass("Presidentname")
Placeofbirth = mention_subclass("Placeofbirth")


def mention_span_matches_file_name(mention):
    president_name_string = mention.get_span()
    file_name = mention.sentence.document.name.replace("_", " ")
    if president_name_string == file_name:
        return True
    else:
        return False


from fonduer.candidates.matchers import LambdaFunctionMatcher, Intersect, Union

president_name_matcher = LambdaFunctionMatcher(func=mention_span_matches_file_name)

from fonduer.utils.data_model_utils import get_row_ngrams


def is_in_birthplace_table_row(mention):
    if not mention.sentence.is_tabular():
        return False
    ngrams = get_row_ngrams(mention, lower=True)
    birth_place_words = set(["birth", "place"])
    if birth_place_words <= set(ngrams):
        return True
    else:
        return False


def birthplace_left_aligned_to_punctuation(mention):
    # Return false, if the cell containing the text candidate does not have any reference
    # to `sentence` objects
    for sentence in mention.sentence.cell.sentences:
        sentence_parts = sentence.text.split(",")
        for sentence_part in sentence_parts:
            if sentence_part.startswith(mention.get_span()):
                return True
    return False


# We only want one granularity of the birth place
def no_commas_in_birth_place(mention):
    if "," in mention.get_span():
        return False
    else:
        return True


# After defining all functions to capture the properties of a birth place mention, we combine them via an `Intersect` matcher. This matcher will only select a `span`, if all three functions agree.

# In[12]:


birth_place_in_labeled_row_matcher = LambdaFunctionMatcher(
    func=is_in_birthplace_table_row
)
birth_place_in_labeled_row_matcher.longest_match_only = False
birth_place_no_commas_matcher = LambdaFunctionMatcher(func=no_commas_in_birth_place)
birth_place_left_aligned_matcher = LambdaFunctionMatcher(
    func=birthplace_left_aligned_to_punctuation
)

place_of_birth_matcher = Intersect(
    birth_place_in_labeled_row_matcher,
    birth_place_no_commas_matcher,
    birth_place_left_aligned_matcher,
)


from fonduer.candidates import MentionNgrams

presname_ngrams = MentionNgrams(n_max=4, n_min=2)
placeofbirth_ngrams = MentionNgrams(n_max=3)

from fonduer.candidates.models import candidate_subclass

PresidentnamePlaceofbirth = candidate_subclass(
    "PresidentnamePlaceofbirth", [Presidentname, Placeofbirth]
)

mention_classes = [Presidentname, Placeofbirth]
mention_spaces = [presname_ngrams, placeofbirth_ngrams]
matchers = [president_name_matcher, place_of_birth_matcher]
candidate_classes = [PresidentnamePlaceofbirth]
