from fonduer.utils.data_model_utils import *
import re

ABSTAIN = 0
FALSE = 1
TRUE = 2


def LF_place_of_birth_has_link(c):
    place = c.placeofbirth
    ancestor_tag_names = get_ancestor_tag_names(place)
    if len(ancestor_tag_names) > 1 and "a" in ancestor_tag_names:
        return ABSTAIN
    else:
        return FALSE


def LF_place_of_birth_is_longest_ordered_span_before_comma(c):
    place = c.placeofbirth
    place_string = place.context.get_span()
    place_sentence_string = place.context.sentence.text
    left_aligned_first_span = place_sentence_string.split(",")[0]
    if place_string == left_aligned_first_span:
        return TRUE
    else:
        return FALSE


def LF_place_in_first_sentence_of_cell(c):
    place = c.placeofbirth
    place_sentence = place.context.sentence
    place_cell = place_sentence.cell
    if place_sentence == place_cell.sentences[0]:
        return TRUE
    else:
        return FALSE


def LF_place_is_full_sentence(c):
    place = c.placeofbirth
    place_sentence = place.context.sentence
    if place.context.get_span() == place_sentence.text:
        return ABSTAIN
    else:
        return FALSE


def LF_place_not_a_US_state(c):
    place = c.placeofbirth
    place_string = place.context.get_span().lower()
    if place_string is None:
        return FALSE
    state_dictionary = set(
        x.lower()
        for x in [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ]
    )
    if place_string == "new york city":  # exception
        return TRUE
    if place_string in state_dictionary:
        return FALSE
    elif any(x in place_string for x in state_dictionary):
        return FALSE
    else:
        return ABSTAIN


# Then, we collect all of the labeling function we would like to use into a single list, which is provided as input to the `Labeler`.

# In[24]:


president_name_pob_lfs = [
    LF_place_of_birth_has_link,
    LF_place_of_birth_is_longest_ordered_span_before_comma,
    LF_place_not_a_US_state,
    LF_place_in_first_sentence_of_cell,
    LF_place_is_full_sentence,
]
