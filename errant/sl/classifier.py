from rapidfuzz.distance import Levenshtein
import spacy.symbols as POS

# Classifier resources
nlp = None
# Open class coarse Spacy POS tags
open_pos1 = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
# Open class coarse Spacy POS tags (strings)
open_pos2 = {"ADJ", "ADV", "NOUN", "VERB"}
# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "NUM", "SYM", "X"}
# Universal Dependencies relations that correspond to a coarse POS-like category
dep_map = {
    "amod": "ADJ",
    "advmod": "ADV",
    "det": "DET",
    "case": "PREP",
    "punct": "PUNCT"}
# Dependency relations that mark an auxiliary (covers both "aux:pass" (UD v2)
# and the older "auxpass" subtype spelling)
aux_deps = {"aux", "auxpass", "aux:pass"}

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Nothing to nothing is a detected but not corrected edit
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    # Missing
    elif not edit.o_toks and edit.c_toks:
        op = "M:"
        cat = get_one_sided_type(edit.c_toks)
        edit.type = op+cat
    # Unnecessary
    elif edit.o_toks and not edit.c_toks:
        op = "U:"
        cat = get_one_sided_type(edit.o_toks)
        edit.type = op+cat
    # Replacement and special cases
    else:
        # Same to same is a detected but not corrected edit
        if edit.o_str == edit.c_str:
            edit.type = "UNK"
        # Special: Ignore case change at the end of multi token edits
        # E.g. [Doctor -> The doctor], [, since -> . Since]
        # Classify the edit as if the last token wasn't there
        elif edit.o_toks[-1].lower == edit.c_toks[-1].lower and \
                (len(edit.o_toks) > 1 or len(edit.c_toks) > 1):
            # Store a copy of the full orig and cor toks
            all_o_toks = edit.o_toks[:]
            all_c_toks = edit.c_toks[:]
            # Truncate the instance toks for classification
            edit.o_toks = edit.o_toks[:-1]
            edit.c_toks = edit.c_toks[:-1]
            # Classify the truncated edit
            edit = classify(edit)
            # Restore the full orig and cor toks
            edit.o_toks = all_o_toks
            edit.c_toks = all_c_toks
        # Replacement
        else:
            op = "R:"
            cat = get_two_sided_type(edit.o_toks, edit.c_toks)
            edit.type = op+cat
    return edit

# Input: Spacy tokens
# Output: A list of pos and dep tag strings
def get_edit_info(toks):
    pos = []
    dep = []
    for tok in toks:
        # Spacy's Universal POS tag (tok.pos_) is already language-agnostic;
        # rename a few categories for readability, same as upstream ERRANT.
        if tok.pos_ == "ADP": pos.append("PREP")
        elif tok.pos_ == "PROPN": pos.append("NOUN")
        elif tok.pos_ == "CCONJ": pos.append("CONJ")
        else: pos.append(tok.pos_)
        dep.append(tok.dep_)
    return pos, dep

# Input: Spacy tokens
# Output: An error type string based on input tokens from orig or cor
# When one side of the edit is null, we can only use the other side
def get_one_sided_type(toks):
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # Auxiliary verbs
    if set(dep_list).issubset(aux_deps):
        return "VERB:TENSE"
    # POS-based tags. Ignores rare, uninformative categories
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_pos:
        return pos_list[0]
    # More POS-based tags using special dependency labels
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return dep_map[dep_list[0]]
    # Verb-related non-finite constructions
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    # Tricky cases
    else:
        return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: An error type string based on orig AND cor
def get_two_sided_type(o_toks, c_toks):
    # Extract pos tags and parse info from the toks as lists
    o_pos, o_dep = get_edit_info(o_toks)
    c_pos, c_dep = get_edit_info(c_toks)

    # Orthography; i.e. whitespace and/or case errors.
    if only_orth_change(o_toks, c_toks):
        return "ORTH"
    # Word Order; only matches exact reordering.
    if exact_reordering(o_toks, c_toks):
        return "WO"

    # 1:1 replacements (very common)
    if len(o_toks) == len(c_toks) == 1:
        # 1. MORPHOLOGY
        # Only ADJ, ADV, NOUN and VERB can have inflectional changes.
        if o_toks[0].lemma == c_toks[0].lemma and \
                o_pos[0] in open_pos2 and \
                c_pos[0] in open_pos2:
            # Same POS on both sides
            if o_pos == c_pos:
                # Adjective form; e.g. comparatives
                if o_pos[0] == "ADJ":
                    return "ADJ:FORM"
                # Noun number
                if o_pos[0] == "NOUN":
                    return "NOUN:NUM"
                # Verbs - rely on the dependency parse; fine-grained tense/
                # aspect/mood distinctions need Slovene-tag-aware rules.
                if o_pos[0] == "VERB":
                    if preceded_by_aux(o_toks, c_toks):
                        return "VERB:FORM"
                    if o_dep[0].startswith("aux") and \
                            c_dep[0].startswith("aux"):
                        return "VERB:TENSE"
            # Use dep labels to find some more ADJ:FORM
            if set(o_dep+c_dep).issubset({"amod"}):
                return "ADJ:FORM"
            # Tricky cases that all have the same lemma.
            else:
                return "MORPH"

        # 2. GENERAL
        # Auxiliaries with different lemmas
        if o_dep[0].startswith("aux") and c_dep[0].startswith("aux"):
            return "VERB:TENSE"
        # POS-based tags. Some of these are context sensitive mispellings.
        if o_pos == c_pos and o_pos[0] not in rare_pos:
            return o_pos[0]
        # Some dep labels map to POS-based tags.
        if o_dep == c_dep and o_dep[0] in dep_map.keys():
            return dep_map[o_dep[0]]
        # Can use dep labels to resolve DET + PRON combinations.
        if set(o_pos+c_pos) == {"DET", "PRON"}:
            # DET cannot be a subject or object.
            if c_dep[0] in {"nsubj", "nsubj:pass", "obj", "iobj", "obl"}:
                return "PRON"
        # NUM and DET are usually DET; e.g. a <-> one
        if set(o_pos+c_pos) == {"NUM", "DET"}:
            return "DET"

        # 3. STRING SIMILARITY
        # Generic Levenshtein-based fallback; no language-specific tuning.
        if o_toks[0].text.isalpha() and c_toks[0].text.isalpha():
            str_sim = Levenshtein.normalized_similarity(o_toks[0].lower_, c_toks[0].lower_)
            if str_sim > 0.55 and c_pos[0] not in rare_pos:
                return "SPELL"
        # Tricky cases
        return "OTHER"

    # Multi-token replacements (uncommon)
    # All auxiliaries
    if set(o_dep+c_dep).issubset(aux_deps):
        return "VERB:TENSE"
    # All same POS
    if len(set(o_pos+c_pos)) == 1:
        # Final verbs with the same lemma are tense; e.g. eat -> has eaten
        if o_pos[0] == "VERB" and \
                o_toks[-1].lemma == c_toks[-1].lemma:
            return "VERB:TENSE"
        # POS-based tags.
        elif o_pos[0] not in rare_pos:
            return o_pos[0]
    # All same special dep labels.
    if len(set(o_dep+c_dep)) == 1 and \
            o_dep[0] in dep_map.keys():
        return dep_map[o_dep[0]]
    # Verb-related non-finite constructions.
    if set(o_pos+c_pos) == {"PART", "VERB"}:
        # Final verbs with the same lemma are form; e.g. to eat -> eating
        if o_toks[-1].lemma == c_toks[-1].lemma:
            return "VERB:FORM"
        # Remaining edits are often verb
        else:
            return "VERB"

    # Tricky cases.
    return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the difference between orig and cor is only whitespace or case
def only_orth_change(o_toks, c_toks):
    o_join = "".join([o.lower_ for o in o_toks])
    c_join = "".join([c.lower_ for c in c_toks])
    if o_join == c_join:
        return True
    return False

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the tokens are exactly the same but in a different order
def exact_reordering(o_toks, c_toks):
    # Sorting lets us keep duplicates.
    o_set = sorted([o.lower_ for o in o_toks])
    c_set = sorted([c.lower_ for c in c_toks])
    if o_set == c_set:
        return True
    return False

# Input 1: An original text spacy token.
# Input 2: A corrected text spacy token.
# Output: Boolean; both tokens have a dependant auxiliary verb.
def preceded_by_aux(o_tok, c_tok):
    # If the toks are aux, we need to check if they are the first aux.
    if o_tok[0].dep_.startswith("aux") and c_tok[0].dep_.startswith("aux"):
        # Find the parent verb
        o_head = o_tok[0].head
        c_head = c_tok[0].head
        # Find the children of the parent
        o_children = o_head.children
        c_children = c_head.children
        # Check the orig children.
        for o_child in o_children:
            # Look at the first aux...
            if o_child.dep_.startswith("aux"):
                # Check if the string matches o_tok
                if o_child.text != o_tok[0].text:
                    # If it doesn't, o_tok is not first so check cor
                    for c_child in c_children:
                        # Find the first aux in cor...
                        if c_child.dep_.startswith("aux"):
                            # If that doesn't match either, neither are first aux
                            if c_child.text != c_tok[0].text:
                                return True
                            # Break after the first cor aux
                            break
                # Break after the first orig aux.
                break
    # Otherwise, the toks are main verbs so we need to look for any aux.
    else:
        o_deps = [o_dep.dep_ for o_dep in o_tok[0].children]
        c_deps = [c_dep.dep_ for c_dep in c_tok[0].children]
        if any(d.startswith("aux") for d in o_deps):
            if any(d.startswith("aux") for d in c_deps):
                return True
    return False
