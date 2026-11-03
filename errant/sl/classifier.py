from pathlib import Path
from rapidfuzz.distance import Levenshtein
import spacy.symbols as POS
from errant.sl.merger import is_conn, is_ref

base_dir = Path(__file__).resolve().parent

def load_word_list(path):
    with open(path, encoding="utf-8") as f:
        return set(line.rstrip("\n") for line in f)

# Sloleks 3.1 wordforms (non-standard-tagged entries excluded at build time)
sloleks = load_word_list(base_dir/"resources"/"sloleks_wordforms.txt")

# Input: A token string
# Output: Boolean; the word is a known Sloleks wordform
def in_sloleks(word):
    if word in sloleks:
        return True
    if word[:1].isupper() and word[0].lower() + word[1:] in sloleks:
        return True
    return False

# Classifier resources
nlp = None
# Open class coarse Spacy POS tags
open_pos1 = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "NUM", "SYM", "X"}
# Universal Dependencies relations that correspond to a coarse POS-like category
dep_map = {
    "amod": "ADJ",
    "advmod": "ADV",
    "det": "PRON",
    "case": "PREP",
    "punct": "PUNCT"}
# Dependency relations that mark an auxiliary (covers both "aux:pass" (UD v2)
# and the older "auxpass" subtype spelling)
aux_deps = {"aux", "auxpass", "aux:pass"}
# Single-letter prepositions confusable by voicing/pronunciation
prep_confusions = {frozenset({"s", "z"}), frozenset({"k", "h"}), frozenset({"u", "v"})}
# Lemma similarity above which two lemmas count as "close enough" for M-POS,
# tolerating lemmatizer noise instead of requiring an exact match
LEMMA_SIM_THRESHOLD = 0.8
# Divisor for scaling the allowed surface-text edit distance with word
# length (allowed = max(1, len // TEXT_EDIT_SCALE)), so short words aren't
# held to the same absolute distance as long ones
TEXT_EDIT_SCALE = 3

# Input: A resolved POS string
# Output: The one-sided category label ("O-" for punctuation, "S-" otherwise)
def label_pos(pos):
    return "O-PUNCT" if pos == "PUNCT" else "S-" + pos

# Input 1: A string
# Input 2: A string
# Output: Boolean; edit distance small relative to word length. Used for
# spelling, where the actual written characters are what matters.
def close_text(a, b):
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    allowed = max(1, max(len(a), len(b)) // TEXT_EDIT_SCALE)
    return Levenshtein.distance(a, b) <= allowed

# Input 1: Spacy orig token
# Input 2: Spacy cor token
# Output: Boolean; the lemmas are identical or similar enough to count as
# the same word. Used for morphology, where the abstract lemma matters, not
# the surface spelling; tolerates lemmatizer noise rather than requiring an
# exact match.
def close_lemma(o_tok, c_tok):
    if same_lemma(o_tok, c_tok):
        return True
    return Levenshtein.normalized_similarity(o_tok.lemma_.lower(), c_tok.lemma_.lower()) > LEMMA_SIM_THRESHOLD

# Input 1: Spacy orig token
# Input 2: Spacy cor token
# Output: Boolean; the lemmas are identical, ignoring case
def same_lemma(o_tok, c_tok):
    return o_tok.lemma_.lower() == c_tok.lemma_.lower()

# Input 1: The orig label POS
# Input 2: The cor label POS
# Output: The vocabulary label: V-<POS> if both sides have the same POS,
# V otherwise (O-PUNCT for punctuation)
def label_vocab(o_pos, c_pos):
    if c_pos == "PUNCT": return "O-PUNCT"
    return "V-" + c_pos if o_pos == c_pos else "V"

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Rewritten clause, preset by the merger; only the operation is added,
    # unless the span is only a whitespace / case change or a reordering
    if edit.type == "S-REWRITE":
        if edit.o_toks and edit.c_toks and only_orth_change(edit.o_toks, edit.c_toks):
            edit.type = "R:"+get_orth_subtype(edit.o_toks, edit.c_toks)
        elif edit.o_toks and edit.c_toks and exact_reordering(edit.o_toks, edit.c_toks):
            edit.type = "R:S-WO"
        elif edit.o_toks and edit.c_toks and get_span_type(edit.o_toks, edit.c_toks):
            edit.type = "R:"+get_span_type(edit.o_toks, edit.c_toks)
        else:
            edit.type = get_op(edit)+"S-REWRITE"
    # Nothing to nothing is a detected but not corrected edit
    elif not edit.o_toks and not edit.c_toks:
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

# Input: An Edit object
# Output: The operation prefix from the span: "M:", "U:" or "R:"
def get_op(edit):
    if not edit.o_toks: return "M:"
    if not edit.c_toks: return "U:"
    return "R:"

# Input: A Spacy token
# Output: The label POS of the token. A token without letters or digits is
# punctuation. UD POS renamed for readability (as upstream ERRANT), plus
# Slovene grammar: determiners and the relative ki
# are pronouns (zaimki), coordinating and subordinating conjunctions are one
# class (vezniki)
def label_pos_of(tok):
    if not any(ch.isalnum() for ch in tok.text): return "PUNCT"
    if tok.lower_ == "ki": return "PRON"
    if tok.pos_ == "ADP": return "PREP"
    if tok.pos_ == "PROPN": return "NOUN"
    if tok.pos_ == "DET": return "PRON"
    if tok.pos_ in {"CCONJ", "SCONJ"}: return "CONJ"
    return tok.pos_

# Input: Spacy tokens
# Output: A list of pos and dep tag strings
def get_edit_info(toks):
    pos = []
    dep = []
    for tok in toks:
        pos.append(label_pos_of(tok))
        dep.append(tok.dep_)
    return pos, dep

# Input: Spacy tokens
# Output: An error type string based on input tokens from orig or cor
# When one side of the edit is null, we can only use the other side
def get_one_sided_type(toks):
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # Punctuation next to a word belongs to it: [, da], [To ,]
    if set(pos_list) != {"PUNCT"}:
        dep_list = [d for p, d in zip(pos_list, dep_list) if p != "PUNCT"]
        pos_list = [p for p in pos_list if p != "PUNCT"]
    # Auxiliary verbs
    if set(dep_list).issubset(aux_deps):
        return "M-AUX"
    # POS-based tags. Ignores rare, uninformative categories; a missing /
    # unnecessary numeral is informative: [eden razlog -> razlog]
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_pos - {"NUM"}:
        return label_pos(pos_list[0])
    # More POS-based tags using special dependency labels
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return label_pos(dep_map[dep_list[0]])
    # Verb-related non-finite constructions
    if set(pos_list) == {"PART", "VERB"}:
        return label_pos("VERB")
    # A missing / unnecessary phrase of mixed POS is a structural change
    if len(pos_list) > 1 and "PUNCT" not in pos_list:
        return "S-REWRITE"
    # Tricky cases
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
        return get_orth_subtype(o_toks, c_toks)
    # Word Order; only matches exact reordering.
    if exact_reordering(o_toks, c_toks):
        return "S-WO"
    # Single-letter preposition confusion; both sides are valid Sloleks
    # entries so the lemma/Sloleks checks below can't catch these.
    if is_prep_confusion(o_toks, c_toks):
        return "O-SPELL"
    # Units the merger keeps together (phrase <-> pronoun)
    span_type = get_span_type(o_toks, c_toks)
    if span_type:
        return span_type

    # 1:1 replacements (very common)
    if len(o_toks) == len(c_toks) == 1:
        # Punctuation <-> conjunction: how clauses are linked, either
        # direction: [, -> in], [in -> ,]
        if {o_pos[0], c_pos[0]} == {"PUNCT", "CONJ"}:
            return "S-CONJ"
        o_in, c_in = in_sloleks(o_toks[0].text), in_sloleks(c_toks[0].text)

        if o_in and c_in:
            # 1. MORPHOLOGY: both confirmed real words, close lemma, same POS
            if close_lemma(o_toks[0], c_toks[0]) and o_pos[0] == c_pos[0]:
                return "O-PUNCT" if o_pos[0] == "PUNCT" else "M-" + o_pos[0]
            # Same lemma but the POS differs: morphology without a POS
            if same_lemma(o_toks[0], c_toks[0]) and "PUNCT" not in {o_pos[0], c_pos[0]}:
                return "M"
            # 2. VOCABULARY: both confirmed real words, not the same word
            return label_vocab(o_pos[0], c_pos[0])

        if o_in != c_in:
            # 3. SPELLING: one confirmed word, one not, close surface text ->
            # same underlying word, just misspelled. Compared on the surface
            # text rather than the lemma: a misspelling often confuses the
            # lemmatizer/tagger too, so the lemma can diverge even when the
            # written forms themselves are clearly close.
            if close_text(o_toks[0].text, c_toks[0].text):
                return "O-SPELL"
            # 4. VOCABULARY: one confirmed word, one not, not the same word
            return label_vocab(o_pos[0], c_pos[0])

        # neither in Sloleks -> fall through to the GENERAL/STRING
        # SIMILARITY fallback below

        # 4. GENERAL
        # Auxiliaries with different lemmas
        if o_dep[0].startswith("aux") and c_dep[0].startswith("aux"):
            return "M-AUX"
        # POS-based tags. Some of these are context sensitive mispellings.
        if o_pos == c_pos and o_pos[0] not in rare_pos:
            return "O-PUNCT" if o_pos[0] == "PUNCT" else "V-" + o_pos[0]
        # Some dep labels map to POS-based tags.
        if o_dep == c_dep and o_dep[0] in dep_map.keys():
            mapped = dep_map[o_dep[0]]
            return "O-PUNCT" if mapped == "PUNCT" else "V-" + mapped

        # 5. STRING SIMILARITY
        # Generic Levenshtein-based fallback; no language-specific tuning.
        if o_toks[0].text.isalpha() and c_toks[0].text.isalpha():
            str_sim = Levenshtein.normalized_similarity(o_toks[0].lower_, c_toks[0].lower_)
            if str_sim > 0.55 and c_pos[0] not in rare_pos:
                return "O-SPELL"
        # Tricky cases
        return "OTHER"

    # Multi-token replacements: the merger only builds these for rewrites,
    # phrase <-> pronoun, connectors and spacing, all handled above
    return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: The label of a non-decomposable unit the merger keeps as one edit,
# or None. Same tests as the merger, so both agree on these spans.
def get_span_type(o_toks, c_toks):
    # ki <-> kateri: a choice between two equivalent relative pronouns, i.e.
    # vocabulary: [kateri -> ki], [kateremu -> ki mu], [ki -> s katerim]
    if is_ki_kateri(o_toks, c_toks) or is_ki_kateri(c_toks, o_toks):
        return "V-PRON"
    # A phrase replaced by a single pronoun, or the reverse: [te vere pa ->
    # katere], [kateremu -> ki mu]. A bound form against preposition +
    # pronoun is a pronoun form, not a reference change: [za njih -> zanj]
    for a, b in ((o_toks, c_toks), (c_toks, o_toks)):
        if is_ref(a, b):
            if "Bound" in a[0].morph.get("Variant") and \
                    any(tok.pos in {POS.PRON, POS.DET} for tok in b):
                return "M-PRON"
            return "S-REF"
    # A multi-word connector replaced by a single one, or the reverse:
    # [za kar -> zato], [ker -> kljub temu da]. The relative ki is a
    # pronoun in Slovene grammar: [ki -> s katerim] is a reference change
    for a, b in ((o_toks, c_toks), (c_toks, o_toks)):
        if is_conn(a, b):
            return "S-REF" if a[0].lower_ == "ki" else "V-CONJ"
    return None

# Input 1: Tokens on one side of an edit
# Input 2: Tokens on the other side
# Output: True if a has the relative ki and b a form of kateri (matched on
# the form, kater-, since the lemmatiser often leaves these unlemmatised),
# and both sides are only function words: [ki -> s katerim], not [katere
# imenuju -> ki jih imenujejo]
def is_ki_kateri(a, b):
    if any(tok.pos not in {POS.PRON, POS.DET, POS.ADP, POS.SCONJ, POS.PUNCT}
            for tok in list(a)+list(b)):
        return False
    return any(tok.lower_ == "ki" for tok in a) and \
        not any(tok.lower_.startswith("kater") for tok in a) and \
        any(tok.lower_.startswith("kater") for tok in b)

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
# Output: The orthography subtype string; only_orth_change() must be true
def get_orth_subtype(o_toks, c_toks):
    o_exact = "".join(o.text for o in o_toks)
    c_exact = "".join(c.text for c in c_toks)
    if o_exact == c_exact:
        return "O-MERGE" if len(o_toks) > len(c_toks) else "O-SPLIT"
    return "O-CAP"

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

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; both sides are single-letter s/z, k/h or u/v prepositions
def is_prep_confusion(o_toks, c_toks):
    if len(o_toks) == 1 and len(c_toks) == 1:
        o_low, c_low = o_toks[0].text.lower(), c_toks[0].text.lower()
        if len(o_low) == 1 and len(c_low) == 1:
            return frozenset({o_low, c_low}) in prep_confusions
    return False
