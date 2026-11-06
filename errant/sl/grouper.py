"""Slovene grouper (developed in experiments/grouper, see NOTES.md there).

Runs after the classifier and adds a second label level: edits that belong to
one morphological change form a group with one group label M-<FEAT>. Single
morphology edits are groups of one. The classifier's word-level labels
(edit.type) are never changed.

Each edit in a group gets:
    edit.group        an int, unique within the sentence
    edit.group_label  M-<FEAT>
    edit.group_op     the operation of the whole group: all members M -> M,
                      all members U -> U, anything else -> R
Edits outside morphology groups keep None in all three. Edit.to_m2() writes
the group into the M2 comment field as G<n>=<group op>:<group label>.
"""

# Tagger feature -> group type
FEAT_TYPES = [("Case", "CASE"), ("Number", "NUM"), ("Number[psor]", "NUM"),
    ("Gender", "GEND"), ("Gender[psor]", "GEND"), ("Person", "PRS"),
    ("Definite", "DEF"), ("Degree", "DEG"), ("Tense", "TENSE"),
    ("Aspect", "ASP"), ("Mood", "MOOD")]
# Group types, most specific first: verb categories, then adjective ones,
# then nominal ones; person before number (2sg -> 1pl is a person error).
# The group label is the first type the group has.
PRIORITY = ["SFORM", "MOOD", "VFORM", "TENSE", "ASP", "REFL", "DEG", "DEF", "GOV", "CASE",
    "PRS", "NUM", "GEND", "MORPH"]
# Reflexive clitics; se <-> si is reflexivity, sebe <-> sabo is case
REFL_CLITICS = {"se", "si"}
# Analytic comparison
DEG_WORDS = {"bolj", "najbolj", "manj", "najmanj"}
# Dependency relations of agreement (NP, subject - predicate, auxiliary /
# copula - participle, predicative, coordination, bolj - adjective);
# government relations (nmod, obj, obl) don't link: each governed case is
# its own error
AGREE_DEPS = {"amod", "det", "nummod", "conj", "appos", "aux", "cop", "nsubj",
    "xcomp", "advmod"}
# Clause relations; they link two verbs only: [učiš -> učimo, spoznaš -> spoznamo]
CLAUSE_DEPS = {"advcl", "ccomp", "parataxis"}
VERBS = {"VERB", "AUX"}
# Modifiers inside an NP; a preposition joins a case change on them
NP_MODS = {"amod", "det", "nummod"}
# Agreement features a bridge edit can pass on, and their group types
BRIDGE_FEATS = {"Case": "CASE", "Number": "NUM", "Gender": "GEND", "Person": "PRS"}
# Heads that can link their agreeing dependents when they don't change
# themselves (syncretism): [svojem strokovnem štabu -> svojemu strokovnemu štabu]
NOMINAL = {"NOUN", "PROPN", "ADJ", "PRON", "DET", "NUM"}

# Input: A spacy token
# Output: {feature: value} from the tagger
def feats(tok):
    return {k: v for k, v in (f.split("=", 1) for f in str(tok.morph).split("|") if "=" in f)}

def same_lemma(a, b):
    return a.lemma_.lower() == b.lemma_.lower()

# Input 1: orig tokens of an edit
# Input 2: cor tokens of an edit
# Output: (orig, cor) word pairs: same lemma, or the only token on each side
def word_pairs(o, c):
    pairs = [(a, b) for a in o for b in c if same_lemma(a, b)]
    if not pairs and len(o) == 1 and len(c) == 1:
        pairs = [(o[0], c[0])]
    return pairs

# Input 1: An orig token
# Input 2: A cor token
# Output: The set of group types for what changed between them
def pair_types(a, b):
    fa, fb = feats(a), feats(b)
    types = {t for k, t in FEAT_TYPES if k in fa and k in fb and fa[k] != fb[k]}
    vforms = {fa.get("VerbForm"), fb.get("VerbForm")}
    if vforms == {"Inf", "Sup"}:
        types.add("SFORM")
    if vforms == {"Fin", "Part"}:
        types.add("TENSE")
    # Non-finite (infinitive / supine) vs finite / participle: the form the
    # construction requires [dela -> delati], [razumeti -> razumel]
    if len(vforms & {"Inf", "Sup"}) == 1 and len(vforms & {"Fin", "Part"}) == 1:
        types.add("VFORM")
    if {a.lower_, b.lower_} == REFL_CLITICS:
        types.add("REFL")
    return types

# Input: An Edit object
# Output: Boolean; the classifier labelled it morphology (M-<POS> or bare M)
def is_morph(edit):
    label = edit.type.split(":", 1)[-1]
    return label == "M" or label.startswith("M-")

# Input: An Edit object
# Output: The single word token of a one-sided edit (punctuation ignored),
# or None
def one_sided_word(edit):
    if bool(edit.o_toks) == bool(edit.c_toks):
        return None
    words = [t for t in (edit.o_toks or edit.c_toks) if any(ch.isalnum() for ch in t.text)]
    return words[0] if len(words) == 1 else None

# Input 1: The single word of a one-sided edit
# Input 2: The Doc it comes from
# Output: The group type of a one-sided category word, or None: conditional
# bi -> MOOD, reflexive se / si -> REFL, analytic degree bolj / najbolj /
# manj / najmanj before an adjective / adverb -> DEG, l-participle of biti
# next to a present form of biti (copula) -> TENSE
def one_sided_type(tok, doc):
    low = tok.lower_
    if low == "bi" and tok.lemma_.lower() == "biti":
        return "MOOD"
    if low in REFL_CLITICS and tok.pos_ == "PRON":
        return "REFL"
    if low in DEG_WORDS:
        nxt = doc[tok.i+1] if tok.i+1 < len(doc) else None
        if (nxt is not None and nxt.pos_ in {"ADJ", "ADV"}) or tok.head.pos_ in {"ADJ", "ADV"}:
            return "DEG"
    # [ki je bila smrtna -> ki je smrtna]; not with a passive participle,
    # which is voice: [ni bil umit -> se ni umil]
    if tok.lemma_.lower() == "biti" and feats(tok).get("VerbForm") == "Part" and \
            (tok.head.i == tok.i or feats(tok.head).get("VerbForm") != "Part"):
        near = doc[max(tok.i-2, 0):tok.i+3]
        if any(t.lemma_.lower() == "biti" and feats(t).get("Tense") == "Pres" for t in near):
            return "TENSE"
    return None

# Input 1: A one-sided Edit object
# Input 2: All edits of its sentence
# Input 3: orig Doc
# Output: Boolean; the word only moves (another edit loses the word this one
# inserts, or gains the word this one deletes) or is a repeated word deleted
# next to its twin: [je še dobro se zavedal -> se je še dobro zavedal], [se se -> se]
def moved_or_repeated(edit, edits, orig):
    w = one_sided_word(edit).lower_
    for other in edits:
        if other is edit: continue
        n_o = sum(t.lower_ == w for t in other.o_toks)
        n_c = sum(t.lower_ == w for t in other.c_toks)
        if (n_o > n_c) if not edit.o_toks else (n_c > n_o):
            return True
    if edit.o_toks:
        near = [orig[i].lower_ for i in (edit.o_start-1, edit.o_end) if 0 <= i < len(orig)]
        return w in near
    return False

# Input 1: An Edit object
# Input 2: All edits of its sentence
# Input 3: orig Doc
# Input 4: cor Doc
# Output: The set of group types of the edit; empty if it doesn't start a
# morphology group
def edit_types(edit, edits, orig, cor):
    tok = one_sided_word(edit)
    if tok is not None:
        if moved_or_repeated(edit, edits, orig):
            return set()
        t = one_sided_type(tok, orig if edit.o_toks else cor)
        if t:
            return {t}
        # Other one-sided morphology edits (auxiliaries)
        return {"MORPH"} if is_morph(edit) else set()
    if not is_morph(edit) or not edit.o_toks or not edit.c_toks:
        return set()
    types = set()
    for a, b in word_pairs(edit.o_toks, edit.c_toks):
        types |= pair_types(a, b)
    return types or {"MORPH"}

# Input: A set of group types
# Output: The label type by priority
def pick(types):
    return next(t for t in PRIORITY if t in types)

# Input 1: An Edit object
# Input 2: Another Edit object
# Input 3: orig Doc
# Input 4: cor Doc
# Output: Boolean; a token of one edit is the head of a token of the other
# by an agreement relation, in the cor parse if both edits have cor tokens,
# else in the orig parse
def dep_linked(a, b, orig, cor):
    if a.c_toks and b.c_toks:
        doc, ra, rb = cor, range(a.c_start, a.c_end), range(b.c_start, b.c_end)
    elif a.o_toks and b.o_toks:
        doc, ra, rb = orig, range(a.o_start, a.o_end), range(b.o_start, b.o_end)
    else:
        return False
    return any(doc[i].head.i in rb and agree(doc[i]) for i in ra) or \
        any(doc[j].head.i in ra and agree(doc[j]) for j in rb)

# Input: A spacy token
# Output: Boolean; the token is attached to its head by an agreement relation
def agree(t):
    dep = t.dep_.split(":")[0]
    return dep in AGREE_DEPS or \
        (dep in CLAUSE_DEPS and t.pos_ in VERBS and t.head.pos_ in VERBS)

# Input: An Edit object without group types
# Output: The agreement types that change in a 1:1 edit (a word changed for
# another reason that still carries the agreement change), else an empty set:
# [ostali gospodari -> ostale gospodarstvenike] (CASE)
def bridge_types(edit):
    if len(edit.o_toks) != 1 or len(edit.c_toks) != 1:
        return set()
    fa, fb = feats(edit.o_toks[0]), feats(edit.c_toks[0])
    return {t for k, t in BRIDGE_FEATS.items() if k in fa and k in fb and fa[k] != fb[k]}

# Input 1: An Edit object
# Input 2: Another Edit object
# Input 3: cor Doc
# Input 4: All edits of the sentence
# Output: Boolean; both edits have a token attached by an agreement relation
# to the same unchanged nominal head (cor parse): [bili so daljši -> bil je
# daljši]
def sibling_linked(a, b, cor, edits):
    if not a.c_toks or not b.c_toks:
        return False
    ia, ib = {t.i for t in a.c_toks}, {t.i for t in b.c_toks}
    ha = {t.head.i for t in a.c_toks if agree(t)} - ia - ib
    hb = {t.head.i for t in b.c_toks if agree(t)} - ia - ib
    changed = {i for e in edits for i in range(e.c_start, e.c_end)}
    return any(cor[h].pos_ in NOMINAL and h not in changed for h in ha & hb)

# Input: All edits of a sentence
# Output: (deletion index, insertion index) pairs of a deleted and an
# inserted auxiliary / copula biti in different forms whose heads are the
# same word (same lemma)
def reordered_biti(edits):
    def biti(e):
        t = one_sided_word(e)
        ok = t is not None and t.lemma_.lower() == "biti" and t.dep_.split(":")[0] in {"aux", "cop"}
        return t if ok else None
    dels = [(i, biti(e)) for i, e in enumerate(edits) if e.o_toks and biti(e)]
    ins = [(j, biti(e)) for j, e in enumerate(edits) if e.c_toks and biti(e)]
    pairs, used = [], set()
    for i, a in dels:
        for j, b in ins:
            if j not in used and a.lower_ != b.lower_ and \
                    a.head.lemma_.lower() == b.head.lemma_.lower():
                pairs.append((i, j))
                used.add(j)
                break
    return pairs

# Input: An Edit object
# Output: Boolean; a one-sided edit of a single auxiliary or copula (dep
# aux / cop)
def is_one_sided_aux(edit):
    tok = one_sided_word(edit)
    return tok is not None and tok.dep_.split(":")[0] in {"aux", "cop"}

# Input: An Edit object
# Output: Boolean; the edit only changes / adds / removes a preposition and
# the classifier labelled it as such (not a spelling confusion like z / s);
# the cor words decide, so a replaced non-preposition counts:
# [samo -> razen], [kod -> k]
def is_prep_edit(edit):
    label = edit.type.split(":", 1)[-1]
    toks = edit.c_toks if edit.c_toks else edit.o_toks
    words = [t for t in toks if any(ch.isalnum() for ch in t.text)]
    return label in {"V-PREP", "S-PREP", "V"} and bool(words) and \
        all(t.pos_ == "ADP" for t in words)

# Input 1: A preposition Edit object
# Input 2: All edits of the sentence
# Input 3: Their group types
# Output: The index of the CASE edit that holds the preposition's head noun,
# else of one that holds a modifier of that noun [z -> na + fantove ->
# fantovi strani] (cor parse; orig parse for a deleted preposition), or None
def prep_target(edit, edits, types):
    side = (lambda e: e.c_toks) if edit.c_toks else (lambda e: e.o_toks)
    heads = {t.head.i for t in side(edit) if t.pos_ == "ADP"}
    cands = [j for j, other in enumerate(edits) if other is not edit and "CASE" in types[j]]
    for j in cands:
        if heads & {t.i for t in side(edits[j])}:
            return j
    for j in cands:
        if any(t.head.i in heads and t.dep_.split(":")[0] in NP_MODS for t in side(edits[j])):
            return j
    return None

# Input: Edit objects of one group
# Output: The group operation: all M -> M, all U -> U, anything else -> R
def group_op(members):
    ops = {e.type.split(":", 1)[0] for e in members}
    return ops.pop() if len(ops) == 1 and ops <= {"M", "U"} else "R"

# Input 1: orig Doc
# Input 2: cor Doc
# Input 3: classified Edit objects of one sentence
# Output: the same edits, with group attributes on the members of groups
def group(orig, cor, edits):
    types = [edit_types(edit, edits, orig, cor) for edit in edits]
    # Union-find over edits that share a type (not the MORPH fallback) and
    # are linked by an agreement relation
    parent = list(range(len(edits)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    # Clitic reordering: a deleted and an inserted form of biti with the same
    # head word are one change, typed by comparing the two forms:
    # [me je poklicala -> sta me poklicala] (NUM)
    for i, j in reordered_biti(edits):
        types[i] = types[j] = pair_types(one_sided_word(edits[i]), one_sided_word(edits[j])) or {"MORPH"}
        parent[find(j)] = find(i)
    # A one-sided auxiliary / copula links to its verb / predicate whatever
    # the types: [∅ -> bi + dela -> delala] (MOOD), [∅ -> je + reče -> rekel]
    # (TENSE), [Pomemben mu je -> Pomembna sta mu] (NUM)
    aux = [bool(t) and is_one_sided_aux(e) for e, t in zip(edits, types)]
    for i in range(len(edits)):
        for j in range(i+1, len(edits)):
            shared = (types[i] & types[j]) - {"MORPH"} or \
                (types[i] and types[j] and (aux[i] or aux[j]))
            if shared and dep_linked(edits[i], edits[j], orig, cor):
                parent[find(j)] = find(i)
    # Agreement through an edit outside the groups that carries the same
    # change [ostali gospodari in lastniki -> ostale gospodarstvenike in
    # lastnike], or through an unchanged head [bili so daljši -> bil je daljši]
    for b, e in enumerate(edits):
        bt = set() if types[b] else bridge_types(e)
        if not bt: continue
        linked = [i for i in range(len(edits)) if types[i] & bt and dep_linked(e, edits[i], orig, cor)]
        for i in linked:
            for j in linked:
                if i < j and types[i] & types[j] & bt:
                    parent[find(j)] = find(i)
    for i in range(len(edits)):
        for j in range(i+1, len(edits)):
            if (types[i] & types[j]) - {"MORPH"} and sibling_linked(edits[i], edits[j], cor, edits):
                parent[find(j)] = find(i)
    # A preposition edit joins the CASE edit of the noun it attaches to:
    # [do -> v + odnosa -> odnosu]; its word label stays
    joined = set()
    for i, e in enumerate(edits):
        if not types[i] and is_prep_edit(e):
            j = prep_target(e, edits, types)
            if j is not None:
                parent[find(i)] = find(j)
                joined.add(i)
    groups = {}
    for i in range(len(edits)):
        if types[i] or i in joined:
            groups.setdefault(find(i), []).append(i)
    # Groups are numbered 0, 1, 2 ... in the order of their first member edit
    for g, members in enumerate(sorted(groups.values(), key=min)):
        es = [edits[i] for i in members]
        label = pick(set().union(*(types[i] for i in members)))
        # A case change together with a preposition change [do -> v + odnosa -> odnosu]
        if label == "CASE" and joined & set(members):
            label = "GOV"
        label = "M-" + label
        for e in es:
            e.group = g
            e.group_label = label
            e.group_op = group_op(es)
    return edits
