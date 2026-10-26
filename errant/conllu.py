from spacy.tokens import Doc

# Read pre-tagged CoNLL-U input (https://universaldependencies.org/format.html)
# and turn each sentence into a spacy Doc, so the rest of ERRANT can use the
# external annotation instead of running spacy.

# A single CoNLL-U token row; keeps the source location for error messages
class Row:
    def __init__(self, cols, path, line_no):
        self.id, self.form, self.lemma, self.upos, self.xpos, \
            self.feats, self.head, self.deprel, self.deps, self.misc = cols
        self.path = path
        self.line_no = line_no
        # Whitespace after this word in the surface text
        self.space_after = "SpaceAfter=No" not in self.misc.split("|")

# Input: A path to a CoNLL-U file
# Output: A generator of sentences; each sentence is a list of Row objects
# Only syntactic words are kept: multi-word token ranges (1-2) and empty
# nodes (1.1) are skipped, since lemma/UPOS/deps live on the syntactic words.
def read_conllu(path):
    sent = []
    # Current multi-word token: (last word id, whitespace after the token)
    mwt = None
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            # Blank line ends a sentence
            if not line.strip():
                if sent: yield sent
                sent = []
                mwt = None
                continue
            # Comments; e.g. # sent_id, # text
            if line.startswith("#"): continue
            cols = line.split("\t")
            if len(cols) != 10:
                raise ValueError(f"{path}:{line_no}: expected 10 tab-separated "
                    f"columns, found {len(cols)}")
            # Multi-word token ranges; e.g. 3-4 vanj -> v + njega. The words
            # inside a token are not separated by whitespace and the last one
            # takes the token's SpaceAfter.
            if "-" in cols[0]:
                mwt = (cols[0].split("-")[1], "SpaceAfter=No" not in cols[9].split("|"))
                continue
            # Empty nodes
            if "." in cols[0]: continue
            row = Row(cols, path, line_no)
            if mwt:
                row.space_after = mwt[1] if row.id == mwt[0] else False
                if row.id == mwt[0]: mwt = None
            sent.append(row)
    # File may not end with a blank line
    if sent: yield sent

# Input: A CoNLL-U field value
# Output: The value, or the empty string if the field is unspecified (_)
def field(value):
    return "" if value == "_" else value

# Input 1: A spacy Vocab
# Input 2: A list of Row objects for one sentence
# Output: A spacy Doc carrying the CoNLL-U annotation
def to_doc(vocab, rows):
    words = [row.form for row in rows]
    # SpaceAfter=No in MISC; the last token never has a trailing space
    spaces = [row.space_after for row in rows]
    spaces[-1] = False
    kwargs = {
        "lemmas": [field(row.lemma) for row in rows],
        "pos": [field(row.upos) or None for row in rows],
        "tags": [field(row.xpos) for row in rows],
        "morphs": [field(row.feats) for row in rows]}
    # Dependency tree; only set if every token has a head. HEAD is 1-based,
    # 0 is the root, which spacy represents as a token headed by itself.
    if all(row.head.isdigit() for row in rows):
        heads = []
        for i, row in enumerate(rows):
            head = int(row.head)
            if head > len(rows):
                raise ValueError(f"{row.path}:{row.line_no}: HEAD {head} is "
                    f"outside the sentence ({len(rows)} words)")
            heads.append(head-1 if head else i)
        kwargs["heads"] = heads
        kwargs["deps"] = [field(row.deprel) for row in rows]
    try:
        return Doc(vocab, words=words, spaces=spaces, **kwargs)
    except Exception as e:
        raise ValueError(f"{rows[0].path}:{rows[0].line_no}: could not build "
            f"a spacy Doc for this sentence: {e}") from e
