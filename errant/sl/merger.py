from itertools import combinations, groupby
from re import sub
from string import punctuation
from rapidfuzz.distance import Indel
import spacy.symbols as POS
from errant.edit import Edit

# Input: An Alignment object
# Output: A list of Edit objects
def get_rule_edits(alignment):
    edits = []
    # Split alignment into groups of M, T and rest. (T has a number after it)
    for op, group in groupby(alignment.align_seq, 
            lambda x: x[0][0] if x[0][0] in {"M", "T"} else False):
        group = list(group)
        # Ignore M
        if op == "M": continue
        # T is always split
        elif op == "T":
            for seq in group:
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
        # Process D, I and S subsequence
        else:
            processed = process_seq(group, alignment)
            # Turn the processed sequence into edits
            for seq in processed: 
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
    return merge_rewritten_clauses(edits, alignment)

# Input 1: A sequence of adjacent D, I and/or S alignments
# Input 2: An Alignment object
# Output: A sequence of merged/split alignments
def process_seq(seq, alignment):
    # Return single alignments
    if len(seq) <= 1: return seq
    # Get the ops for the whole sequence
    ops = [op[0] for op in seq]
    # Merge all D xor I ops. (95% of human multi-token edits contain S).
    if set(ops) == {"D"} or set(ops) == {"I"}: return merge_edits(seq)

    # Get indices of all start-end combinations in the seq: 012 = 01, 02, 12
    combos = list(combinations(range(0, len(seq)), 2))
    # Sort them starting with largest spans first
    combos.sort(key = lambda x: x[1]-x[0], reverse=True)
    # Loop through combos
    for start, end in combos:
        # Ignore ranges that do NOT contain a substitution.
        if "S" not in ops[start:end+1]: continue
        # Get the tokens in orig and cor. They will now never be empty.
        o = alignment.orig[seq[start][1]:seq[end][2]]
        c = alignment.cor[seq[start][3]:seq[end][4]]
        # Case changes
        if o[-1].lower == c[-1].lower:
            # Merge first token I or D: [Cat -> The big cat]
            if start == 0 and ((len(o) == 1 and c[0].text[0].isupper()) or \
                    (len(c) == 1 and o[0].text[0].isupper())):
                return merge_edits(seq[start:end+1]) + \
                    process_seq(seq[end+1:], alignment)
            # Merge with previous punctuation: [, we -> . We], [we -> . We]
            if (len(o) > 1 and is_punct(o[-2])) or \
                    (len(c) > 1 and is_punct(c[-2])):
                return process_seq(seq[:end-1], alignment) + \
                    merge_edits(seq[end-1:end+1]) + \
                    process_seq(seq[end+1:], alignment)
        # Merge a phrase replaced by a single pronoun / determiner, or the
        # reverse: [te vere pa -> katere], [kateremu -> ki mu]
        if is_ref(o, c) or is_ref(c, o):
            return process_seq(seq[:start], alignment) + \
                merge_edits(seq[start:end+1]) + \
                process_seq(seq[end+1:], alignment)
        # Merge a multi-word connector replaced by a single one, or the
        # reverse: [za kar -> zato], [ker -> kljub temu da]
        if is_conn(o, c) or is_conn(c, o):
            return process_seq(seq[:start], alignment) + \
                merge_edits(seq[start:end+1]) + \
                process_seq(seq[end+1:], alignment)
        # Merge whitespace/hyphens: [acat -> a cat], [sub - way -> subway]
        s_str = sub("['-]", "", "".join([tok.lower_ for tok in o]))
        t_str = sub("['-]", "", "".join([tok.lower_ for tok in c]))
        # (RLC: also near matches with a hyphen inside: [e - pošto -> e-pošto])
        o_str = "".join([tok.lower_ for tok in o])
        c_str = "".join([tok.lower_ for tok in c])
        if s_str == t_str or (len(o) + len(c) <= 4 and
                (0 < o_str.find("-") < len(o_str)-1 or 0 < c_str.find("-") < len(c_str)-1) and
                1-Indel.normalized_distance(s_str, t_str) >= 0.75):
            return process_seq(seq[:start], alignment) + \
                merge_edits(seq[start:end+1]) + \
                process_seq(seq[end+1:], alignment)
    # Keep word-level edits, but merge runs of only D or only I
    out = []
    for op, group in groupby(seq, lambda x: x[0]):
        group = list(group)
        out += merge_edits(group) if op in {"D", "I"} else group
    return out

# Clause boundaries: punctuation tokens, or words with an attached , : ;
CLAUSE_PUNCT = {",", ":", ";", ".", "!", "?", "—", "–"}

# Input 1: A spacy Doc
# Input 2: A set of changed token indices in that Doc
# Output: A list of (start, end) clause spans that count as rewritten:
# >= 3 changed words, > 40 % of the words changed
def rewritten_clauses(doc, changed):
    clauses, start = [], 0
    for i, tok in enumerate(doc):
        if tok.text in CLAUSE_PUNCT or (len(tok.text) > 1 and tok.text[-1] in ",:;"):
            clauses.append((start, i+1))
            start = i+1
    clauses.append((start, len(doc)))
    out = []
    for a, b in clauses:
        words = [i for i in range(a, b) if not is_punct(doc[i])]
        n = sum(i in changed for i in words)
        if n >= 3 and n > 0.4*len(words): out.append((a, b))
    return out

# Input 1: Orig tokens of a substitution
# Input 2: Cor tokens of a substitution
# Output: True if the two sides are forms / spellings of the same word
def is_word_pair(o, c):
    if len(o) != 1 or len(c) != 1: return False
    o, c = o[0], c[0]
    return o.lemma_.lower() == c.lemma_.lower() or char_cost(o, c) > 0.75

# Input 1: A list of Edit objects
# Input 2: An Alignment object
# Output: The edits, with all edits that overlap a rewritten clause (orig or
# cor side) merged into one edit
def merge_rewritten_clauses(edits, alignment):
    # Only changes that are not word-level pairs count: deletions,
    # insertions, and substitutions between different words (a different
    # lemma and not a near-identical spelling)
    o_changed, c_changed = set(), set()
    o_subst, c_subst = set(), set()
    for op in alignment.align_seq:
        if op[0] == "S" and is_word_pair(alignment.orig[op[1]:op[2]],
                alignment.cor[op[3]:op[4]]):
            continue
        if op[0] in {"S", "D"}: o_changed.update(range(op[1], op[2]))
        if op[0] in {"S", "I"}: c_changed.update(range(op[3], op[4]))
        if op[0] == "S":
            o_subst.update(range(op[1], op[2]))
            c_subst.update(range(op[3], op[4]))
    # A firing clause needs a substitution (so it isn't only D or only I)
    o_fire = [cl for cl in rewritten_clauses(alignment.orig, o_changed)
        if any(cl[0] <= i < cl[1] for i in o_subst)]
    c_fire = [cl for cl in rewritten_clauses(alignment.cor, c_changed)
        if any(cl[0] <= i < cl[1] for i in c_subst)]
    if not o_fire and not c_fire: return edits
    # Each firing clause collects the edits that overlap it; clauses that
    # share an edit are joined
    groups = []
    for side, clauses in (("o", o_fire), ("c", c_fire)):
        for a, b in clauses:
            if side == "o":
                members = {i for i, e in enumerate(edits) if e.o_start < b and e.o_end > a}
            else:
                members = {i for i, e in enumerate(edits) if e.c_start < b and e.c_end > a}
            if not members: continue
            for g in [g for g in groups if g & members]:
                members |= g
                groups.remove(g)
            groups.append(members)
    # One span per group; absorb any other edit that falls inside a span
    spans = []
    for g in groups:
        spans.append([min(edits[i].o_start for i in g), max(edits[i].o_end for i in g),
            min(edits[i].c_start for i in g), max(edits[i].c_end for i in g)])
    spans.sort()
    merged = []
    for span in spans:
        if merged and span[0] < merged[-1][1]:
            last = merged[-1]
            merged[-1] = [last[0], max(last[1], span[1]), min(last[2], span[2]), max(last[3], span[3])]
        else:
            merged.append(span)
    def inside(e):
        return any(a <= e.o_start and e.o_end <= b and c <= e.c_start and e.c_end <= d
            for a, b, c, d in merged)
    out = [e for e in edits if not inside(e)]
    out += [Edit(alignment.orig, alignment.cor, span) for span in merged]
    return sorted(out, key=lambda e: (e.o_start, e.o_end, e.c_start))

# Input 1: Tokens on one side of a span
# Input 2: Tokens on the other side
# Output: True if a is a single pronoun / determiner and b has 2+ words,
# none of which is a form of the same word (except bound forms: zanj, vanjo)
def is_ref(a, b):
    if len(a) != 1 or a[0].pos not in {POS.PRON, POS.DET} or \
            sum(not is_punct(tok) for tok in b) < 2:
        return False
    if "Bound" in a[0].morph.get("Variant"): return True
    return not any(is_word_pair(a, b[i:i+1]) for i in range(len(b)))

# Input 1: Tokens on one side of a span
# Input 2: Tokens on the other side
# Output: True if a is a single connector and b has 2+ words, all of them
# function words
def is_conn(a, b):
    if len(a) != 1 or not (a[0].pos in {POS.CCONJ, POS.SCONJ} or
            a[0].dep_ in {"cc", "mark"}):
        return False
    words = [tok for tok in b if not is_punct(tok)]
    return len(words) >= 2 and all(tok.pos in {POS.ADP, POS.PRON, POS.DET,
        POS.PART, POS.SCONJ, POS.CCONJ, POS.ADV} for tok in words)

# Check whether token is punctuation
def is_punct(token):
    return token.pos == POS.PUNCT or token.text in punctuation

# Calculate the cost of character alignment; i.e. char similarity
def char_cost(a, b):
    return 1-Indel.normalized_distance(a.text, b.text)

# Merge the input alignment sequence to a single edit span
def merge_edits(seq):
    if seq: return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
    else: return seq