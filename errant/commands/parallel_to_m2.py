import argparse
from contextlib import ExitStack
import sys
import spacy
import errant
from errant import conllu

def main():
    # Parse command line args
    args = parse_args()
    print("Loading resources...")
    # Load Errant
    if args.format == "conllu":
        if args.tok:
            sys.exit("-tok cannot be used with -format conllu; "
                "CoNLL-U input is already tokenised.")
        if args.model:
            print("Warning: -model is ignored with -format conllu; "
                "the pre-tagged annotation is used instead.", file=sys.stderr)
        # Only the vocab is needed, so a blank pipeline is enough
        annotator = errant.load(args.lang, nlp=spacy.blank(args.lang))
    else:
        annotator = errant.load(args.lang, model=args.model)

    print("Processing parallel files...")
    # CoNLL-U is read and validated up front, before the output is opened
    if args.format == "conllu":
        pairs = read_conllu_pairs(args, annotator)
    else:
        pairs = read_text_pairs(args, annotator)
    with open(args.out, "w") as out_m2:
        for orig, cors in pairs:
            write_block(args, annotator, out_m2, orig, cors)

# Plain text input: one sentence per line, parsed with spacy
# Output: A generator of (orig Doc, list of cor Docs; None for a noop)
def read_text_pairs(args, annotator):
    # Process an arbitrary number of files line by line simultaneously. Python 3.3+
    # See https://tinyurl.com/y4cj4gth .
    with ExitStack() as stack:
        in_files = [stack.enter_context(open(i)) for i in [args.orig]+args.cor]
        # Process each line of all input files
        for line in zip(*in_files):
            # Get the original and all the corrected texts
            orig = line[0].strip()
            cors = line[1:]
            # Skip the line if orig is empty
            if not orig: continue
            # Parse orig with spacy
            orig = annotator.parse(orig, args.tok)
            # Parse the corrected texts; the same texts are noop edits
            cors = [cor.strip() for cor in cors]
            cors = [None if orig.text.strip() == cor else annotator.parse(cor, args.tok)
                for cor in cors]
            yield orig, cors

# Pre-tagged CoNLL-U input: sentences are paired by their order in each file
# Output: A list of (orig Doc, list of cor Docs; None for a noop)
def read_conllu_pairs(args, annotator):
    paths = [args.orig]+args.cor
    try:
        sents = [list(conllu.read_conllu(path)) for path in paths]
    except ValueError as e:
        sys.exit(f"Invalid CoNLL-U input: {e}")
    counts = [len(s) for s in sents]
    if len(set(counts)) > 1:
        sys.exit("CoNLL-U files have different numbers of sentences:\n" +
            "\n".join(f"  {n}\t{path}" for n, path in zip(counts, paths)) +
            "\nEach sentence in -orig must have exactly one counterpart in "
            "every -cor file. Check that the tagger did not re-segment "
            "sentences and that no correction was empty.")
    try:
        docs = [[annotator.parse_conllu(rows) for rows in sent] for sent in zip(*sents)]
    except ValueError as e:
        sys.exit(f"Invalid CoNLL-U input: {e}")
    # The same texts are noop edits; compared on the surface text (rebuilt
    # from SpaceAfter=No) to match plain text mode
    return [(orig, [None if orig.text == cor.text else cor for cor in cors])
        for orig, *cors in docs]

# Input 4: A parsed orig Doc
# Input 5: A list of parsed cor Docs; None for a cor identical to orig
# Writes one M2 block to out_m2
def write_block(args, annotator, out_m2, orig, cors):
    # Write orig to the output m2 file
    out_m2.write(" ".join(["S"]+[token.text for token in orig])+"\n")
    for cor_id, cor in enumerate(cors):
        # If the texts are the same, write a noop edit
        if cor is None:
            out_m2.write(noop_edit(cor_id)+"\n")
        # Otherwise, do extra processing
        else:
            # Align the texts and extract and classify the edits
            edits = annotator.annotate(orig, cor, args.lev, args.merge)
            # Loop through the edits
            for edit in edits:
                # Write the edit to the output m2 file
                out_m2.write(edit.to_m2(cor_id)+"\n")
    # Write a newline when we have processed all corrections for each line
    out_m2.write("\n")

# Parse command line args
def parse_args():
    parser=argparse.ArgumentParser(
        description="Align parallel text files and extract and classify the edits.\n",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [-h] [options] -orig ORIG -cor COR [COR ...] -out OUT")
    parser.add_argument(
        "-orig",
        help="The path to the original text (or CoNLL-U) file.",
        required=True)
    parser.add_argument(
        "-cor",
        help="The paths to >= 1 corrected text (or CoNLL-U) files.",
        nargs="+",
        default=[],
        required=True)
    parser.add_argument(
        "-out", 
        help="The output filepath.",
        required=True)
    parser.add_argument(
        "-format",
        help="The input file format.\n"
            "text: One sentence per line, parsed with spacy (default)\n"
            "conllu: Pre-tagged CoNLL-U from any tool; sentences are paired by\n"
            "order and spacy is not run. Uses syntactic words (MWT ranges skipped).",
        choices=["text", "conllu"],
        default="text")
    parser.add_argument(
        "-tok",
        help="Word tokenise the text using spacy (default: False).",
        action="store_true")
    parser.add_argument(
        "-lev",
        help="Align using standard Levenshtein (default: False).",
        action="store_true")
    parser.add_argument(
        "-merge",
        help="Choose a merging strategy for automatic alignment.\n"
            "rules: Use a rule-based merging strategy (default)\n"
            "all-split: Merge nothing: MSSDI -> M, S, S, D, I\n"
            "all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI\n"
            "all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I",
        choices=["rules", "all-split", "all-merge", "all-equal"],
        default="rules")
    parser.add_argument(
        "-lang",
        help="The language of the input text (default: en).",
        choices=["en", "sl"],
        default="en")
    parser.add_argument(
        "-model",
        help="Override the spacy model used to tag/parse the text, e.g. "
            "sl_core_news_md (default: <lang>_core_web_sm for en, "
            "<lang>_core_news_sm for sl).",
        default=None)
    args=parser.parse_args()
    return args

# Input: A coder id
# Output: A noop edit; i.e. text contains no edits
def noop_edit(id=0):
    return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||"+str(id)