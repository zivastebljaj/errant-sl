from importlib import import_module
import spacy
from errant.annotator import Annotator

# ERRANT version
__version__ = '3.0.2'

# spacy model family per language; used to build the default model name
_model_family = {"en": "core_web", "sl": "core_news"}

# Load an ERRANT Annotator object for a given language
# Input 3: An optional spacy model name override, e.g. "sl_core_news_md",
# to use a different sized/named spacy model than the language default.
# For pre-tagged CoNLL-U input only the vocab is used, so a blank pipeline is
# enough: load(lang, nlp=spacy.blank(lang))
def load(lang, nlp=None, model=None):
    # Make sure the language is supported
    supported = {"en", "sl"}
    if lang not in supported:
        raise Exception(f"{lang} is an unsupported or unknown language")

    # Load spacy (small model if no model supplied)
    nlp = nlp or spacy.load(model or f"{lang}_{_model_family[lang]}_sm", disable=["ner"])

    # Load language edit merger
    merger = import_module(f"errant.{lang}.merger")

    # Load language edit classifier
    classifier = import_module(f"errant.{lang}.classifier")
    # The classifier needs spacy
    classifier.nlp = nlp

    # Return a configured ERRANT annotator
    return Annotator(lang, nlp, merger, classifier)
