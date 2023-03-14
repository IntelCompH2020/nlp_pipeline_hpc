import os
import sys
import json
import spacy
import argparse
import fasttext
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from parser import Parser
from pathlib import Path
from keybert import KeyBERT
from typing import List, Optional
from yaml import dump, load, safe_load
from dataclasses import dataclass, field
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import spacy_universal_sentence_encoder.util as encoder_utils
import textacy.extract.basics as textacy
from spacy.lang.char_classes import (
        ALPHA, 
        ALPHA_LOWER, 
        ALPHA_UPPER, 
        CONCAT_QUOTES, 
        LIST_ELLIPSES, 
        LIST_ICONS
    )
# Add progress bar to pandas
tqdm.pandas()

# Silence fasttext warnings
fasttext.FastText.eprint = lambda x: None

# Paths to translation models
translation_models_path = {
    "es": "translation_models/es-en",
    "de": "translation_models/de-en",
    "fr": "translation_models/fr-en",
    "sv": "translation_models/sv-en",
    "gl": "translation_models/el-en"
}

@dataclass
class PipelineArguments:
    parquet_file: str = field(
        default=None,
        metadata={"help": "Path to parquet file"}
    )
    config_file: str = field(
        default=None,
        metadata={"help": "Path to yaml file with configuration parameters."}
    )
    columns_to_process: List[str] = field(
        default=None,
        metadata={"help": "The specific text columns of the parquet table to process."},
    )
    id_column_name: str = field(
        default=None,
        metadata={"help": "The name of the id column."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Path to the output dir."},
    )
    do_translate: bool = field(
        default=False, 
        metadata={"help": "Whether to translate documents in Spanish, German, French, Swedish or Greek to English. Note that if a document is not translated the pipelinewill only process english documents."}
    )
    do_pos: bool = field(
        default=False, 
        metadata={"help": "Whether to extract Part Of Speech features."}
    )
    do_lemmas: bool = field(
        default=False, 
        metadata={"help": "Whether to extract lemmas features."}
    )
    do_ner: bool = field(
        default=False, 
        metadata={"help": "Whether to extract ner features."}
    )
    do_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to extract docuement embeddings."}
    )
    do_keywords: bool = field(
        default=False,
        metadata={"help": "Whether to extract keywords."}
    )
    keywords_model_path: str = field(
        default="keywords/all-minilm-l6-v2",
        metadata ={"help": "Path to the keywords model."}
    )
    lang_id_model_path: str = field(
        default="lang_detection/lid.176.bin", 
        metadata={"help": "Name of the language identification model."}
    )
    ngrams: List[int] = field(
        default=None,
        metadata={"help": "List of integers indicating the N-grams. For instance [2, 3] will produce bigrams and trigrams."}
    )
    ngrams_filter_stops: bool = field(
        default=False,
        metadata={"help": "If True, remove ngrams that start or end with a stop word."}
    )
    ngrams_filter_punct: bool = field(
        default=False,
        metadata={"help": "If True, remove ngrams that contain any punctuation-only tokens."}
    )
    ngrams_filter_nums: bool = field(
        default=False,
        metadata={"help": "If True, remove ngrams that contain any numbers or number-like tokens (e.g. 10, ‘ten’)."}
    )
    ngrams_include_pos: List[str] = field(
        default=None,
        metadata={"help": "Remove ngrams if any constituent tokens’ part-of-speech tags ARE NOT included in this param. Note that the part-of-speech tags come from spacy."}
    )
    ngrams_exclude_pos: List[str] = field(
        default=None,
        metadata={"help": "Remove ngrams if any constituent tokens’ part-of-speech tags ARE included in this param. Note that the part-of-speech tags come from spacy."}
    )
    ngrams_min_freq: int = field(
        default=1,
        metadata={"help": "Remove ngrams that occur in doclike fewer than min_freq times."}
    )
    device: int = field(
        default=-1,
        metadata={"help": "Device to use for the transformers models. None will default to gpu if available, -1 means only cpu."}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Default is False. If set to True, debugging information will activate."}
    )
    
# def set_path_to_encoder_model(path):
#     """ Specify path to local spacy_universal_sentence_encoder. """
#     encoder_utils.configs["en_use_lg"]["use_model_url"] = path
# 
# def get_language(text, model):
#     lang = model.predict(text, k=1)[0][0]
#     return lang

def assertions(args):
    assert os.path.exists(args.keywords_model_path), "--keywords_model_path should point to a directory."
    assert os.path.isfile(args.lang_id_model_path), "--lang_id_model_path should point to a .bin file."
    assert args.lang_id_model_path.endswith(".bin"), "--lang_id_model_path should point to a .bin file."
    assert isinstance(args.do_pos,bool), "--do_pos must be a boolean: either True or False."
    assert isinstance(args.do_ner,bool), "--do_ner must be a boolean: either True or False."
    assert isinstance(args.do_lemmas,bool), "--do_lemmas must be a boolean: either True or False."
    assert isinstance(args.do_keywords,bool), "--do_keywords must be a boolean: either True or False."
    assert isinstance(args.do_translate,bool), "--do_translate must be a boolean: either True or False."
    assert isinstance(args.do_embeddings,bool), "--do_embeddings must be a boolean: either True or False."
    

def read_config_file(yaml_file):
    """ Read YAML configuration file. """

    assert yaml_file.endswith(".yaml"), "The configuration file should be YAML."

    print(f"###CONFIGURATION###\n{open(yaml_file).read()}")

    with open(yaml_file, "r") as yf:
        config = safe_load(yf)

    return config if config is not None else []

def get_doc(text, nlp):
    """ Apply the nlp from spacy if text is not None"""
    doc = None
    try:
        if text is not None:
            doc = nlp(text)
    except Exception as e:
        print(e, file=sys.stderr)
    return doc

def get_pos(doc):
    """ Get the pos tags of a given text. """
    pos = None
    try:
        if doc is not None:
            pos = [token.pos_ for token in doc]
    except Exception as e:
        print(e, file=sys.stderr)
    return pos

def get_lemmas(doc):
    """ Lemmatize a given text. """
    lemmas = None
    try:
        if doc is not None:
            lemmas = [token.lemma_ for token in doc]
    except Exception as e:
        print(e, file=sys.stderr)
    return lemmas

def get_entities(doc):
    """ Get the named entities tags of a given text. """
    ner = None
    try:
        if doc is not None:
            ner = strit([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
    except Exception as e:
        print(e, file=sys.stderr)
    return ner

def strit(t):
    """ Convert a list or tuple to str recursively. """
    return list(map(strit, t)) if isinstance(t, (list, tuple)) else str(t)

def get_keywords(text, keywords_model, keyphrase_ngram_range=(1,2), stop_words=None):
    """ Get the keywords from the text """
    keywords = None
    try:
        if text is not None:
            keywords = strit(keywords_model.extract_keywords(text, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words))
    except Exception as e:
        print(e, file=sys.stderr)
    return keywords

def get_ngrams(doc, n_value=3, filter_stops=False, filter_punct=True, filter_nums=False, include_pos=None, exclude_pos=None, min_freq=1):
    """ Extract ngrams from the given text. 
        From the textacy documentation:
        Parameters
            n – Number of tokens included per n-gram; for example, 2 yields bigrams and 3 yields trigrams. 
                If multiple values are specified, then the collections of n-grams are concatenated together; 
                for example, (2, 3) yields bigrams and then trigrams.
            filter_stops – If True, remove ngrams that start or end with a stop word.
            filter_punct – If True, remove ngrams that contain any punctuation-only tokens.
            filter_nums – If True, remove ngrams that contain any numbers or number-like tokens (e.g. 10, ‘ten’).
            include_pos – Remove ngrams if any constituent tokens’ part-of-speech tags ARE NOT included in this param.
            exclude_pos – Remove ngrams if any constituent tokens’ part-of-speech tags ARE included in this param.
            min_freq – Remove ngrams that occur in doclike fewer than min_freq times
    """
    ngrams = None
    try:
        if doc is not None:
            ngrams = [str(ngram) for ngram in textacy.ngrams(doc, 
                n_value, 
                filter_stops=filter_stops, 
                filter_punct=filter_punct, 
                filter_nums=filter_nums, 
                include_pos=include_pos, 
                exclude_pos=exclude_pos, 
                min_freq=min_freq)]
    except Exception as e:
        print(e, file=sys.stderr)
    return ngrams

def get_avg_word_embeddings(doc):
    """ Extract spacy vector """
    embeddings = None
    try:
        if doc is not None:
            embeddings = doc.vector
    except Exception as e:
        print(e, file=sys.stderr)
    return embeddings

def custom_tokenizer(nlp):
    """ Custom tokenizer to keep hyphenized words together. """
    # source: https://stackoverflow.com/questions/58105967/spacy-tokenization-of-hyphenated-words
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

def extract_text_from_iterable(text):
    out_text = None
    try:
        out_text = text
        if isinstance(text, (list, np.ndarray, np.generic)):
            out_text = text[0]
    except Exception as e:
        print(e, file=sys.stderr)
    return out_text

def get_language(text, lang_id_model):
    """
    """
    lang = None
    try:
        if text is not None:
            lang = lang_id_model.predict(text, k=1)[0][0][-2:]
    except Exception as e:
        print(e, file=sys.stderr)
    return lang

def translate(text, lang, translation_models, do_translate=True):
    """
    """
    en_text = None
    try:
        if lang in translation_models or lang == 'en':
            if lang != 'en' and do_translate:
                en_text = translation_models[lang](text, truncation=True)[0]["translation_text"]
            else:
                en_text = text
    except Exception as e:
        print(e, file=sys.stderr)
    return en_text

def main():

    # Parse arguments
    print("Parsing arguments")
    parser = Parser(PipelineArguments)
    pipe_args = parser.parse_args_into_dataclasses()[0]

    if pipe_args.config_file:
        # Read configuration file
        if pipe_args.config_file is not None:
            config = read_config_file(pipe_args.config_file)

        # Assign config values to pipe_args
        for arg in vars(pipe_args):
            if arg in config and config[arg] is not None:
                exec(f"pipe_args.{arg}=config['{arg}']", locals(), globals())

    # check parameters
    if pipe_args.output_dir is None:
        dirname = os.path.basename(os.path.dirname(pipe_args.parquet_file))
        pipe_args.output_dir = os.path.join(os.path.dirname(os.path.dirname(pipe_args.parquet_file)), dirname + "_nlp")


    # Load language identfication model
    print("Loading language identification model")
    try:
        lang_id_model = fasttext.load_model(pipe_args.lang_id_model_path)
    except Exception as e:
        raise Exception("--lang_id_model_path has to point to a fasttext language identification model.")

    # Load spacy disabling useless components
    disabled_features = ['parser', 'attribute_ruler', 'senter', 'tok2vec', 'ner', 'tagger', 'lemmatizer']
    if pipe_args.do_ner:
        disabled_features = list(filter(lambda e: not e in ['attribute_ruler', 'tok2vec', 'ner'], disabled_features))
    if pipe_args.do_pos or pipe_args.ngrams_include_pos or pipe_args.ngrams_exclude_pos:
        disabled_features = list(filter(lambda e: not e in ['attribute_ruler', 'tok2vec', 'tagger'], disabled_features))
    if pipe_args.do_lemmas:
        disabled_features = list(filter(lambda e: not e in ['attribute_ruler', 'tok2vec', 'tagger', 'lemmatizer'], disabled_features))
    if pipe_args.do_embeddings:
        disabled_features = list(filter(lambda e: not e in ['tok2vec'], disabled_features))
    print("Loading Spacy model")
    nlp = spacy.load("en_core_web_sm", disable=disabled_features)
    nlp.tokenizer = custom_tokenizer(nlp)

    # check if gpu is available
    if pipe_args.device is None:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            pipe_args.device = -1
            print("No cuda device available using cpu.")
        else:
            pipe_args.device = torch.cuda.current_device()
            print(f"Using gpu device named {torch.cuda.get_device_name(pipe_args.device)}.")

    # Load translation models
    translation_models = {}
    if pipe_args.do_translate:
        print("Loading translation models")
        for lang in tqdm(translation_models_path):
            tokenizer = AutoTokenizer.from_pretrained(translation_models_path[lang])
            tokenizer.model_max_length=460
            model = AutoModelForSeq2SeqLM.from_pretrained(translation_models_path[lang])
            translation_models[lang] = pipeline(f"translation_{lang}_to_en", model=model, tokenizer=tokenizer, device=pipe_args.device)

    # Load KeyBERT model
    if pipe_args.do_keywords:
        print("Loading keywords model")
        try:
            keywords_model =KeyBERT(pipe_args.keywords_model_path)
        except Exception as e:
            raise Exception("--keywords_model_path has to point to a valid keyword extraction model.")
                
    # Read parquet file
    print("Reading parquet file")
    df = pd.read_parquet(pipe_args.parquet_file, columns=[pipe_args.id_column_name] + pipe_args.columns_to_process)
#     df = df[]
    print("Parquet shape:", df.shape)

    # For each column process the text rows
    df_out = pd.DataFrame()
    df_out[pipe_args.id_column_name] = df[pipe_args.id_column_name]
    for column_name in pipe_args.columns_to_process:
        print(f"Processing column {column_name}")
        print("\tExtracting text from iterable")
        df_out[column_name + "_unfolded-text"] = df[column_name].progress_apply(extract_text_from_iterable)
        print("\tIdentifying original language")
        df_out[column_name + "_original-language"] = df_out[column_name + "_unfolded-text"].progress_apply(lambda text: get_language(text, lang_id_model))
        print("\tTranslating to english")
        df_out[column_name + "_en-text"] = df_out.progress_apply(lambda row: translate(row[column_name + "_unfolded-text"], row[column_name + "_original-language"], translation_models, pipe_args.do_translate), axis=1)
        print("\tApplying spacy nlp")
        df_out[column_name + "_doc"] = df_out[column_name + "_en-text"].progress_apply(lambda text: get_doc(text, nlp))
        if pipe_args.do_pos:
            print("\tExtracting pos")
            df_out[column_name + "_pos"] = df_out[column_name + "_doc"].progress_apply(get_pos)
        if pipe_args.do_lemmas:
            print("\tExtracting lemmas")
            df_out[column_name + "_lemmas"] = df_out[column_name + "_doc"].progress_apply(get_lemmas)
        if pipe_args.do_ner:
            print("\tExtracting entities")
            df_out[column_name + "_ner"] = df_out[column_name + "_doc"].progress_apply(get_entities)
        if pipe_args.ngrams is not None:
            for ngram in pipe_args.ngrams:
                print(f"\tExtracting {ngram}-grams")
                df_out[column_name + f"_{ngram}-grams"] = df_out[column_name + "_doc"].progress_apply(
                        lambda doc: get_ngrams(
                            doc,
                            n_value=ngram, 
                            filter_stops=pipe_args.ngrams_filter_stops, 
                            filter_punct=pipe_args.ngrams_filter_punct, 
                            filter_nums=pipe_args.ngrams_filter_nums, 
                            include_pos=pipe_args.ngrams_include_pos, 
                            exclude_pos=pipe_args.ngrams_exclude_pos, 
                            min_freq=pipe_args.ngrams_min_freq
                        )
                    )
        if pipe_args.do_embeddings:
            print("\tExtracting embeddings")
            df_out[column_name + "_embeddings"] = df_out[column_name + "_doc"].progress_apply(get_avg_word_embeddings)
        if pipe_args.do_keywords:
            print("\tExtracting keywords")
            df_out[column_name + "_keywords"] = df_out[column_name + "_en-text"].progress_apply(lambda text: get_keywords(text, keywords_model))
        print("\tDropping unnecessary columns")
        df_out = df_out.drop([column_name + extra for extra in ["_unfolded-text", "_doc"]], axis=1)

    if pipe_args.debug:
        print(df_out)
        for n_example in range(df_out.shape[0]):
            print("\n--------------------------------------------------------\n")
            print(f"Example {n_example}:")
            for column in df_out.columns:
                print(f"- {column}:")
                print(f"\t{df_out.iloc[n_example][column]}")
                print("\t--------------")
            import code
            # add the following line where you want to debug
            code.interact(local=dict(globals(), **locals()))

    print("Opening output dir")
    Path(pipe_args.output_dir).mkdir(parents=True, exist_ok=True)
    cat_output_dir_path = os.path.join(pipe_args.output_dir, os.path.basename(pipe_args.parquet_file))
    print(f"Saving parquet file as {cat_output_dir_path}")
    df_out.to_parquet(cat_output_dir_path) 

    

if __name__ == "__main__":
    main()
