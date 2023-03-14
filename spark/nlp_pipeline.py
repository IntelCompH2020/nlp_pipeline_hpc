import os, glob, time, argparse, json, nltk, sparknlp
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StringType, IntegerType
from sparknlp.base import *
from sparknlp.annotator import *

from arguments import *
from lang_detection.lang_detection import *
from embeddings.embeddings import *
from lemmatizer.lemmatizer import *
from keywords.keywords import *
from ngram.ngram import *
from ner.ner import *


# prevents TensorFlow's INFO and WARNING messages from being printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def start_spark(SparkArguments, config):
    """ Creates a SparkSession object with the desired configuration."""
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .master(SparkArguments.master_url) \
        .config("spark.driver.memory", SparkArguments.driver_memory) \
        .config("spark.executor.memory", SparkArguments.executor_memory) \
        .config("spark.driver.maxResultSize", SparkArguments.max_result_size) \
        .config("spark.kryoserializer.buffer.max", SparkArguments.buffer_max) \
        .config("spark.jars", config["jars"]["gpu"] if SparkArguments.use_gpu else config["jars"]["cpu"]) \
        .config("spark.jsl.settings.pretrained.cache_folder", SparkArguments.cache_folder) \
        .config("spark.jsl.settings.storage.cluster_tmp_dir", SparkArguments.cluster_tmp_dir) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(SparkArguments.log_level)
    return spark


def show_df_info(df):
    print(f'Num documents: {df.count()}')
    print(f'Schema names: {df.schema.names}')
    df.printSchema()


def read_data(spark, data, id_field, fields_to_process, max_samples):
    """ Returns a Spark dataframe with one document per row and two columns: filename (path to the file) and text (content of the file)."""
    parts = spark.read.parquet(data)
    if max_samples >= 0:
        parts = parts.limit(max_samples).cache()
    parts = parts.select([id_field]+fields_to_process)
    return parts


def get_document_assembler(input_cols="text", output_col="document"):
    """ Returns a transformer that converts raw data into an annotation of type Document."""
    documentAssembler = DocumentAssembler() \
        .setInputCol(input_cols) \
        .setOutputCol(output_col)
    return documentAssembler


def get_tokenizer(input_cols=["document"], output_col="token"):
    """ Returns an annotator that identifies tokens with tokenization open standards."""
    tokenizer = Tokenizer() \
        .setInputCols(input_cols) \
        .setOutputCol(output_col)
    return tokenizer


def get_sentence_detector(input_cols="document", output_col="sentence"):
    """ Returns an annotator the finds sentence bounds in raw text."""
    sentenceDetector = SentenceDetector() \
        .setInputCols(input_cols) \
        .setOutputCol(output_col)
    return sentenceDetector


def get_stopwords(lang):
    """ Returns a list of stopwords for a given language."""
    stopwords = nltk.corpus.stopwords.words(lang)
    return stopwords


if __name__ == "__main__":
    parser = HfArgumentParser((SparkArguments, PipelineArguments)) 
    SparkArguments, PipelineArguments = parser.parse_args_into_dataclasses()

    with open(SparkArguments.config_file) as config_file:
        config = json.load(config_file)

    spark_session = start_spark(SparkArguments, config)
    print("Spark NLP version:   ", sparknlp.version())
    print("Apache Spark version:", spark_session.version)

    document_assembler = get_document_assembler(input_cols="text")
    sentence_detector  = get_sentence_detector()
    sentence_tokenizer = get_tokenizer(input_cols=["sentence"], output_col="token")
    document_tokenizer = get_tokenizer(input_cols=["document"], output_col="token")

    ### READ DATA ###
    if PipelineArguments.verbose:
        print(f"Reading parquet from {PipelineArguments.data}")
    spark_df = read_data(spark=spark_session, 
                         data=PipelineArguments.data, 
                         id_field=PipelineArguments.id_field, 
                         fields_to_process=PipelineArguments.fields_to_process,
                         max_samples=PipelineArguments.max_samples)

    if PipelineArguments.verbose:
        show_df_info(spark_df)
        spark_df.show(10, truncate=True)

    
    fields_to_process = PipelineArguments.fields_to_process
    spark_df = spark_df.withColumn("text", F.concat_ws('. ', *fields_to_process))
    document_assembler = get_document_assembler(input_cols="text")
    spark_df = lang_identifier(
            spark_df, 
            config, 
            document_assembler  = document_assembler,
            verbose             = PipelineArguments.verbose, 
            new_col_name="language")

    if PipelineArguments.ngrams:
        spark_df = ngram_identifier(
            data                = spark_df, 
            document_assembler  = document_assembler, 
            document_tokenizer  = document_tokenizer, 
            grams_size          = PipelineArguments.grams_size,
            enable_cumulative   = PipelineArguments.enable_cumulative, 
            ngram_delimiter     = PipelineArguments.ngram_delimiter,
            verbose             = PipelineArguments.verbose,
        )

    if PipelineArguments.lemmatize:
        spark_df = lemmatize(
            data               = spark_df,
            document_assembler = document_assembler, 
            sentence_detector  = sentence_detector, 
            sentence_tokenizer = sentence_tokenizer,
            config             = config,
            verbose            = PipelineArguments.verbose,
        )

    if PipelineArguments.keywords:
        spark_df = keywords_detector(
            data               = spark_df, 
            document_assembler = document_assembler, 
            document_tokenizer = document_tokenizer, 
            num_keywords       = PipelineArguments.num_keywords,
            min_ngrams         = PipelineArguments.min_ngrams_for_keywords,
            max_ngrams         = PipelineArguments.max_ngrams_for_keywords,
            config             = config,
            verbose            = PipelineArguments.verbose,
        )

    if PipelineArguments.extract_entities:
        spark_df = ner(
            data               = spark_df,
            document_assembler = document_assembler, 
            sentence_detector  = sentence_detector, 
            sentence_tokenizer = sentence_tokenizer,
            config             = config,
            verbose            = PipelineArguments.verbose,
        )

    if PipelineArguments.embeddings:
        spark_df = word_embeddings(
            data                = spark_df,
            document_assembler  = document_assembler,
            sentence_detector   = sentence_detector,
            sentence_tokenizer  = sentence_tokenizer,
            config              = config,
            verbose             = PipelineArguments.verbose,
        )

    spark_df = spark_df.drop("text")

    if PipelineArguments.verbose:
        print("Output dataframe")
        spark_df.show()

    if PipelineArguments.output_dir is not None:
        print(f"Writing parquet to {PipelineArguments.output_dir}")
        spark_df.write.parquet(PipelineArguments.output_dir)
