import nltk, sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
import pyspark.sql.functions as F

def keywords_detector(data, document_assembler, document_tokenizer, num_keywords, min_ngrams, max_ngrams, config, verbose, new_col_name='keywords'):
    """Adds a column with a list of keywords."""
    first = True
    for lang in config["langs"].keys():
        if verbose:
            print(f"Processing keywords for language {lang}")

        nltk.data.path = ["keywords/nltk_data"]
        stopwords = nltk.corpus.stopwords.words(lang)
        keywords = YakeKeywordExtraction() \
            .setInputCols("token") \
            .setOutputCol("keywords") \
            .setMinNGrams(min_ngrams) \
            .setMaxNGrams(max_ngrams) \
            .setNKeywords(num_keywords) \
            .setStopWords(stopwords)

        pipeline = Pipeline(stages=[document_assembler, document_tokenizer, keywords])

        columns_of_interest = data.columns + ['keywords.result']
        column_names = data.columns + [new_col_name]

        lang_spec_data = data.filter(F.col("language")[0] == lang)
        output = pipeline.fit(lang_spec_data).transform(lang_spec_data)
        df = output.select(columns_of_interest).toDF(*column_names)
        df = df.withColumn("keywords", F.udf(set)(df.keywords))

        if first:
            result = df
            first = False
        else:
            result = result.union(df)
    if verbose: result.show()
    return result
