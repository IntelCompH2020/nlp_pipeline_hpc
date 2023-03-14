import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

def ngram_identifier(data, document_assembler, document_tokenizer, grams_size, enable_cumulative, ngram_delimiter, verbose, new_col_name='ngrams'):
    """Adds a column with a list of n-grams."""

    if verbose:
        print("Processing ngrams")
    ngrams_cum = NGramGenerator() \
                .setInputCols(["token"]) \
                .setOutputCol("ngrams") \
                .setN(grams_size) \
                .setEnableCumulative(enable_cumulative) \
                .setDelimiter(ngram_delimiter)

    pipeline = Pipeline(stages=[document_assembler, document_tokenizer, ngrams_cum])

    output = pipeline.fit(data).transform(data)
    columns_of_interest = data.columns + ['ngrams.result']
    column_names = data.columns + [new_col_name]
    result = output.select(columns_of_interest).toDF(*column_names)
    if verbose: result.show()
    return result
