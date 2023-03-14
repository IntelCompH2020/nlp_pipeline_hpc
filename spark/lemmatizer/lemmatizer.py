import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
import pyspark.sql.functions as F

def lemmatize(data, document_assembler, sentence_detector, sentence_tokenizer, config, verbose, new_col_name='lemmas'):
    """Adds a column with a list of lemmas."""
    first = True
    for lang in config["langs"].keys():
        if verbose:
            print(f"Processing lemmas in language {lang}")
        lemmatizer = LemmatizerModel.load(config["langs"][lang]["lemmatizer"]) \
            .setInputCols(["normalized"]) \
            .setOutputCol("lemma")

        normalizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized") \
            .setLowercase(True)

        pipeline = Pipeline(stages=[document_assembler, sentence_detector, sentence_tokenizer, normalizer, lemmatizer])

        columns_of_interest = data.columns + ['lemma.result']
        column_names = data.columns + [new_col_name]


        lang_data = data.filter(F.col("language")[0] == lang)
        output = pipeline.fit(lang_data).transform(lang_data)
        df = output.select(columns_of_interest).toDF(*column_names)

        if first:
            result = df
            first = False
        else:
            result = result.union(df)
    if verbose: result.show()
    return result
