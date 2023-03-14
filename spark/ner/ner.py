import sparknlp
import pyspark.sql.functions as F
from sparknlp.base import *
from sparknlp.annotator import *

def ner(data, document_assembler, sentence_detector, sentence_tokenizer, config, verbose, new_col_name='ner'):
    """Adds a column with named entities from the input text."""
    first = True
    for lang in config["langs"].keys():
        if verbose:
            print(f"Processing ner for language {lang}")
        embeds_model = eval(config["langs"][lang]["embed_model_type"]).load(config["langs"][lang]["embeddings"]) \
            .setInputCols(["sentence",'token']) \
            .setOutputCol("embeddings")

        ner_model = eval(config["langs"][lang]["ner_model_type"]).load(config["langs"][lang]["ner"]) \
          .setInputCols(['sentence', 'token', 'embeddings']) \
          .setOutputCol("ner")
        
        ner_converter = NerConverter() \
            .setInputCols(['sentence', 'token', 'ner']) \
            .setOutputCol('entities')

        pipeline = Pipeline(stages=[document_assembler, sentence_detector, sentence_tokenizer, embeds_model, ner_model, ner_converter])

        columns_of_interest = data.columns + ['entities.result']
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
