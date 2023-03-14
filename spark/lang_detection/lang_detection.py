import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

def lang_identifier(spark_df, config, verbose, document_assembler, new_col_name='language'):
    """Adds a column with the language code detected."""
    languageDetector = LanguageDetectorDL.load(config["lang_pipe"]) \
            .setInputCols(["document"]) \
            .setOutputCol("language")

    pipeline = Pipeline(stages=[document_assembler, languageDetector])
    output = pipeline.fit(spark_df).transform(spark_df)

    columns_of_interest = spark_df.columns + ['language.result']
    column_names = spark_df.columns + [new_col_name]

    result = output.select(columns_of_interest).toDF(*column_names)

    if verbose: result.show()

    return result
