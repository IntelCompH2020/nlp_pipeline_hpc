from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.training import CoNLL

# spark = sparknlp.start()
BASE_PATH = '/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/sparknlp/'
PATH_TO_JAR_v2_CPU = BASE_PATH + 'jars/spark-nlp-cpu-spark24-assembly-3.3.4.jar'
PATH_TO_JAR_v2_GPU = BASE_PATH + 'jars/spark-nlp-gpu-spark24-assembly-3.3.4.jar'


def start_spark(jar_file, driver_memory="32G", log_level="WARN"):
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .master("local[*]") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", driver_memory) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars", jar_file)\
        .config("spark.jsl.settings.pretrained.cache_folder", "sample_data/pretrained") \
        .config("spark.jsl.settings.storage.cluster_tmp_dir", "sample_data/storage")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    print("Spark NLP version:   ", sparknlp.version())
    print("Apache Spark version:", spark.version)
    print(spark)
    return spark

spark = start_spark(jar_file = PATH_TO_JAR_v2_CPU) # PATH_TO_JAR_v2_GPU if use gpu 

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)

training_data = CoNLL().readDataset(spark, './joined_eng.train')
training_data.show()

# RoBERTa embeddings
RoBERTa = RoBertaEmbeddings.load(BASE_PATH + 'embeddings/hf_models/en_roberta_sparknlp')\
    .setInputCols(["sentence",'token'])\
    .setOutputCol("RoBERTa")\

# NER tagger
nerTagger = NerDLApproach()\
    .setInputCols(["sentence", "token", "RoBERTa"])\
    .setLabelColumn("label")\
    .setOutputCol("ner")\
    .setMaxEpochs(5)\
    .setRandomSeed(0)\
    .setVerbose(1)\
    .setValidationSplit(0.2)\
    .setEvaluationLogExtended(True)\
    .setEnableOutputLogs(True)\
    .setIncludeConfidence(True)\
    .setTestDataset("test_withEmbeds.parquet")

# Test dataset
test_data = CoNLL().readDataset(spark, './eng.testa')
test_data = RoBERTa.transform(test_data)
test_data.show()
test_data.write.parquet("test_withEmbeds.parquet")

# Create a pipeline with these two annotators
ner_pipeline = Pipeline(stages = [RoBERTa, nerTagger])

# Train
ner_model = ner_pipeline.fit(training_data)

# Save model
ner_model.stages[1].write().save('NER_RoBERTa_20200221_plus')
