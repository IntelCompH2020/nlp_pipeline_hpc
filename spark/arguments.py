from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import HfArgumentParser

@dataclass
class SparkArguments:
    """
    Spark arguments.
    """
    log_level: str = field(
        default="OFF",
        metadata={"help": "Verbosity level of the log messages. Options: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN."},
    )
    use_gpu: bool = field(
        default=False,
        metadata={"help": "Whether to enable GPU-computing or not."},
    )
    driver_memory: str = field(
        default="32G",
        metadata={"help": "Amount of memory to use for the driver process."},
    )
    executor_memory: str = field(
        default="32G",
        metadata={"help": "Amount of memory to use per executor process."},
    )
    max_result_size: str = field(
            default="0",
        metadata={"help": "Limit of total size of serialized results of all partitions for each Spark action in bytes. Should be at least 1M, or 0 for unlimited. Jobs will be aborted if the total size is above this limit."},
    )
    master_url: str = field(
        default="local[*]",
        metadata={"help": "The cluster manager to connect to. The default value runs Spark locally with as many worker threads as logical cores on your machine."},
    )
    buffer_max: str = field(
        default="2000M",
        metadata={"help": "Maximum allowable size of Kryo serialization buffer, in MiB unless otherwise specified. This must be larger than any object you attempt to serialize and must be less than 2048m. Increase this if you get a 'buffer limit exceeded' exception inside Kryo."},
    )
    config_file: str = field(
        default="config_spark_v2.json",
        metadata={"help": "Path to Spark's configuration file and all components of the pipeline."},
    )
    cache_folder: str = field(
        default="cache/pretrained",
        metadata={"help": "Location to download and extract pretrained Models and Pipelines."},
    )
    cluster_tmp_dir: str = field(
        default="tmp/storage",
        metadata={"help": "Location to use on a cluster for temporary files such as unpacking indexes for WordEmbeddings. By default, this is the location of hadoop.tmp.dir set via Hadoop configuration for Apache Spark."},
    )


@dataclass
class PipelineArguments:
    """
    Pipeline arguments.
    """
    data: str = field(
        default = "/gpfs/projects/shared/public/intelcomp/semantic/20220201/papers.parquet/part*.parquet",
        metadata={"help": "Path to the parquet files."},
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples. WARNING: If number of parts is high, the limit function is slow. Negative value to not limit."},
    )
    id_field: str = field(
        default ='id',
        metadata={"help": "Name of the field that contains a unique id code."},
    )
    fields_to_process: List[str] = field(
        default_factory = lambda: ["title", "paperAbstract"],
        metadata={"help": "List of fields to be processed by the pipeline. All fields will be concatenated."},
    )
    output_csv: str = field(
        default='spark_df.csv',
        metadata={"help": "Path of the output csv file."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Directory where the generated files will be stored."},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether the program should be verbose or not."},
    )
    lemmatize: bool = field(
        default=False,
        metadata={"help": "Whether to lemmatize the content of each document."},
    )
    keywords: bool = field(
        default=False,
        metadata={"help": "Whether to extract keywords from each document."},
    )
    min_ngrams_for_keywords: int = field(
        default=1,
        metadata={"help": "Minimum n-grams a keyword should have."},
    )
    max_ngrams_for_keywords: int = field(
        default=1,
        metadata={"help": "Maximum n-grams a keyword should have."},
    )
    num_keywords: int = field(
        default=1,
        metadata={"help": "Number of keywords to extract."},
    )
    extract_entities: bool = field(
        default=False,
        metadata={"help": "Whether to perform named-entity recognition on each document."},
    )
    embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to extract word embeddings."},
    )
    ngrams: bool = field(
        default=False,
        metadata={"help": "Whether to enable n-grams extraction."},
    )
    grams_size: int = field(
        default=1,
        metadata={"help": "Number of elements per n-gram, must be greater or equal to 1."},
    )
    enable_cumulative: bool = field(
        default=True,
        metadata={"help": "Whether to calculate just the actual n-grams."},
    )
    ngram_delimiter: str = field(
        default='_',
        metadata={"help": "Character to use to join tokens when N is greater than 1."},
    )
