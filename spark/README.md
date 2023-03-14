# NLP pipeline

Collection of tools that apply the requested transformations to unstructured textual data, which will be used by the Intelcomp services (document classification, subcorpus generation, topic modeling...) as a preliminary step to process the datasets of interest. 

It has been designed to carry out standard text preprocessing tasks (e.g. n-grams detection, keywords extraction, lemmatization, etc) in a High Performance Computing environment, allowing the efficient and scalable processing of large amounts of documents. 

The final version of the pipeline will be deployed over the HPC infrastructure provided by the Barcelona Supercomputing Center and fully integrated with Intelcomp's Data Space. 


Please refer to deliverable D3.1 for more information.

## Main Components
- Language identification
- N-grams detection
- Keywords extraction
- Lemmatization
- Word embeddings
- Named Entity Recognition

## Input Data
Collection of parquet files. The user must specify the columns to be proccessed by command argument.

## Output Data
Collection of parquet files. The user must specify an output directory that will be created to store the generated files.
