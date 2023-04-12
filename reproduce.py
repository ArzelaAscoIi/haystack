import os
from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import BM25Retriever, FileTypeClassifier, MarkdownConverter, PreProcessor
from haystack.pipelines import Pipeline

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

# run elasticsearch
#  docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:latest


document_store = OpenSearchDocumentStore(host=host, username="admin", password="admin", index="document")
file_type_classifier = FileTypeClassifier()
markdown_converter = MarkdownConverter(add_frontmatter_to_meta=True, extract_headlines=True)
preprocessor = PreProcessor(
    split_by="word", split_length=100, split_overlap=10, split_respect_sentence_boundary=True, language="en"
)
retriever = BM25Retriever(document_store=document_store, top_k=20)

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
indexing_pipeline.add_node(
    component=markdown_converter, name="MarkdownConverter", inputs=["FileTypeClassifier.output_1"]
)
indexing_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["MarkdownConverter"])
indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["Preprocessor"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])

# this rases the exception
# failed to parse field [completion_time] of type [boolean] in document with id '9218900bb7f6122fcf72f3f84c484d27'.
# Preview of field's value: '20 min'", 'caused_by': {'type': 'illegal_argument_exception',
# 'reason': 'Failed to parse value [20 min] as only [true] or [false] are allowed.'}}
results = indexing_pipeline.run(
    file_paths=["02_Finetune_a_model_on_your_data.txt", "07_RAG_Generator.txt", "21_Customizing_PromptNode.txt"]
)
