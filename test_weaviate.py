import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv
from weaviate.classes.config import Configure, Property, DataType

load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_key = os.getenv("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_key),
)

print(client.is_ready())

collection_name = "IntentDetection6"

# if client.collections.exists(collection_name):
#     client.collections.delete(collection_name)
#     print(f"Deleted existing collection:'{collection_name}'")

client.collections.create(
    name=collection_name,
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(
        model="Snowflake/snowflake-arctic-embed-m-v1.5"
    ),
    properties=[
        Property(name="prompt", data_type=DataType.TEXT),
        Property(name="response", data_type=DataType.TEXT, skip_vectorization=True),
    ]
)

print(f"Successfully created collection: '{collection_name}'")
client.close()