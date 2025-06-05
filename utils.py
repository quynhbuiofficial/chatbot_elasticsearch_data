import time
from pprint import pprint
from elasticsearch import Elasticsearch
import os

def get_es_client(max_retries = 1, sleep_time = 1):
    i = 0
    while i < max_retries:
        try:
            # es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "R5nRHqCLO7HQ9-8O5P*n"))
            es_ip = os.getenv("ELASTICSEARCH_IP")
            print("ES IP: ", es_ip)
            es = Elasticsearch(f"http://{es_ip}:9200")
            client_info = es.info()
            print("Connected to Elasticsearch! \n")
            pprint(client_info)
            return es
        except Exception:
            pprint('Could not connect to ElasticSearch, trying,,,,')
            time.sleep(sleep_time)
            i += 1
    raise ConnectionError("Failed to connect to Elasticsearch after multiple attempts.")
