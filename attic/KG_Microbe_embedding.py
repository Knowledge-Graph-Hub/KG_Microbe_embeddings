#!/usr/bin/env python
# coding: utf-8

# run this before running the script:
# source /global/cfs/cdirs/m4689/kg-microbe-projects/venv/bin/activate

# untar if data/merged-kg_edges.tsv and data/merged-kg_nodes.tsv are not present
import os 
if not os.path.exists("data/merged-kg_edges.tsv") or not os.path.exists("data/merged-kg_nodes.tsv"):
    # !tar -xzf data/merged-kg.tar.gz via system call
    os.system("tar -xzf data/merged-kg.tar.gz")    


from grape import Graph

kg_microbe = Graph.from_csv(
    edge_path="data/merged-kg_edges.tsv",
    node_path="data/merged-kg_nodes.tsv",
    node_list_separator="\t",
    edge_list_separator="\t",
    node_list_header=True,  # Always true for KG-Hub KGs
    edge_list_header=True,  # Always true for KG-Hub KGs
    nodes_column='id',  # Always true for KG-Hub KGs
    node_list_node_types_column='category',  # Always true for KG-Hub KGs
    node_types_separator='|',
    sources_column='subject',  # Always true for KG-Hub KGs
    destinations_column='object',  # Always true for KG-Hub KGs
    edge_list_edge_types_column='predicate',
    directed=False,
    name='KG Microbe',
    # Since we are not providing the node types and edge types lists, the order of
    # the node types and edge types would not be deterministic if we were to populate
    # the vocabulary in parallel. For this reason, we process them sequentially.
    load_node_list_in_parallel=False,
    load_edge_list_in_parallel=False,
)
kg_microbe


# In[4]:


kg_microbe = kg_microbe.remove_disconnected_nodes()
kg_microbe


# In[5]:


import grape
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
from grape.embedders import DeepWalkSkipGramEnsmallen
from embiggen.utils.abstract_models import EmbeddingResult
from typing import Union, List
from pathlib import Path

# Set the flag to determine if embeddings should be made
make_embeddings = False
embedding_file_path = Path("./DeepWalkSkipGramEnsmallen.tsv.gz")

if make_embeddings:
    # Generate embeddings using DeepWalkSkipGramEnsmallen
    sg = DeepWalkSkipGramEnsmallen(normalize_by_degree=True, embedding_size=200)
    degree_normalized_deepwalk_embedding = sg.fit_transform(kg_microbe)

    # Retrieve node embeddings
    node_embed = degree_normalized_deepwalk_embedding.get_all_node_embedding()

    # Save the first embedding as a compressed TSV file
    node_embed_df = pd.DataFrame(node_embed[0])
    node_embed_df.to_csv(embedding_file_path.with_name(f"DeepWalkSkipGramEnsmallen.tsv.gz"), sep="\t", compression='gzip')
else:  # already have embeddings, just read them in
    # Read the embeddings from a file if they already exist
    if embedding_file_path.exists():
        node_embed_df = pd.read_csv(embedding_file_path, sep="\t", compression='gzip')
        node_embed = [node_embed_df.to_numpy()]
    else:
        raise FileNotFoundError(f"Embedding file not found at {embedding_file_path}")

# Create an EmbeddingResult object with the loaded or generated embeddings
embedding_result = EmbeddingResult(
    embedding_method_name="DeepWalkSkipGramEnsmallen",
    node_embeddings=node_embed
)


from grape import GraphVisualizer
visualizer = GraphVisualizer(kg_microbe).fit_and_plot_all(degree_normalized_deepwalk_embedding)
visualizer.fit_and_plot_all(degree_normalized_deepwalk_embedding)
plt.savefig('../output/DeepWalkSkipGramEnsmallen.png')
plt.savefig('../output/DeepWalkSkipGramEnsmallen.pdf')
plt.close() 

