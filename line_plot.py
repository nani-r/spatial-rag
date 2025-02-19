import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data
sparsity_levels = [75, 50, 25, 0]
vector_easy = [1, 5, 12, 20]
vector_medium = [5, 12, 18, 20]
vector_hard = [1, 5, 11, 16]
sparql_easy = [1, 5, 12, 20]
sparql_medium = [4, 12, 18, 20]
sparql_hard = [0, 0, 0, 0]


all_data = vector_easy + vector_medium + vector_hard + sparql_easy + sparql_medium + sparql_hard
y_min = min(all_data)
y_max = max(all_data)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Easy plot
axes[0].plot(sparsity_levels, vector_easy, marker='o', linestyle='-', label='SpatialRAG - Vector Similarity')
axes[0].plot(sparsity_levels, sparql_easy, marker='s', linestyle='--', label='Spatial RAG - SPARQL')
axes[0].set_title('Easy Questions')
axes[0].set_xlabel('Sparsity Level')
axes[0].set_ylabel('Questions Answered')
axes[0].set_xticks(sparsity_levels)
axes[0].set_ylim(y_min, y_max) 
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))  
axes[0].legend(loc='upper right')

# Medium plot
axes[1].plot(sparsity_levels, vector_medium, marker='o', linestyle='-', label='SpatialRAG - Vector Similarity')
axes[1].plot(sparsity_levels, sparql_medium, marker='s', linestyle='--', label='Spatial RAG - SPARQL')
axes[1].set_title('Medium Questions')
axes[1].set_xlabel('Sparsity Level')
axes[1].set_ylabel('Questions Answered')
axes[1].set_xticks(sparsity_levels)
axes[1].set_ylim(y_min, y_max)  
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True)) 
axes[1].legend(loc='upper right')

# Hard plot
axes[2].plot(sparsity_levels, vector_hard, marker='o', linestyle='-', label='SpatialRAG - Vector Similarity')
axes[2].plot(sparsity_levels, sparql_hard, marker='s', linestyle='--', label='SpatialRAG - SPARQL')
axes[2].set_title('Difficult Questions')
axes[2].set_xlabel('Sparsity Level')
axes[2].set_ylabel('Questions Answered')
axes[2].set_xticks(sparsity_levels)
axes[2].set_ylim(y_min, y_max)  # Set y-limits
axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
