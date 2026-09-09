import tiktoken
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

encoding = tiktoken.get_encoding("cl100k_base")

text = "My Name is Vivek Singh"

token = encoding.encode(text)
token_len = len(token)
# print(f"Token: {token}")
# print(f"Token Length: {token_len}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

tokens = tokenizer.encode(text)
tokens_len = len(tokens)
# print(f"Tokens: {tokens}")
# print(f"Token Length: {tokens_len}")

model = SentenceTransformer("all-MiniLM-L6-v2")

#Harcode Sentences

sentences = [
    "Artificial intelligence is transforming global industries.",
    "The solar system consists of eight distinct planets.",
    "Deep learning models require high-quality training datasets.",
    "Los algoritmos de aprendizaje automático aprenden patrones a partir de los datos.",
    "Machine learning algorithms learn patterns from data."
]

embeddings = model.encode(sentences)

print("Embedding matric shape:",embeddings.shape)

input_text = "Los algoritmos de aprendizaje automático aprenden patrones a partir de los datos."
input_embeddings = model.encode(input_text)

similarities = np.dot(embeddings, input_embeddings)

ranked_indices = np.argsort(similarities)[::-1]

#print Result
# for i, score in enumerate(similarities):
#     print (f"Similarity with sentences: {i+1}: {score:.4f}")


#print the ranked list
print("Ranked Sentences (Highest Similarity will first print):\n")
for rank, idx in enumerate(ranked_indices):
    print(f"{rank}.[{similarities[idx]:.4f}] {sentences[idx]}")

