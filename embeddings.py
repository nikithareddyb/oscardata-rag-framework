from sentence_transformers import SentenceTransformer
import numpy as np


def text_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text, normalize_embeddings=True)


def vector_similarity(vec1, vec2):
    return np.dot(np.squeeze(np.array(vec1)), np.squeeze(np.array(vec2)))


phrase1 = "Apple is a fruit"
embedding1 = text_embedding(phrase1)

phrase2 = "Apple iPhone is expensive"
embedding2 = text_embedding(phrase2)

phrase3 = "Mango is a fruit"
embedding3 = text_embedding(phrase3)

phrase4 = "There is a new Apple iPhone"
embedding4 = text_embedding(phrase4)

print(embedding1, len(embedding1))

print("\n", phrase1, "\n", phrase3, "\n", vector_similarity(embedding1, embedding3))
print("\n", phrase1, "\n", phrase4, "\n", vector_similarity(embedding1, embedding4))
print("\n", phrase2, "\n", phrase3, "\n", vector_similarity(embedding2, embedding3))
print("\n", phrase2, "\n", phrase4, "\n", vector_similarity(embedding2, embedding4))
