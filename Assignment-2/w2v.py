from builtins import Exception
import math
import os
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

"""
Using word2vec model to illustrate linear algebra operations
on word vectors, such as computing similarity and inner products.
"""

def load_model(word2vec_model_path:str) -> KeyedVectors:
    try:
        fast_model_path = os.path.expanduser(word2vec_model_path)
        return KeyedVectors.load(fast_model_path, mmap='r')
    except Exception as e:
        print(f"Failed to load model in word2vec format: {e}")
    return None

def get_word_vector(model, word:str):
    try:
        v = model[word]
        assert len(v) == 50
        return v
    except KeyError:
        print(f"Word '{word}' not in the model vocabulary.")
    return None

# let the model compute a^T b / (||a|| * ||b||)
def similarity(model, word1:str, word2:str):
    try:
        score = model.similarity(word1, word2)
        return round(score, 7)
    except KeyError as e:
        print(f"One of the words '{word1}' or '{word2}' not in the model vocabulary: {e}")
    return None


# explicitly compute a^T b / (||a|| * ||b||)
def compute_similarity_raw(model, word1:str, word2:str):
    vec1 = get_word_vector(model, word1)
    vec2 = get_word_vector(model, word2)
    if vec1 is not None and vec2 is not None:
        score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return round(score, 7)
    return None


# compute a^T b, which is same as ||a|| * ||b|| * cos(theta)
# equivalent to np.dot(a, b)
def compute_inner_product_raw(model, word1:str, word2:str):
    vec1 = get_word_vector(model, word1)
    vec2 = get_word_vector(model, word2)
    if vec1 is not None and vec2 is not None and len(vec1) == len(vec2):
        # iterate over vector elements
        inner_product = 0.0
        for i in range(len(vec1)):
            inner_product += round(vec1[i] * vec2[i], 7)
        return round(inner_product, 7)
    return None


def test_similarity_diff_words(model):
    v1 = get_word_vector(model, "india")
    v2 = get_word_vector(model, "asia")
    assert v1 is not None and v2 is not None, "Word vectors retrieval failed."
    sim = similarity(model, "india", "asia")
    assert sim is not None, "Similarity computation failed."
    raw_sim = compute_similarity_raw(model, "india", "asia")
    assert raw_sim is not None, "Raw similarity computation failed."
    # print(f"similarity: {sim:0.7f}, raw similarity: {raw_sim:0.7f}")
    assert abs(sim - raw_sim) < 1e-6, "Computed similarity does not match the model's similarity."

def test_similarity_same_word(model):
    v1 = get_word_vector(model, "india")
    sim = similarity(model, "india", "india")
    raw_sim = compute_similarity_raw(model, "india", "india")
    assert raw_sim is not None, "Raw similarity computation failed."
    # print(f"similarity: {sim:0.7f}, raw similarity: {raw_sim:0.7f}")
    assert abs(sim - raw_sim) < 1e-6, "Computed similarity does not match the model's similarity."

def test_inner_product(model):
    v1 = get_word_vector(model, "india")
    inner_product = round(np.dot(v1, v1), 7)
    comp_inner_prod = compute_inner_product_raw(model, "india", "india")
    # print(f"inner product: {inner_product:0.7f}")
    # print(f"computed inner product: {comp_inner_prod:0.7f}")
    assert abs(inner_product - comp_inner_prod) < 1e-6, "Computed similarity does not match the model's similarity."

def test_most_similar(model, word:str):
    # can we find out if (king - man + woman = queen)?
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    # print(f"vector math: (king - man + woman) = {result[0][0]} (Confidence: {result[0][1]:.4f})")
    # print(result)
    # queen_vector = model['queen']
    # print(queen_vector.shape)  
    # print(queen_vector)

    # king_vector = model['king']
    # print(king_vector.shape)  
    # print(king_vector)

    # man_vector = model['man']
    # print(man_vector.shape)  
    # print(man_vector)
    # assert result[0][0] == 'queen', "Most similar word computation failed."


def norm_of_word1(model, word: str):
    v = model[word]
    for i in range (len(v)):
        norm += v[i] * v[i]

    norm = math.sqrt(norm)
    return norm


def norm_of_word2(model, word: str):
    v = model[word]
    for i in range (len(v)):
        norm += v[i] * v[i]

    norm = math.sqrt(norm)
    return norm

def words_text_comparison(model, text: str, text1: str):
    word2vec_model_path = "../glove50/glove_50_fast.wordvectors"
    model = load_model(word2vec_model_path)

    for i in range(len(text)):
        v = get_word_vector(model, text[i])
        if v is None:
            print(f"Word '{text[i]}' not found in the model vocabulary.")
            continue

    for i in range(len(text1)):
        v1 = get_word_vector(model, text1[i])
        if v1 is None:
            print(f"Word '{text1[i]}' not found in the model vocabulary.")
            continue

    norm1 = 0.0
    norm2 = 0.0

    for i in range (len(v)):
        norm1 += v[i] * v[i]

    for j in range (len(v1)):
        norm2 += v1[j] * v1[j]

        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)

    print(f"norm1: {norm1:0.7f}, norm2: {norm2:0.7f}")
    if(norm1 == norm2):
        print("Suraj")
        # if(norm1 == norm2): test_similarity_same_wordss(norm1)
        # test_similarity_diff_wordss(norm1, norm2)
    else: print ("Not Suraj")

def test_similarity_diff_wordss(model, norm1: str, norm2: str):
    v1 = get_word_vector(model, norm1)
    v2 = get_word_vector(model, norm2)
    assert v1 is not None and v2 is not None, "Word vectors retrieval failed."
    sim = similarity(model, norm1, norm2)
    assert sim is not None, "Similarity computation failed."
    raw_sim = compute_similarity_raw(model, norm1, norm2)
    assert raw_sim is not None, "Raw similarity computation failed."
    print(f"similarity: {sim:0.7f}, raw similarity: {raw_sim:0.7f}")
    assert abs(sim - raw_sim) < 1e-6, "Computed similarity does not match the model's similarity."

def test_similarity_same_wordss(model, norm1: str):
    v1 = get_word_vector(model, norm1)
    inner_product = round(np.dot(v1, v1), 7)
    comp_inner_prod = compute_inner_product_raw(model, norm1, norm1)
    print(f"inner product: {inner_product:0.7f}")
    print(f"computed inner product: {comp_inner_prod:0.7f}")
    assert abs(inner_product - comp_inner_prod) < 1e-6, "Computed similarity does not match the model's similarity."

if __name__ == "__main__":
    word2vec_model_path = "../glove50/glove_50_fast.wordvectors"
    model = load_model(word2vec_model_path)
    assert model is not None, "Model loading failed."
    test_similarity_diff_words(model)
    test_similarity_same_word(model)
    test_inner_product(model)
    test_most_similar(model, "king")
    # norm_of_word1(model, "Hello")
    # norm_of_word2(model, "Bye")
    s = ['the', 'lord', 'of', 'the', 'rings']
    v = ['game', 'of', 'thrones']
    words_text_comparison(model, s, v)


