import torch
import torch.nn as nn
import numpy as np


def gen_M_N(data, num_attribute):
    """ generate M matrix, N matrix:
    Args:
        data (list(tuple(Image, list [num_attribute])))
    Returns:
        M matrix (m, n)
        N matrix (n)
    """
    M = np.zeros((num_attribute, num_attribute))
    N = np.zeros((num_attribute))
    for _, label in data:
        N += label
        for i in range(len(label)):
            if label[i] == 1:
                for j in range(len(label)):
                    if i != j and label[j] == 1:
                        M[i][j] += 1
    return M, N


def word_embedding(attribute_name, dim=300):
    """ https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    """
    word_to_ix = {y: x for x, y in enumerate(attribute_name)}
    embeds = nn.Embedding(len(attribute_name), dim)
    word2vec = torch.tensor([])
    for x in attribute_name:
        lookup_tensor = torch.tensor([word_to_ix[x]], dtype=torch.long)
        embed = embeds(lookup_tensor)
        word2vec = torch.cat((word2vec, embed), 0)
    word2vec = word2vec.detach()
    return word2vec
