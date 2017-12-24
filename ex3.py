from nltk.corpus import dependency_treebank
import numpy as np
import sys
import random
from scipy.sparse import vstack
import mst
from sparse_vector import sparse_vector

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix

parsed_sents=dependency_treebank.parsed_sents()
for sent in parsed_sents:
    sent.nodes[0].update({'word': 'ROOT','tag': 'ROOT','ctag': 'ROOT'})
tagged_sents_orig=dependency_treebank.tagged_sents()
tagged_sents = []
for sent in tagged_sents_orig:
    tagged_sents.append([('ROOT', 'ROOT')] + sent)
train_sents=tagged_sents[:int(len(parsed_sents)*0.9)]
train=parsed_sents[:int(len(parsed_sents)*0.9)]
test=parsed_sents[int(len(parsed_sents)*0.9):]
def get_all_possible_tags_and_words(data_set):
    all_words = set()
    all_tags=set()
    for sen in data_set:
        for tagged_word in sen:
            all_words.add(tagged_word[0])
            all_tags.add(tagged_word[1])
    all_tags.add('ROOT')
    all_words.add('ROOT')
    all_words = list(all_words)
    all_tags = list(all_tags)
    all_tags.sort()
    all_words.sort()
    return all_words,all_tags

all_words,all_tags = get_all_possible_tags_and_words(train_sents)
tag2i = {pos: i for (i, pos) in enumerate(all_tags)}
word2i = {word: i for (i, word) in enumerate(all_words)}
N = len(all_words)
T = len(all_tags)

def word_bigram_feature(w1,w2):
    # feature_vec =dok_matrix((N ** 2, 1), dtype=np.int32)
    # feature_vec[word2i[w1] * N+word2i[w2]]=1
    return word2i[w1] * N+word2i[w2]

def tag_bigram_feature(w1,w2):
    # feature_vec =dok_matrix((T ** 2, 1), dtype=np.int32)
    # feature_vec[tag2i[w1] * T+tag2i[w2]]=1
    return tag2i[w1] * T+tag2i[w2]

#
def feature_function(w1, t1, w2, t2):
    w_feature = word_bigram_feature(w1,w2)
    t_feature = tag_bigram_feature(t1,t2)
    # vstack([w_feature, t_feature])
    return sparse_vector([w_feature,t_feature])


# the feature_function gets 2 words and return feature-vector; the weight is this vector multiply w;
# the result is graph as dict of dict as the mst wants.
# the keys in the dictionary are the index of the word in the sentence.
def sentence_to_full_graph(feature_function, w, sentence):
    graph = dict()
    for i in range(len(sentence)):
        graph[i] = dict()
        for j in range(len(sentence)):
            if i!=j:
                weight = feature_function(sentence[i][0],sentence[i][1], sentence[j][0], sentence[j][1]).sparse_dot(w) # todo tags
                graph[i][j] = -weight
    return graph
def sum_tree(tree,parsed_sent):
    ret=sparse_vector([])
    for node1 in tree:
        neighbours=tree[node1]
        for node2 in neighbours:
            feature_vec=feature_function(parsed_sent[node1][0], parsed_sent[node1][1], parsed_sent[node2][0],parsed_sent[node2][1])
            ret=ret.add(feature_vec)
    return ret

# w_z = np.zeros((N**2 + T**2,1))
# w_rand = np.random.randint(0,10, N**2 + T**2)
# print(tagged_sents[0][:3])
# G = sentence_to_full_graph(feature_function, w_rand, tagged_sents[0][:3])
# tr = mst.mst(0, G)
# print(G)
# print(tr)
def to_tree(prs_sent):
    ret = {}
    for w1 in prs_sent.nodes:
        ret[w1] = {}
        d = prs_sent.nodes[w1]['deps']
        for w2 in d['']:
            ret[w1][w2] = 0
        for w2 in d['ROOT']:
            ret[w1][w2] = 0
    return ret

t=to_tree(parsed_sents[0])
sum=sum_tree(t,train_sents[0])
print(t)
print("...........................")
print(train_sents[0])
print(".............................")
print(sum.vec)

def perceptron(learning_rate=1,itertations=2):
    weight=np.zeros(N**2+T++2)
    rand_iter = list(range(len(train)))
    random.shuffle(rand_iter)

    for i in range(itertations):
        for j in rand_iter:
            G = sentence_to_full_graph(feature_function, weight, train[j])
            T_opt=mst.mst(0,G)

            weight=weight+learning_rate*()




