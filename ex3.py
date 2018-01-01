from nltk.corpus import dependency_treebank
import random
import mst
from sparse_vector import sparse_vector

# read parsed data ('gold' parsed sentences) and add 'ROOT' node with 'ROOT' tag
parsed_sents=dependency_treebank.parsed_sents()
for sent in parsed_sents:
    sent.nodes[0].update({'word': 'ROOT','tag': 'ROOT','ctag': 'ROOT'})
# read taged data and add the word 'ROOT'
tagged_sents_orig=dependency_treebank.tagged_sents()
tagged_sents = []
for sent in tagged_sents_orig:
    tagged_sents.append([('ROOT', 'ROOT')] + sent)

# split train and test, from the parsed and frome the tagged-only sentences
train_tagged= tagged_sents[:int(len(parsed_sents) * 0.9)]
train_parsed= parsed_sents[:int(len(parsed_sents) * 0.9)]
test_parsed= parsed_sents[int(len(parsed_sents) * 0.9):]
test_tagged= tagged_sents[int(len(tagged_sents) * 0.9):]

# create set of all possible tags and words.
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

all_words,all_tags = get_all_possible_tags_and_words(tagged_sents)
tag2i = {pos: i for (i, pos) in enumerate(all_tags)}
word2i = {word: i for (i, word) in enumerate(all_words)}
N = len(all_words)
T = len(all_tags)

# return the index of the '1' in the words feature vector
def word_bigram_feature(w1,w2):
    return word2i[w1] * N+word2i[w2]

# return the index of the '1' in the POS tags feature vector
def tag_bigram_feature(w1,w2):
    return tag2i[w1] * T+tag2i[w2]

# return the index of the '1' in the distance feature vector, or None if this feature vector should not contain '1'
def distance_feature(i,j):
    ret = min(j-i-1,3)
    # if ret < 0:
    #     return None
    return min(j-i-1,3)

# feature function, contain only word-bigram and POS-bigram tags
def feature_function(w1, t1, w2, t2,i,j):
    w_feature = word_bigram_feature(w1,w2)
    t_feature = tag_bigram_feature(t1,t2)
    temp1=sparse_vector([w_feature],N**2)
    temp2=sparse_vector([t_feature],T**2)
    temp1.concatenate(temp2)
    return temp1

# full feature function, contain word-bigram feature, POS-bigram feature, and distance feature
def feature_function_w_dist(w1, t1, w2, t2,i,j):
    feature_vec = feature_function(w1, t1, w2, t2,i,j)
    d_feature = distance_feature(i, j)
    # if d_feature == None:
    #     temp3 = sparse_vector([], 4)
    # else:
    temp3 = sparse_vector([d_feature], 4)
    feature_vec.concatenate(temp3)
    return feature_vec

# the result is graph as dict of dict as the mst algorithem gets.
# a key in the dictionary are the index of the word in the sentence.
# graph[i][j] = -weight of the w_i ->w_j arch in the graph, the minus is because the mst search for minimum and we want maximum
def sentence_to_full_graph(feature_function, w, sentence):
    graph = dict()
    for i in range(len(sentence)):
        graph[i] = dict()
        for j in range(len(sentence)):
            if i!=j:
                weight = feature_function(sentence[i][0],sentence[i][1], sentence[j][0], sentence[j][1], i, j).sparse_dot_by_sparse(w)
                graph[i][j] = -weight
    return graph

# compute and sum all the feature-vectors of all the archs in a graph
def sum_tree(tree, tagged_sent, feature_function):
    ret=sparse_vector([],N**2+T++2)
    for node1 in tree:
        neighbours=tree[node1]
        for node2 in neighbours:
            feature_vec=feature_function(tagged_sent[node1][0], tagged_sent[node1][1], tagged_sent[node2][0], tagged_sent[node2][1], node1, node2)
            ret.add(feature_vec)
    return ret

# convert dependency-tree to tree of the type dict-in-dict, as the mst algo' return
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

# the perceptron learning algorithem
def perceptron(learning_rate=1,itertations=2, dist_feature=False):
    # the weight vector is sparse, for convinient and running time reasons.
    weight=sparse_vector([],N**2+T++2)
    rand_iter = list(range(len(train_parsed)))
    random.shuffle(rand_iter)

    if dist_feature:
        feature_function_to_use = feature_function
    else:
        feature_function_to_use = feature_function_w_dist

    w_sum = sparse_vector([],N**2+T++2)
    for i in range(itertations):
        for j in rand_iter:
            G = sentence_to_full_graph(feature_function_to_use, weight, train_tagged[j])
            T_opt=mst.mst(0,G)
            t = to_tree(train_parsed[j])
            temp=sum_tree(t, train_tagged[j], feature_function_to_use)
            temp.sub(sum_tree(T_opt, train_tagged[j],feature_function_to_use))
            temp.mult_by_scalar(learning_rate)
            weight.add(temp)
            w_sum.add(weight)
            w_sum.mult_by_scalar(1.0 / (len(train_tagged) * itertations))
    return w_sum
    # return weight

def score_sent(w_train,tagged_sent,feature_function,gold_tree):
    G = sentence_to_full_graph(feature_function, w_train, tagged_sent)
    T = mst.mst(0, G)
    gold_tree=to_tree(gold_tree)
    num_of_right_edges=0
    for node in gold_tree:
        if node in T.keys():
            neighbours=gold_tree[node].keys()
            for node2 in neighbours:
                if node2 in T[node].keys():
                    num_of_right_edges+=1
    return num_of_right_edges/len(tagged_sent)

def score(w_train, feature_function, test_gold, test_tag):
    sum_of_scores = 0
    for i in range(len(test_gold)):
        sum_of_scores += score_sent(w_train,test_tag[i],feature_function,test_gold[i])
    return sum_of_scores / len(test_gold)

import time
start_time = time.time()
w = perceptron(dist_feature=True)
print("--- %s seconds for learning ---" % (time.time() - start_time))
start_time = time.time()
print("score for learning with the distance feature function is ", score(w, feature_function_w_dist, test_parsed, test_tagged))
print("--- %s seconds for evaluation ---" % (time.time() - start_time))