import numpy as np
import copy
class sparse_vector():
    def __init__(self,indexes):
        self.vec ={}
        for idx in indexes:
            self.vec[idx]=1

    def sparse_dot(self, weight_vec):
        return np.sum(weight_vec[list(self.vec.keys())])
    def add(self,other):
        ret = sparse_vector([])
        ret.vec = copy.deepcopy(self.vec)
        for key in other.vec:
            if key in ret.vec:
                ret.vec[key] += other.vec[key]
            else:
                ret.vec[key] = other.vec[key]
        return ret
    def sub(self,other):
        other_copy = sparse_vector([])
        other_copy.vec = copy.deepcopy(self.vec)
        for key in other_copy.vec:
            other_copy.vec[key]=other.vec[key]*(-1)
        return self.add(other)
