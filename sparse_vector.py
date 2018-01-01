import numpy as np
import copy


class sparse_vector():
    def __init__(self, indexes, size):
        self.vec = {}
        self.size = size
        for idx in indexes:
            self.vec[idx] = 1

    def concatenate(self, other):
        for node in other.vec:
            self.vec[node + self.size] = other.vec[node]
        self.size += other.size

    def sparse_dot_by_sparse(self, other):
        sum = 0
        for key in self.vec:
            if key in other.vec:
                sum += self.vec[key] * other.vec[key]
        return sum

    def sparse_dot(self, weight_vec):
        sum = 0
        for key in self.vec:
            sum += self.vec[key] * weight_vec[key]
        return sum

    def mult_by_scalar(self, a):
        for key in self.vec:
            self.vec[key] *= a

    def add(self, other):
        # ret = sparse_vector([],self.size)
        # ret.vec = copy.deepcopy(self.vec)
        for key in other.vec:
            if key in self.vec:
                self.vec[key] += other.vec[key]
            else:
                self.vec[key] = other.vec[key]

    def sub(self, other):
        # other_copy = sparse_vector([],self.size)
        # other_copy.vec = copy.deepcopy(other.vec)
        other.mult_by_scalar(-1)
        self.add(other)
