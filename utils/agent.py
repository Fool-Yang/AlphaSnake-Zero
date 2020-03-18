from numpy import reshape, exp
from numpy.random import choice


class Agent:
    
    def __init__(self, nnet, snake_ids, training=False):
        self.nnet = nnet
        self.training = training
        if training:
            self.records = {i:[] for i in snake_ids}
            self.values = {i:[] for i in snake_ids}
            self.moves = {i:[] for i in snake_ids}
    
    def make_moves(self, states, snake_ids):
        X = reshape(states, (-1, len(states[0]), len(states[0][0]), 3))
        V = self.nnet.v(X)
        if self.training:
            moves = [choice([0, 1, 2], p=self.softmax(v)) for v in V]
            for i in range(len(states)):
                # record the info for traininig
                self.records[snake_ids[i]].insert(0, X[i])
                self.values[snake_ids[i]].insert(0, V[i])
                self.moves[snake_ids[i]].insert(0, moves[i])
        else:
            moves = self.argmaxs(V)
        return moves
    
    def softmax(self, z):
        return exp(z)/sum(exp(z))
    
    def argmaxs(self, Z):
        argmaxs = [-1] * len(Z)
        for i in range(len(Z)):
            if Z[i][0] > Z[i][1]:
                if Z[i][0] > Z[i][2]:
                    argmaxs[i] = 0
                else:
                    argmaxs[i] = 2
            else:
                if Z[i][1] > Z[i][2]:
                    argmaxs[i] = 1
                else:
                    argmaxs[i] = 2
        return argmaxs
    
    def clear(self):
        for i in self.records:
            self.records[i] = []
            self.values[i] = []
            self.moves[i] = []