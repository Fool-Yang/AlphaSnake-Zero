from numpy import power, prod
from numpy.random import choice

class Agent:
    
    def __init__(self, nnet, snake_ids = None, training = False, softmax_base = None):
        self.nnet = nnet
        self.training = training
        self.softmax_base = softmax_base
        if training:
            self.records = {i:[] for i in snake_ids}
            self.values = {i:[] for i in snake_ids}
            self.moves = {i:[] for i in snake_ids}
            self.odds = {i:[] for i in snake_ids}
    
    def make_moves(self, states, snake_ids = None):
        V = self.nnet.v(states)
        if self.softmax_base:
            pmfs = [self.softermax(v) for v in V]
            moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
            if self.training:
                chance = prod([pmfs[i][moves[i]] for i in range(len(states))])
                for i in range(len(states)):
                    # record the info for traininig
                    self.records[snake_ids[i]].append(states[i])
                    self.values[snake_ids[i]].append(V[i])
                    self.moves[snake_ids[i]].append(moves[i])
                    self.odds[snake_ids[i]].append(chance/pmfs[i][moves[i]])
        else:
            moves = self.argmaxs(V)
            if self.training:
                for i in range(len(states)):
                    # record the info for traininig
                    self.records[snake_ids[i]].append(states[i])
                    self.values[snake_ids[i]].append(V[i])
                    self.moves[snake_ids[i]].append(moves[i])
                    self.odds[snake_ids[i]].append(1.0)
        return moves
    
    # a softmax-like function that highlights the higher values even more
    def softermax(self, z):
        # the higher the power base is, the more it highlights the higher ones
        normalized = power(self.softmax_base, z)
        return normalized/sum(normalized)
    
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
            self.odds[i] = []
