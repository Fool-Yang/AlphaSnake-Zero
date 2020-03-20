from numpy import power
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
        V = self.nnet.v(states)
        if self.training:
            moves = [choice([0, 1, 2], p=self.softermax(v)) for v in V]
            for i in range(len(states)):
                # record the info for traininig
                self.records[snake_ids[i]].insert(0, states[i])
                self.values[snake_ids[i]].insert(0, V[i])
                self.moves[snake_ids[i]].insert(0, moves[i])
        else:
            moves = self.argmaxs(V)
        return moves
    
    # a softmax-like function that highlights the higher values even more
    def softermax(self, z):
        # the higher the power base is, the more it highlights the higher ones
        normalized = power(100, z)
        return return normalized/sum(normalized)
    
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