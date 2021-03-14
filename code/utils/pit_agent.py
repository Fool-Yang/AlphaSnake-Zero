from numpy import power, array
from numpy.random import choice

class Agent:
    
    def __init__(self, nnet, game_and_snake_cnt = None):
        self.nnet = nnet
        self.game_and_snake_cnt = game_and_snake_cnt
    
    def make_moves(self, states, ids = None):
        V = self.nnet.v(states)
        moves = self.argmaxs(V) 
        return moves
    
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
