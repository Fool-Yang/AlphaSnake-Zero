from numpy import power, array
from numpy.random import choice

class Agent:
    
    def __init__(self, nnet, softmax_base = None, game_and_snake_cnt = None):
        self.nnet = nnet
        self.softmax_base = softmax_base
        self.game_and_snake_cnt = game_and_snake_cnt
        # training mode (exlporative)
        if softmax_base:
            game_cnt = game_and_snake_cnt[0]
            snake_cnt = game_and_snake_cnt[1]
            self.records = {i: {j: [] for j in range(snake_cnt)} for i in range(game_cnt)}
            self.values = {i: {j: [] for j in range(snake_cnt)} for i in range(game_cnt)}
            self.moves = {i: {j: [] for j in range(snake_cnt)} for i in range(game_cnt)}
    
    def make_moves(self, states, ids = None):
        V = self.nnet.v(states)
        if self.softmax_base:
            pmfs = [self.softermax(v) for v in V]
            moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
            for i in range(len(states)):
                game_id = ids[i][0]
                snake_id = ids[i][1]
                # record the info for traininig
                self.records[game_id][snake_id].append(states[i])
                self.values[game_id][snake_id].append(V[i])
                self.moves[game_id][snake_id].append(moves[i])
        else:
            moves = self.argmaxs(V)
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
