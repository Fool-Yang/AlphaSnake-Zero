from numpy import power, array, float32
from numpy.random import choice

from utils.mp_game_runner import MPGameRunner

class Agent:
    
    def __init__(self, nnet, softmax_base = 10, training = False, MCTS_breadth = 16, MCTS_depth = 4):
        self.nnet = nnet
        self.softmax_base = softmax_base
        self.training = training
        self.MCTS_breadth = MCTS_breadth
        self.MCTS_depth = MCTS_depth
        # training mode (exlporative)
        if training:
            self.records = []
            self.values = []
    
    def make_moves(self, games, ids):
        # the index of the value a game_id and a snake_id corespond to
        value_index = {}
        for i in range(len(ids)):
            try:
                value_index[ids[i][0]][ids[i][1]] = i
            except KeyError:
                value_index[ids[i][0]] = {ids[i][1]: i}
        # make many subgames for each game and record their parents
        parent_game = {}
        subgames = {}
        subgame_id = 0
        for game_id in games:
            for _ in range(self.MCTS_breadth):
                parent_game[subgame_id] = game_id
                subgames[subgame_id] = games[game_id].subgame(subgame_id)
                subgame_id += 1
        MCTSAlice = MCTSAgent(self.nnet, self.softmax_base, subgames)
        MCTS = MPGameRunner(game_cnt = len(subgames), games = subgames)
        rewards = MCTS.run(MCTSAlice, MCTS_depth = self.MCTS_depth)
        V = [array([1.0, 1.0, 1.0], dtype = float32) for _ in range(len(ids))]
        for subgame_id in MCTSAlice.values:
            game_id = parent_game[subgame_id]
            for snake_id in MCTSAlice.values[subgame_id]:
                v = MCTSAlice.values[subgame_id][snake_id]
                m = MCTSAlice.moves[subgame_id][snake_id]
                # assign refined values
                if not rewards[subgame_id][snake_id] is None:
                    last_max = rewards[subgame_id][snake_id]
                else:
                    last_max = v[-1][m[-1]]
                for i in range(len(v) - 1, -1, -1):
                    v[i][m[i]] = last_max
                    last_max = max(v[i])
                index = value_index[game_id][snake_id]
                for i in range(3):
                    if v[0][i] < V[index][i]:
                        V[index][i] = v[0][i]
        for i in range(len(V)):
            V[i] /= self.MCTS_breadth
        if self.training:
            pmfs = [self.softermax(v) for v in V]
            moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
            states = []
            for game_id in games:
                states += games[game_id].get_states()
            self.records += states
            self.values += V
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

class MCTSAgent(Agent):
    
    def __init__(self, nnet, softmax_base, games):
        self.nnet = nnet
        self.softmax_base = softmax_base
        # training data
        self.values = {i: {s.id: [] for s in games[i].snakes} for i in games}
        self.moves = {i: {s.id: [] for s in games[i].snakes} for i in games}
    
    def make_moves(self, games, ids):
        states = []
        for game_id in games:
            states += games[game_id].get_states()
        V = self.nnet.v(states)
        pmfs = [self.softermax(v) for v in V]
        moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
        for i in range(len(states)):
            game_id = ids[i][0]
            snake_id = ids[i][1]
            # record the info for traininig
            self.values[game_id][snake_id].append(V[i])
            self.moves[game_id][snake_id].append(moves[i])
        return moves
