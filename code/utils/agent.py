from numpy import power, arctanh, array, float32
from numpy.random import choice
from time import time

from utils.mp_game_runner import MCTSMPGameRunner

class Agent:
    
    def __init__(self, nnet, softmax_base = 100, training = False,
                 max_MCTS_depth = 4, max_MCTS_breadth = 128):
        self.nnet = nnet
        self.softmax_base = softmax_base
        self.training = training
        self.max_MCTS_depth = max_MCTS_depth
        self.max_MCTS_breadth = max_MCTS_breadth
        self.cached_values = {}
        self.total_rewards = {}
        self.visit_cnts = {}
        # record data for training
        if training:
            self.records = []
            self.values = []
    
    def make_moves(self, games, ids):
        cached_values = self.cached_values
        total_rewards = self.total_rewards
        visit_cnts = self.visit_cnts
        parallel = 32
        if self.max_MCTS_breadth < parallel:
            parallel = self.max_MCTS_breadth
        
        # MCTS
        for _ in range(self.max_MCTS_breadth//parallel):
            # make a subgame for each game
            MCTS_depth = {}
            parent_games = {}
            subgames = {}
            subgame_id = 0
            for game_id in games:
                # calculate the max MCTS depth for each game
                depth = self.max_MCTS_depth - (len(games[game_id].snakes) - 2)
                for _ in range(parallel):
                    MCTS_depth[subgame_id] = depth
                    parent_games[subgame_id] = game_id
                    subgames[subgame_id] = games[game_id].subgame(subgame_id)
                    subgame_id += 1
            # run MCTS subgames
            MCTSAlice = MCTSAgent(self.nnet, self.softmax_base, subgames,
                                  cached_values, total_rewards, visit_cnts)
            MCTS = MCTSMPGameRunner(subgames)
            t0 = time()
            rewards = MCTS.run(MCTSAlice, MCTS_depth)
            if self.training:
                print("MCTS epoch finished. Time spent:", time() - t0)
            
            # update edge stats
            for subgame_id in MCTSAlice.keys:
                for snake_id in MCTSAlice.keys[subgame_id]:
                    my_keys = MCTSAlice.keys[subgame_id][snake_id]
                    my_moves = MCTSAlice.moves[subgame_id][snake_id]
                    if not rewards[subgame_id][snake_id] is None:
                        estimated_reward = rewards[subgame_id][snake_id]
                        # back up
                        for i in range(len(my_keys) - 1, -1, -1):
                            key = my_keys[i]
                            move = my_moves[i]
                            visit_cnts[key][move] += 1.0
                            total_rewards[key][move] += estimated_reward
                            cached_values[key][move] = total_rewards[key][move]/visit_cnts[key][move]
                            estimated_reward = self.softermax(cached_values[key])@cached_values[key]
        
        V = [None]*len(ids)
        # store the index of the value in V a (game_id, snake_id) coresponds to
        value_index = {}
        for i in range(len(ids)):
            try:
                value_index[ids[i][0]][ids[i][1]] = i
            except KeyError:
                value_index[ids[i][0]] = {ids[i][1]: i}
        # set Q values based on the subgames' stats
        for subgame_id in MCTSAlice.keys:
            game_id = parent_games[subgame_id]
            for snake_id in MCTSAlice.keys[subgame_id]:
                first_key = MCTSAlice.keys[subgame_id][snake_id][0]
                V[value_index[game_id][snake_id]] = cached_values[first_key]
        
        # make moves
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
    
    # a softmax function with customized base
    def softermax(self, z):
        # the higher the base is, the more it highlights the higher ones
        normalized = power(self.softmax_base, arctanh(z))
        # in case all three cells are obstacles (-1.0), softmax will fail on 0.0/0.0
        sigma = sum(normalized)
        if sigma == 0.0:
            return array([1.0/3.0]*3, dtype = float32)
        else:
            return normalized/sigma
    
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
    
    # clear memory
    def clear(self):
        self.cached_values = {}
        self.total_rewards = {}
        self.visit_cnts = {}
        if self.training:
            self.records = []
            self.values = []

class MCTSAgent(Agent):
    
    def __init__(self, nnet, softmax_base, games, cached_values, total_rewards, visit_cnts):
        self.nnet = nnet
        self.softmax_base = softmax_base
        self.cached_values = cached_values
        self.total_rewards = total_rewards
        self.visit_cnts = visit_cnts
        self.keys = {i: {s.id: [] for s in games[i].snakes} for i in games}
        self.moves = {i: {s.id: [] for s in games[i].snakes} for i in games}
    
    def make_moves(self, games, ids):
        cached_values = self.cached_values
        total_rewards = self.total_rewards
        visit_cnts = self.visit_cnts
        V = [None]*len(ids)
        keys = [None]*len(ids)
        all_states = []
        
        # get states without duplicates
        i = 0
        for game_id in games:
            states = games[game_id].get_states()
            for state in states:
                key = state.tostring()
                keys[i] = key
                try:
                    cache = cached_values[key]
                    if not cache is None:
                        V[i] = cache
                except KeyError:
                    all_states.append(state)
                    # a new state to be stored
                    cached_values[key] = None
                i += 1
        
        # calculate values using the net
        if all_states:
            calculated_V = self.nnet.v(all_states)
            # assign values calculated by the net and store them into the cache
            j = 0
            for i in range(len(V)):
                if V[i] is None:
                    if cached_values[keys[i]] is None:
                        # the calculated Q values will be a prior
                        total_rewards[keys[i]] = calculated_V[j]
                        visit_cnts[keys[i]] = array([1.0, 1.0, 1.0], dtype = float32)
                        cached_values[keys[i]] = total_rewards[keys[i]]/visit_cnts[keys[i]]
                        j += 1
                    V[i] = cached_values[keys[i]]
        
        # make randomized moves
        pmfs = [self.softermax(v) for v in V]
        moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
        
        # update MCTS edge stats and record state keys and moves
        for i in range(len(ids)):
            game_id = ids[i][0]
            snake_id = ids[i][1]
            my_keys = self.keys[game_id][snake_id]
            my_moves = self.moves[game_id][snake_id]
            # back up
            estimated_reward = pmfs[i]@V[i]
            for j in range(len(my_keys) - 1, -1, -1):
                key = my_keys[j]
                move = my_moves[j]
                visit_cnts[key][move] += 1.0
                total_rewards[key][move] += estimated_reward
                cached_values[key][move] = total_rewards[key][move]/visit_cnts[key][move]
            my_keys.append(keys[i])
            my_moves.append(moves[i])
        return moves
