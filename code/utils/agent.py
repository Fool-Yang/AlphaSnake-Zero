from numpy import power, array, float32
from numpy.random import choice

from utils.mp_game_runner import MCTSMPGameRunner

class Agent:
    
    def __init__(self, nnet, softmax_base = 10, training = False, max_MCTS_depth = 8, MCTS_breadth_factor = 4):
        self.nnet = nnet
        self.softmax_base = softmax_base
        self.training = training
        self.max_MCTS_depth = max_MCTS_depth
        self.MCTS_breadth_factor = MCTS_breadth_factor
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
        
        # make many subgames for each game
        parent_games = {}
        subgames = {}
        MCTS_depth = {}
        MCTS_breadth = {}
        subgame_id = 0
        for game_id in games:
            game = games[game_id]
            # calculate a good MCTS depth and breadth
            snake_cnt = len(game.snakes)
            depth = self.max_MCTS_depth//(snake_cnt - 1)
            breadth = self.MCTS_breadth_factor*snake_cnt*depth
            MCTS_depth[game_id] = depth
            MCTS_breadth[game_id] = breadth
            for _ in range(breadth):
                # record the parent game's id
                parent_games[subgame_id] = game_id
                subgames[subgame_id] = game.subgame(subgame_id)
                subgame_id += 1
        
        # run MCTS subgames
        MCTSAlice = MCTSAgent(self.nnet, self.softmax_base, subgames,
                              cached_values, total_rewards, visit_cnts)
        MCTS = MCTSMPGameRunner(subgames)
        rewards = MCTS.run(MCTSAlice, MCTS_depth, parent_games)
        
        V = [None]*len(ids)
        # the index of the value in V a (game_id, snake_id) coresponds to
        value_index = {}
        for i in range(len(ids)):
            try:
                value_index[ids[i][0]][ids[i][1]] = i
            except KeyError:
                value_index[ids[i][0]] = {ids[i][1]: i}
        # set Q values based on the subgames' stats
        for subgame_id in MCTSAlice.keys:
            game_id = parent_games[subgame_id]
            breadth = MCTS_breadth[game_id]
            for snake_id in MCTSAlice.keys[subgame_id]:
                my_keys = MCTSAlice.keys[subgame_id][snake_id]
                my_moves = MCTSAlice.moves[subgame_id][snake_id]
                # update the last edge stat if the reward was assigned to the snake
                if not rewards[subgame_id][snake_id] is None:
                    # back up
                    for i in range(len(my_keys) - 1, -1, -1):
                        last_key = my_keys[i]
                        last_move = my_moves[i]
                        visit_cnts[last_key][last_move] += 1.0
                        total_rewards[last_key][last_move] += rewards[subgame_id][snake_id]
                        cached_values[last_key][last_move] = (total_rewards[last_key][last_move]
                                                              /visit_cnts[last_key][last_move])
                V[value_index[game_id][snake_id]] = cached_values[my_keys[0]]
        
        # training mode (exlporative)
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
            i = 0
            j = 0
            while i < len(V):
                if V[i] is None:
                    if cached_values[keys[i]] is None:
                        # the calculated Q values will be a prior
                        total_rewards[keys[i]] = calculated_V[j]
                        visit_cnts[keys[i]] = array([1.0, 1.0, 1.0], dtype = float32)
                        cached_values[keys[i]] = total_rewards[keys[i]]/visit_cnts[keys[i]]
                        j += 1
                    V[i] = cached_values[keys[i]]
                i += 1
        
        # make randomized moves
        pmfs = [self.softermax(v) for v in V]
        moves = [choice([0, 1, 2], p = pmf) for pmf in pmfs]
        # update MCTS edge stats
        for i in range(len(ids)):
            game_id = ids[i][0]
            snake_id = ids[i][1]
            my_keys = self.keys[game_id][snake_id]
            my_moves = self.moves[game_id][snake_id]
            # back up
            for j in range(len(my_keys) - 1, -1, -1):
                last_key = my_keys[j]
                last_move = my_moves[j]
                visit_cnts[last_key][last_move] += 1.0
                total_rewards[last_key][last_move] += max(V[i])
                cached_values[last_key][last_move] = (total_rewards[last_key][last_move]
                                                      /visit_cnts[last_key][last_move])
            my_keys.append(keys[i])
            my_moves.append(moves[i])
        return moves
