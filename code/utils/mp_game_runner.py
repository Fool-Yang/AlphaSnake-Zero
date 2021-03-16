from time import time

from utils.game import Game

class MPGameRunner:
    
    def __init__(self, height = 11, width = 11, snake_cnt = 4, health_dec = 1, game_cnt = 1):
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.health_dec = health_dec
        self.game_cnt = game_cnt
        self.games = {ID: Game(ID, height, width, snake_cnt, health_dec) for ID in range(game_cnt)}
        # log
        self.wall_collision = 0
        self.body_collision = 0
        self.head_collision = 0
        self.starvation = 0
        self.food_eaten = 0
        self.game_length = 0
    
    # Alice is the agent
    def run(self, Alice):
        t0 = time()
        games = self.games
        show = self.game_cnt == 1
        rewards = [None]*self.game_cnt
        
        # run all the games in parallel
        turn = 0
        while games:
            turn += 1
            # print information
            if len(games) == 1:
                print("Running the root game. On turn", str(turn) + "...")
            else:
                print("Concurrently running", len(games), "root games. On turn", str(turn) + "...")
            
            # ask for moves from the Agent
            ids = []
            for game_id in games:
                ids += games[game_id].get_ids()
            moves = Alice.make_moves(games, ids)
            moves_for_game = {game_id: [] for game_id in games}
            for i in range(len(moves)):
                moves_for_game[ids[i][0]].append(moves[i])
            
            # tic all games
            kills = set()
            for game_id in games:
                game = games[game_id]
                result = game.tic(moves_for_game[game_id], show)
                # if game ended
                if result != 0:
                    # log
                    self.wall_collision += game.wall_collision
                    self.body_collision += game.body_collision
                    self.head_collision += game.head_collision
                    self.starvation += game.starvation
                    self.food_eaten += game.food_eaten
                    self.game_length += game.game_length
                    rewards[game_id] = result
                    kills.add(game_id)
            # remove games that ended
            for game_id in kills:
                del games[game_id]
            
            print("Root game turn", str(turn), "finished. Total time spent:", time() - t0, end = "\n\n")
        
        # log
        self.wall_collision /= self.game_cnt
        self.body_collision /= self.game_cnt
        self.head_collision /= self.game_cnt
        self.starvation /= self.game_cnt
        self.food_eaten /= self.game_cnt
        self.game_length /= self.game_cnt
        return rewards

class MCTSMPGameRunner(MPGameRunner):

    def __init__(self, games):
        self.games = games
    
    # MCTSAlice is the agent
    def run(self, MCTSAlice, MCTS_depth):
        t0 = time()
        games = self.games
        rewards = {game_id: None for game_id in games}
        print("Running", len(games), "MCTS...")
        
        # run all the games in parallel
        turn = 0
        while games:
            turn += 1
            # ask for moves from the Agent
            ids = []
            for game_id in games:
                ids += games[game_id].get_ids()
            moves = MCTSAlice.make_moves(games, ids)
            moves_for_game = {game_id: [] for game_id in games}
            for i in range(len(moves)):
                moves_for_game[ids[i][0]].append(moves[i])
            
            # tic all games
            kills = set()
            for game_id in games:
                game = games[game_id]
                result = game.tic(moves_for_game[game_id])
                # if game ended or MCTS subgame max length reached
                if result != 0 or game.game_length >= MCTS_depth[game_id]:
                    rewards[game_id] = game.rewards
                    kills.add(game_id)
            # remove games that ended
            for game_id in kills:
                del games[game_id]
        
        print("MCTS epoch finished. Time spent:", time() - t0)
        return rewards
