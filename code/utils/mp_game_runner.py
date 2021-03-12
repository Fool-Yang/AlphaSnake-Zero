from time import time

from utils.game import Game

class MPGameRunner:
    
    def __init__(self, height = 11, width = 11, snake_cnt = 4, health_dec = 1, game_cnt = 1, games = None):
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.health_dec = health_dec
        self.game_cnt = game_cnt
        if games is None:
            self.games = {ID: Game(ID, height, width, snake_cnt, health_dec) for ID in range(game_cnt)}
        else:
            self.games = games
        # log
        self.wall_collision = 0
        self.body_collision = 0
        self.head_collision = 0
        self.starvation = 0
        self.food_eaten = 0
        self.game_length = 0
    
    # Alice is the agent
    def run(self, Alice, MCTS_depth = None):
        t0 = time()
        games = self.games
        show = self.game_cnt == 1
        food_spawn_chance = 0.15 if MCTS_depth is None else 0.0
        rewards = [None]*self.game_cnt
        turn = 0
        while games:
            turn += 1
            if MCTS_depth is None:
                if len(games) == 1:
                    print("Running the game. On turn", str(turn) + "...")
                else:
                    print("Concurrently running", len(games), "games. On turn", str(turn) + "...")
            else:
                if len(games) == 1:
                    print("MCTS running the game. On step", str(turn) + "...")
                else:
                    print("MCTS running", len(games), "games. On step", str(turn) + "...")
            ids = []
            for game_id in games:
                ids += games[game_id].get_ids()
            moves = Alice.make_moves(games, ids)
            moves_for_game = {game_id: [] for game_id in games}
            for i in range(len(moves)):
                moves_for_game[ids[i][0]].append(moves[i])
            kills = set()
            for game_id in games:
                game = games[game_id]
                result = game.tic(moves_for_game[game_id], show, food_spawn_chance)
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
            for game_id in kills:
                del games[game_id]
            if MCTS_depth is None:
                print("Turn finished. Total time spent:", time() - t0)
            elif turn >= MCTS_depth:
                for game_id in games:
                    rewards[game_id] = games[game_id].rewards
                break
        # log
        self.wall_collision /= self.game_cnt
        self.body_collision /= self.game_cnt
        self.head_collision /= self.game_cnt
        self.starvation /= self.game_cnt
        self.food_eaten /= self.game_cnt
        self.game_length /= self.game_cnt
        return rewards
