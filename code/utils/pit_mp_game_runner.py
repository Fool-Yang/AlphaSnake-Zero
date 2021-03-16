from time import time

from utils.game import Game
from utils.mp_game_runner import MCTSMPGameRunner

class MPGameRunner:
    
    def __init__(self, height = 11, width = 11, snake_cnt = 4, health_dec = 1, game_cnt = 1):
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.health_dec = health_dec
        self.game_cnt = game_cnt
        self.games = {ID: Game(ID, height, width, snake_cnt, health_dec) for ID in range(game_cnt)}
    
    # Alice and Bob are agents using different nets
    def run(self, Alice, Bob, Alice_snake_cnt = None):
        t0 = time()
        games = self.games
        show = self.game_cnt == 1
        if Alice_snake_cnt is None:
            Alice_snake_cnt = games[0].snake_cnt//2
        winners = [None]*self.game_cnt
        turn = 0
        while games:
            turn += 1
            ids = []
            for game_id in games:
                ids += games[game_id].get_ids()
            moves_A = Alice.make_moves(games, ids)
            moves_B = Bob.msake_moves(games, ids)
            moves_for_game = {game_id: [] for game_id in games}
            for i in range(len(ids)):
                if ids[i][1] < Alice_snake_cnt:
                    moves_for_game[ids[i][0]].append(moves_A[i])
                else:
                    moves_for_game[ids[i][0]].append(moves_B[i])
            kills = set()
            for game_id in games:
                game = games[game_id]
                result = game.tic(moves_for_game[game_id], show)
                # if game ended
                if result != 0:
                    for i in range(len(result)):
                        if result[i] == 1.0:
                            winners[game_id] = i
                    kills.add(game_id)
                else:
                    # to speed up the competing process
                    # the team with snakes left wins
                    snakes = games[game_id].snakes
                    A = False
                    B = False
                    for snake in snakes:
                        A = A or snake.id < Alice_snake_cnt
                        B = B or snake.id >= Alice_snake_cnt
                    if not A or not B:
                        winners[game_id] = snakes[0].id
                        kills.add(game_id)
            for game_id in kills:
                del games[game_id]
        return winners
