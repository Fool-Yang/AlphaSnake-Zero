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
    
    # Alice and Bob are agents using different nets
    def run(self, Alice, Bob = None, Alice_snake_cnt = None):
        games = self.games
        show = self.game_cnt == 1
        if show:
            f = open("replay.rep", 'w')
            f.close()
        if Bob and Alice_snake_cnt is None:
            Alice_snake_cnt = games[0].snake_cnt//2
        winners = [None]*self.game_cnt
        while games:
            if Bob:
                states_A = []
                ids_A = []
                states_B = []
                ids_B = []
                for game_id in games:
                    states, ids = games[game_id].get_states_and_ids()
                    states_A += [states[i] for i in range(len(ids)) if ids[i][1] < Alice_snake_cnt]
                    ids_A += [ids[i] for i in range(len(ids)) if ids[i][1] < Alice_snake_cnt]
                    states_B += [states[i] for i in range(len(ids)) if ids[i][1] >= Alice_snake_cnt]
                    ids_B += [ids[i] for i in range(len(ids)) if ids[i][1] >= Alice_snake_cnt]
                moves = Alice.make_moves(states_A, ids_A) + Bob.make_moves(states_B, ids_B)
                ids = ids_A + ids_B
            else:
                states = []
                ids = []
                for game_id in games:
                    states_and_ids = games[game_id].get_states_and_ids()
                    states += states_and_ids[0]
                    ids += states_and_ids[1]
                moves = Alice.make_moves(states, ids)
            moves_for_game = {game_id: [] for game_id in games}
            for i in range(len(moves)):
                moves_for_game[ids[i][0]].append(moves[i])
            kills = set()
            for game_id in games:
                game = games[game_id]
                result = game.tic(moves_for_game[game_id], show)
                # if game ended
                if result != -1:
                    # log
                    self.wall_collision += game.wall_collision
                    self.body_collision += game.body_collision
                    self.head_collision += game.head_collision
                    self.starvation += game.starvation
                    self.food_eaten += game.food_eaten
                    self.game_length += game.game_length
                    winners[game_id] = result
                    kills.add(game_id)
                # to speed up the competing process
                # the team with snakes left wins
                elif Bob:
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
        
        # log
        self.wall_collision /= self.game_cnt
        self.body_collision /= self.game_cnt
        self.head_collision /= self.game_cnt
        self.starvation /= self.game_cnt
        self.food_eaten /= self.game_cnt
        self.game_length /= self.game_cnt
        return winners
