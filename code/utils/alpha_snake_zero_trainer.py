from numpy import array, flip, float32
from math import ceil
from time import time

from utils.agent import Agent
from utils.game import Game
from utils.alphaNNet import AlphaNNet

class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                 self_play_games = 1024,
                 pit_games = 128,
                 threshold = 0.53,
                 height = 11,
                 width = 11,
                 snake_cnt = 4,
                 TPU = None):
        
        self.self_play_games = self_play_games
        self.pit_games = pit_games
        self.threshold = threshold
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.TPU = TPU
    
    def train(self, nnet, name = "AlphaSnake", iteration = 0):
        current_nnet = nnet.copy_and_compile()
        # log
        new_generation = True
        if iteration == 0:
            f = open("log.csv", 'w')
            f.write('iteration, wall_collision, body_collision, head_collision, starvation, food_eaten, game_length\n')
            f.close()
        health_dec = 9
        while True:
            if iteration > 64:
                health_dec = 1
            elif iteration > 32:
                health_dec = 3
            # log
            wall_collision = 0
            body_collision = 0
            head_collision = 0
            starvation = 0
            food_eaten = 0
            game_length = 0
            # for training, all agents uses the same nnet
            Alice = Agent(current_nnet, range(self.snake_cnt), training = True, softmax_base = 100 + 2*iteration)
            X = []
            V = []
            t0 = time()
            # the loop below can use distributed computing
            for ep in range(self.self_play_games):
                # collect examples from a new game
                g = Game(self.height, self.width, self.snake_cnt, health_dec)
                winner_id = g.run(Alice)
                # log
                wall_collision += g.wall_collision
                body_collision += g.body_collision
                head_collision += g.head_collision
                starvation += g.starvation
                food_eaten += g.food_eaten
                game_length += g.game_length
                for snake_id in Alice.records:
                    x = Alice.records[snake_id]
                    v = Alice.values[snake_id]
                    m = Alice.moves[snake_id]
                    p = Alice.odds[snake_id]
                    # assign estimated values
                    last_max = max(v[-1])
                    if snake_id == winner_id:
                        v[-1][m[-1]] += (1.0 - v[-1][m[-1]])*p[-1]
                    else:
                        v[-1][m[-1]] = 0.0
                    delta = max(v[-1]) - last_max
                    i = len(x) - 2
                    while i >= 0 and delta != 0.0:
                        last_max = max(v[i])
                        v[i][m[i]] += delta*p[i]
                        # once the network is somewhat good this should never happen
                        if v[i][m[i]] < 0.0:
                            v[i][m[i]] = 0.0
                        elif v[i][m[i]] > 1.0:
                            v[i][m[i]] = 1.0
                        delta = max(v[i]) - last_max
                        i -= 1
                    # sampling
                    sample_x = x[i + 1:]
                    sample_v = v[i + 1:]
                    i = len(sample_x) + 1
                    # can result in an infinite loop if sample_length is too small
                    while i <= len(x):
                        sample_x.append(x[-i])
                        sample_v.append(v[-i])
                        i = ceil(1.5*i)
                    X += sample_x
                    V += sample_v
                    X += self.mirror_states(sample_x)
                    V += self.mirror_values(sample_v)
                Alice.clear()
            X = X[len(X) % 2048:]
            V = V[len(V) % 2048:]
            if new_generation:
                log_list = [wall_collision/self.self_play_games,
                            body_collision/self.self_play_games,
                            head_collision/self.self_play_games,
                            starvation/self.self_play_games,
                            food_eaten/self.self_play_games,
                            game_length/self.self_play_games]
                log = str(iteration) + ', ' + ', '.join(map(str, log_list)) + '\n'
                f = open("log.csv", 'a')
                f.write(log)
                f.close()
            print("Self play time", time() - t0)
            t0 = time()
            current_nnet = current_nnet.copy_and_compile(TPU = self.TPU)
            current_nnet.train(X, V)
            current_nnet = current_nnet.copy_and_compile()
            iteration += 1
            print("Training time", time() - t0)
            t0 = time()
            # compare new net with previous net
            score = self.compete(current_nnet, nnet)
            new_generation = score > self.threshold
            if new_generation:
                nnet = current_nnet
                nnet.save(name + str(iteration))
                print("Iteration", iteration, "beats the previouse version. score =", score, "\nIt is now the new champion!")
            else:
                print("Iteration", iteration, "failed to beat the previouse one. score =", score)
            print("Competing time", time() - t0, "\n")
    
    def mirror_states(self, states):
        # flip return a numpy.ndarray
        # need to return a list
        # otherwise X += does vector addition
        return list(flip(states, axis = 2))
    
    def mirror_values(self, values):
        return list(flip(values, axis = 1))
    
    def compete(self, new_net, old_net):
        number_of_new_nets = 2
        Alice = Agent(new_net)
        Bob = Agent(old_net)
        win = 0.0
        loss = 0.0
        for _ in range(self.pit_games):
            g = Game(self.height, self.width, self.snake_cnt)
            winner_id = g.run(Alice, Bob, sep=number_of_new_nets)
            if winner_id is None:
                win += 0.5
                loss += 0.5
            elif winner_id < number_of_new_nets:
                win += 1.0
            else:
                loss += 1.0
        return win/(win + loss)
