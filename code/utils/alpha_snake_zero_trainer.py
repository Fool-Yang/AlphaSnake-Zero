from numpy import array, flip, float32
from math import ceil
from time import time

from utils.agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.mp_game_runner import MPGameRunner

class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                 self_play_games = 2048,
                 height = 11,
                 width = 11,
                 snake_cnt = 4,
                 TPU = None):
        
        self.self_play_games = self_play_games
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.TPU = TPU
    
    def train(self, nnet, name = "AlphaSnake", iteration = 0):
        # log
        if iteration == 0:
            f = open("log.csv", 'w')
            f.write("iteration, wall_collision, body_collision, head_collision, "
                     + "starvation, food_eaten, game_length\n")
            f.close()
        health_dec = 9
        while True:
            if iteration > 64:
                health_dec = 1
            elif iteration > 32:
                health_dec = 3
            # for training, all snakes are played by the same agent
            Alice = Agent(nnet, 100 + 2*iteration, True, (self.self_play_games, self.snake_cnt))
            gr = MPGameRunner(self.height, self.width, self.snake_cnt, health_dec, self.self_play_games)
            t0 = time()
            winner_ids = gr.run(Alice)
            print("Self play time", time() - t0)
            t0 = time()
            X = []
            V = []
            # collect training examples
            for game_id in Alice.records:
                for snake_id in Alice.records[game_id]:
                    x = Alice.records[game_id][snake_id]
                    v = Alice.values[game_id][snake_id]
                    m = Alice.moves[game_id][snake_id]
                    # assign estimated values
                    last_max = max(v[-1])
                    if snake_id == winner_ids[game_id]:
                        v[-1][m[-1]] += (1.0 - v[-1][m[-1]])
                    else:
                        v[-1][m[-1]] = 0.0
                    delta = max(v[-1]) - last_max
                    i = len(x) - 2
                    while i >= 0 and delta != 0.0:
                        last_max = max(v[i])
                        v[i][m[i]] += delta
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
            X = X[len(X) % 2048:]
            V = V[len(V) % 2048:]
            print("Data collecting time", time() - t0)
            # training
            nnet = nnet.copy_and_compile(TPU = self.TPU)
            t0 = time()
            nnet.train(X, V)
            print("Training time", time() - t0)
            nnet = nnet.copy_and_compile()
            # log
            log_list = [gr.wall_collision, gr.body_collision, gr.head_collision,
                        gr.starvation, gr.food_eaten, gr.game_length]
            log = str(iteration) + ', ' + ', '.join(map(str, log_list)) + '\n'
            f = open("log.csv", 'a')
            f.write(log)
            f.close()
            # save the model
            iteration += 1
            nnet.save(name + str(iteration))
    
    def mirror_states(self, states):
        # flip return a numpy.ndarray
        # need to return a list
        # otherwise X += does vector addition
        return list(flip(states, axis = 2))
    
    def mirror_values(self, values):
        return list(flip(values, axis = 1))
