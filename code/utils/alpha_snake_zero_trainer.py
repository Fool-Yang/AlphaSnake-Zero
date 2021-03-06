from math import ceil
from numpy import flip
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
        nnet = nnet.copy_and_compile()
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
            # self play
            # for training, all snakes are played by the same agent
            print("\nSelf playing games...")
            Alice = Agent(nnet, 2 + iteration, (self.self_play_games, self.snake_cnt))
            gr = MPGameRunner(self.height, self.width, self.snake_cnt, health_dec, self.self_play_games)
            winner_ids = gr.run(Alice, printing = True)
            print("\nCollecting data...")
            X = []
            V = []
            # collect training examples
            for game_id in Alice.records:
                for snake_id in Alice.records[game_id]:
                    x = Alice.records[game_id][snake_id]
                    v = Alice.values[game_id][snake_id]
                    m = Alice.moves[game_id][snake_id]
                    # assign estimated values
                    delta = 0.5
                    gamma = delta
                    sample_delta = 1.2
                    sample_index = 1
                    if snake_id == winner_ids[game_id]:
                        reward = 1.0
                    else:
                        reward = 0.0
                    last_max = reward
                    for i in range(len(x) - 1, -1, -1):
                        for j in range(3):
                            if j == m[i]:
                                v[i][j] = last_max
                            else:
                                v[i][j] += (reward - v[i][j])*gamma
                        last_max = max(v[i])
                        gamma *= delta
                        # sampling to avoid memory overflow
                        if len(x) - i == sample_index:
                            X.append(x[i])
                            V.append(v[i])
                            sample_index = ceil(sample_index*sample_delta)
                    # do this if not using the sampling method
                    # X += x
                    # V += v
            bs = 2048
            if len(X) < bs:
                bs = len(X)
            else:
                # chop for TPU
                X = X[len(X) % bs:]
                V = V[len(V) % bs:]
            X += self.mirror_states(X)
            V += self.mirror_values(V)
            # training
            nnet = nnet.copy_and_compile(TPU = self.TPU)
            t0 = time()
            nnet.train(X, V, batch_size = bs)
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
            print("\nSaving the model...")
            iteration += 1
            nnet.save(name + str(iteration))
    
    def mirror_states(self, states):
        # flip return a numpy.ndarray
        # need to return a list
        # otherwise X += does vector addition
        return list(flip(states, axis = 2))
    
    def mirror_values(self, values):
        return list(flip(values, axis = 1))
