from math import ceil
from numpy import flip
from time import time

from utils.agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.mp_game_runner import MPGameRunner

class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                 self_play_games,
                 max_MCTS_depth,
                 max_MCTS_breadth,
                 learning_rate,
                 learning_rate_decay,
                 height = 11,
                 width = 11,
                 snake_cnt = 4,
                 TPU = None):
        
        self.self_play_games = self_play_games
        self.max_MCTS_depth = max_MCTS_depth
        self.max_MCTS_breadth = max_MCTS_breadth
        self.lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.TPU = TPU
    
    def train(self, nnet, name = "AlphaSnake", iteration = 0):
        nnet = nnet.copy_and_compile()
        # log
        if iteration == 0:
            f = open("log.csv", 'a')
            f.write("new model " + name + "\n")
            f.write("iteration, wall_collision, body_collision, head_collision, "
                    + "starvation, food_eaten, game_length\n")
            f.close()
        health_dec = 9
        while True:
            if iteration > 32:
                health_dec = 1
            elif iteration > 8:
                health_dec = 3
            # self play
            # for training, all snakes are played by the same agent
            print("\nSelf playing games...")
            # the second arg is the softmax base (a snake with lower base is more explorative)
            Alice = Agent(nnet, 2 + iteration, True, self.max_MCTS_depth, self.max_MCTS_breadth)
            gr = MPGameRunner(self.height, self.width, self.snake_cnt, health_dec, self.self_play_games)
            gr.run(Alice)
            # log
            log_list = [gr.wall_collision, gr.body_collision, gr.head_collision,
                        gr.starvation, gr.food_eaten, gr.game_length]
            log = str(iteration) + ', ' + ', '.join(map(str, log_list)) + '\n'
            f = open("log.csv", 'a')
            f.write(log)
            f.close()
            # collect training examples
            X = Alice.records
            V = Alice.values
            Alice.clear()
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
            nnet = nnet.copy_and_compile(learning_rate = self.lr, TPU = self.TPU)
            t0 = time()
            nnet.train(X, V, batch_size = bs)
            print("Training time", time() - t0)
            nnet = nnet.copy_and_compile()
            # learning rate decay
            self.lr *= self.lr_decay
            X = None
            V = None
            # save the model
            iteration += 1
            print("\nSaving the model " + name + str(iteration) + "...")
            nnet.save(name + str(iteration))
    
    def mirror_states(self, states):
        # flip return a numpy.ndarray
        # need to return a list
        # otherwise X += does vector addition
        return list(flip(states, axis = 2))
    
    def mirror_values(self, values):
        return list(flip(values, axis = 1))
