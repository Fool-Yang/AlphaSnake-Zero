from numpy import array, flip
from time import time

from utils.agent import Agent
from utils.game import Game
from utils.alphaNNet import AlphaNNet

# https://web.stanford.edu/~surag/posts/alphazero.html


class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                numEps=256,
                competeEps=256,
                threshold=0.28,
                height=11,
                width=11,
                snake_cnt=4):
        
        self.numEps = numEps
        self.competeEps = competeEps
        self.threshold = threshold
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
    
    def train(self, nnet, iter = 0):
        # for training, all agents uses the same nnet
        # unless we want to use a evolution algorithm
        while True:
            iter += 1
            new_nnet = nnet.copy()
            Alice = Agent(new_nnet, range(self.snake_cnt), training=True)
            X = []
            V = []
            t0 = time()
            # the loop below can use distributed computing
            for ep in range(self.numEps):
                # collect examples from a new game
                g = Game(self.height, self.width, self.snake_cnt)
                winner_id = g.run(Alice)
                for snake_id in Alice.records:
                    v = Alice.values[snake_id]
                    m = Alice.moves[snake_id]
                    # assign estimated values
                    if snake_id == winner_id:
                        v[0][m[0]] += (1.0 - Alice.values[snake_id][0][m[0]])/3.0
                    else:
                        v[0][m[0]] = 0.0
                    for j in range(1, len(Alice.records[snake_id])):
                        v[j][m[j]] = max(Alice.values[snake_id][j - 1])
                    X += Alice.records[snake_id]
                    V += Alice.values[snake_id]
                    X += self.mirror_states(Alice.records[snake_id])
                    V += self.mirror_values(Alice.values[snake_id])
                Alice.clear()
            print("Self play time", time() - t0)
            t0 = time()
            new_nnet.train(array(X), array(V), ep=32, bs=len(X)//8)
            print("Training time", time() - t0)
            t0 = time()
            # compare new net with previous net
            frac_win = self.compete(new_nnet, nnet)
            if frac_win > self.threshold:
                # replace with new net
                nnet = new_nnet
                nnet.save("nn" + str(iter))
                print("Iteration", iter, "beats the previouse version with a WR of", frac_win, "\nIt is now the new champion!")
            else:
                print("Iteration", iter, "failed to beat the previouse one. WR =", frac_win)
            print("Competing time", time() - t0, "\n")
    
    def mirror_states(self, states):
        return flip(states, axis = 2)
        
    def mirror_values(self, moves):
        return flip(states, axis = 1)
    
    def compete(self, nnet1, nnet2):
        sep = 1
        Alice = Agent(nnet1, range(sep))
        Bob = Agent(nnet2, range(sep, self.snake_cnt))
        win = 0
        loss = 0
        for _ in range(self.competeEps):
            g = Game(self.height, self.width, self.snake_cnt)
            winner_id = g.run(Alice, Bob, sep=sep)
            if winner_id is None:
                win += 1
                loss += 1
            elif winner_id < sep:
                win += 1
            else:
                loss += 1
        return win/(win + loss)