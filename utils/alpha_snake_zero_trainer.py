from numpy import array, flip
from time import time

from utils.agent import Agent
from utils.game import Game
from utils.alphaNNet import AlphaNNet


class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                numEps=1024,
                competeEps=1024,
                threshold=0.275,
                height=11,
                width=11,
                snake_cnt=4
                iter=0):
        
        self.numEps = numEps
        self.competeEps = competeEps
        self.threshold = threshold
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
        self.iter = iter
    
    def train(self, nnet):
        # for training, all agents uses the same nnet
        # unless we want to use a evolution algorithm
        while True:
            self.iter += 1
            Alice = Agent(nnet, range(self.snake_cnt), mode=0, greedy=10 + iter)
            X = []
            V = []
            t0 = time()
            # the loop below can use distributed computing
            for ep in range(self.numEps):
                # collect examples from a new game
                g = Game(self.height, self.width, self.snake_cnt)
                winner_id = g.run(Alice)
                for snake_id in Alice.records:
                    x = Alice.records[snake_id]                                                
                    v = Alice.values[snake_id]
                    m = Alice.moves[snake_id]
                    p = Alice.odds[snake_id]
                    # assign estimated values
                    last_max = max(v[0])
                    if snake_id == winner_id:
                        v[0][m[0]] += (1.0 - v[0][m[0]])*p[0]
                    else:
                        v[0][m[0]] = 0.0
                    for j in range(1, len(x)):
                        delta = max(v[j - 1]) - last_max
                        if delta == 0.0:
                            break
                        last_max = max(v[j])
                        v[j][m[j]] += delta*p[j]
                    # sampling
                    sample_length = 8 if len(x) >= 8 else len(x)
                    sample_x = x[:sample_length]
                    sample_v = v[:sample_length]
                    i = sample_length
                    # can result in an infinite loop if sample_length is too small
                    while i < len(x):
                        sample_x.append(x[i])
                        sample_v.append(v[i])
                        i = round(1.2*i)
                    X += sample_x
                    V += sample_v
                    X += self.mirror_states(sample_x)
                    V += self.mirror_values(sample_v)
                Alice.clear()
            if len(X) > 100000:
                self.numEps //= 2
            print("Self play time", time() - t0)
            t0 = time()
            new_nnet = nnet.copy(lr=0.001*(0.96**self.iter))
            new_nnet.train(array(X), array(V), ep=32, bs=32000)
            print("Training time", time() - t0)
            t0 = time()
            # compare new net with previous net
            frac_win = self.compete(new_nnet, nnet)
            if frac_win > self.threshold:
                # replace with new net
                nnet = new_nnet
                nnet.save("nn" + str(self.iter))
                print("Iteration", self.iter, "beats the previouse version. WR =", frac_win, "\nIt is now the new champion!")
            else:
                print("Iteration", self.iter, "failed to beat the previouse one. WR =", frac_win)
            print("Competing time", time() - t0, "\n")
    
    def mirror_states(self, states):
        # flip return a numpy.ndarray
        # need to return a list
        # otherwise X += does vector addition
        return list(flip(states, axis = 2))
        
    def mirror_values(self, values):
        return list(flip(values, axis = 1))
    
    def compete(self, nnet1, nnet2):
        sep = 1
        Alice = Agent(nnet1, range(sep))
        Bob = Agent(nnet2, range(sep, self.snake_cnt))
        win = 0.0
        for _ in range(self.competeEps):
            g = Game(self.height, self.width, self.snake_cnt)
            winner_id = g.run(Alice, Bob, sep=sep)
            if winner_id is None:
                win += 0.75
            elif winner_id < sep:
                win += 1.0
        return win/self.competeEps