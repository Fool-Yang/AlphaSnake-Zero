from numpy import array, flip
from time import time

from utils.agent import Agent
from utils.game import Game
from utils.alphaNNet import AlphaNNet


class AlphaSnakeZeroTrainer:
    
    def __init__(self,
                numEps=1024,
                competeEps=1024,
                threshold=0.55,
                height=11,
                width=11,
                snake_cnt=4):
        
        self.numEps = numEps
        self.competeEps = competeEps
        self.threshold = threshold
        self.height = height
        self.width = width
        self.snake_cnt = snake_cnt
    
    def train(self, nnet, name="nn", itr = 0):
        # log
        if itr == 0:
            f = open("results.txt", 'w')
            f.write('wall_collision, body_collision, head_collision, starvation, food_eaten, game_length\n')
            f.close()
        health_dec = 9
        while True:
            itr += 1
            if itr > 64:
                health_dec = 1
            elif itr > 32:
                health_dec = 3
            # log
            wall_collision = 0
            body_collision = 0
            head_collision = 0
            starvation = 0
            food_eaten = 0
            game_length = 0
            # for training, all agents uses the same nnet
            Alice = Agent(nnet, range(self.snake_cnt), training=True, greedy=10 + itr)
            X = []
            V = []
            t0 = time()
            # the loop below can use distributed computing
            for ep in range(self.numEps):
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
                    last_max = max(v[0])
                    if snake_id == winner_id:
                        v[0][m[0]] += (1.0 - v[0][m[0]])*p[0]
                    else:
                        v[0][m[0]] = -1.0
                    for j in range(1, len(x)):
                        delta = max(v[j - 1]) - last_max
                        if delta == 0.0:
                            break
                        last_max = max(v[j])
                        v[j][m[j]] += delta*p[j]
                        # once the network is somewhat good this should never happen
                        if v[j][m[j]] < -1.0:
                            v[j][m[j]] = -1.0
                        elif v[j][m[j]] > 1.0:
                            v[j][m[j]] = 1.0
                    # sampling
                    sample_length = 8 if len(x) >= 8 else len(x)
                    sample_x = x[:sample_length]
                    sample_v = v[:sample_length]
                    i = sample_length
                    # can result in an infinite loop if sample_length is too small
                    while i < len(x):
                        sample_x.append(x[i])
                        sample_v.append(v[i])
                        i = round(1.1*i)
                    X += sample_x
                    V += sample_v
                    X += self.mirror_states(sample_x)
                    V += self.mirror_values(sample_v)
                Alice.clear()
            if len(X) > 100000:
                self.numEps //= 2
            print("Self play time", time() - t0)
            t0 = time()
            new_nnet = nnet.copy(lr=0.0001*(0.97**itr))
            new_nnet.train(array(X), array(V), ep=32, bs=4096)
            print("Training time", time() - t0)
            t0 = time()
            # compare new net with previous net
            score = self.compete(new_nnet, nnet)
            if score > self.threshold:
                # replace with new net
                nnet = new_nnet
                nnet.save(name + str(itr))
                print("Iteration", itr, "beats the previouse version. score =", score, "\nIt is now the new champion!")
                log_array = [itr, wall_collision, body_collision, head_collision, starvation, food_eaten, game_length]
                log = ', '.join(map(str, log_array)) + '\n'
                f = open("results.txt", 'a')
                f.write(log)
                f.close()
            else:
                print("Iteration", itr, "failed to beat the previouse one. score =", score)
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
        Alice = Agent(nnet1)
        Bob = Agent(nnet2)
        win = 0.0
        loss = 0.0
        for _ in range(self.competeEps):
            g = Game(self.height, self.width, self.snake_cnt)
            winner_id = g.run(Alice, Bob, sep=sep)
            if winner_id is None:
                win += 1.5
                loss += 1.0
            elif winner_id < sep:
                win += 3.0
            else:
                loss += 1.0
        return win/(win + loss)
