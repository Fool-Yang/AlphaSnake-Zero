from time import time
from utils.alphaNNet import AlphaNNet
from utils.agent import Agent
from utils.game import Game

height = 11
width = 11
snake_cnt = 4
competeEps = 128

file_name1 = input("\nEnter the model 1 name:\n")
file_name2 = input("\nEnter the model 2 name:\n")
nnet1 = AlphaNNet(model = "models/" + file_name1 + ".h5")
nnet2 = AlphaNNet(model = "models/" + file_name2 + ".h5")
sep = 1
Alice = Agent(nnet1, range(sep))
Bob = Agent(nnet2, range(sep, snake_cnt))
win = 0
draw = 0
t0 = time()
for _ in range(competeEps):
    g = Game(height, width, snake_cnt)
    winner_id = g.run(Alice, Bob, sep=sep)
    if winner_id is None:
        draw += 1
    elif winner_id < sep:
        win += 1
print("1v3 WR =", file_name1, win/(competeEps), "DR = ", draw/(competeEps))
print("Competing time", time() - t0)
sep = 1
Alice = Agent(nnet2, range(sep))
Bob = Agent(nnet1, range(sep, snake_cnt))
win = 0
draw = 0
t0 = time()
for _ in range(competeEps):
    g = Game(height, width, snake_cnt)
    winner_id = g.run(Alice, Bob, sep=sep)
    if winner_id is None:
        draw += 1
    elif winner_id < sep:
        win += 1
print("\n1v3 WR =", file_name2, win/(competeEps), "DR = ", draw/(competeEps))
print("Competing time", time() - t0)
snake_cnt = 2
sep = snake_cnt//2
Alice = Agent(nnet1, range(sep))
Bob = Agent(nnet2, range(sep, snake_cnt))
win = 0
loss = 0
t0 = time()
for _ in range(competeEps):
    g = Game(height, width, snake_cnt)
    winner_id = g.run(Alice, Bob, sep=sep)
    if winner_id is None:
        pass
    elif winner_id < sep:
        win += 1
    else:
        loss += 1
print("\n1v1 WR =", file_name1, win/(competeEps))
print("1v1 WR =", file_name2, loss/(competeEps))
print("Competing time", time() - t0)