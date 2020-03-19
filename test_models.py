from time import time
from utils.alphaNNet import AlphaNNet
from utils.agent import Agent
from utils.test_game import Game
from player import Player

height = 11
width = 11
snake_cnt = 4
competeEps = 3

file_name1 = input("\nEnter the model 1 name:\n")
file_name2 = input("\nEnter the model 2 name:\n")
t0 = time()
nnet1 = AlphaNNet(model = "models/" + file_name1 + ".h5")
nnet2 = AlphaNNet(model = "models/" + file_name2 + ".h5")

f = open("replay.txt", 'w')
f.write('')
f.close()

sep = snake_cnt//2
Alice = Agent(nnet1, range(sep))
Bob = Agent(nnet2, range(sep, snake_cnt))
win = 0
loss = 0
for _ in range(competeEps):
    g = Game(height, width, snake_cnt)
    winner_id = g.run(Alice, Bob, sep=sep)
    if winner_id is None:
        win += 1
        loss += 1
    elif winner_id < sep:
        win += 1
    else:
        loss += 1
t1 = time()
n = input("\nHit Enter to watch replay")
Player().main()
print("WR of", file_name1, win/(win + loss))
print("WR of", file_name2, loss/(win + loss))
print("Competing time", t1 - t0)