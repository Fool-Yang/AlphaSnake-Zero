from utils.pit_agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.pit_mp_game_runner import MPGameRunner

from time import time

height = 11
width = 11
snake_cnt = 4
health_dec = 1
competeEps = 300

file_name1 = input("\nEnter the model 1 name:\n")
file_name2 = input("\nEnter the model 2 name:\n")
nnet1 = AlphaNNet(model_name = "models/" + file_name1 + ".h5")
nnet2 = AlphaNNet(model_name = "models/" + file_name2 + ".h5")
Alice = Agent(nnet1)
Bob = Agent(nnet2)

print("\nRunning games...")
win = 0
draw = 0
t0 = time()
gr = MPGameRunner(height, width, snake_cnt, health_dec, competeEps)
winner_ids = gr.run(Alice, Bob, 1)
for winner_id in winner_ids:
    if winner_id is None:
        draw += 1
    elif winner_id < 1:
        win += 1
print("1v3 Win Rate of", file_name1, win/(competeEps), "Draw Rate =", draw/(competeEps))
print("Competing time", time() - t0)

print("\nRunning games...")
win = 0
draw = 0
t0 = time()
gr = MPGameRunner(height, width, snake_cnt, health_dec, competeEps)
winner_ids = gr.run(Bob, Alice, 1)
for winner_id in winner_ids:
    if winner_id is None:
        draw += 1
    elif winner_id < 1:
        win += 1
print("1v3 Win Rate of", file_name2, win/(competeEps), "Draw Rate =", draw/(competeEps))
print("Competing time", time() - t0)

print("\nRunning games...")
snake_cnt = 2
win = 0
loss = 0
draw = 0
t0 = time()
gr = MPGameRunner(height, width, snake_cnt, health_dec, competeEps)
winner_ids = gr.run(Alice, Bob, snake_cnt//2)
for winner_id in winner_ids:
    if winner_id is None:
        draw += 1
    elif winner_id < 1:
        win += 1
    else:
        loss += 1
print("2v2 Win Rate of", file_name1, win/(competeEps))
print("2v2 Win Rate of", file_name2, loss/(competeEps))
print("Competing time", time() - t0)
