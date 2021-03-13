from utils.agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.mp_game_runner import MPGameRunner

from player import Player

height = 11
width = 11
snake_cnt = 4

file_name = input("\nEnter the model name:\n")
net = AlphaNNet(model_name = "models/" + file_name + ".h5")
net.v_net.summary()
Alice = Agent(net)

f = open("replay.rep", 'w')
f.close()

n = input("\nHit Enter to run the game")
print("\nRunning games...")
gr = MPGameRunner(height, width, snake_cnt)
gr.run(Alice)
n = input("\nHit Enter to watch the replay")
Player().main()
