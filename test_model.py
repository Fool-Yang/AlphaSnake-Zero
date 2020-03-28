from utils.alphaNNet import AlphaNNet
from utils.agent import Agent
from utils.test_game import Game
from player import Player

height = 11
width = 11
snake_cnt = 4
competeEps = 3

file_name = input("\nEnter the model name:\n")
net = AlphaNNet(model = "models/" + file_name + ".h5")
net.v_net.summary()
agent = Agent(net, list(range(snake_cnt)))

f = open("replay.txt", 'w')
f.write('')
f.close()

print("Running games...")
for _ in range(competeEps):
    g = Game(height, width, snake_cnt)
    g.run(agent)
n = input("\nHit Enter to watch replay")
Player().main()