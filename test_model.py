from utils.alphaNNet import AlphaNNet
from utils.agent import Agent
from utils.test_game import Game
from player import Player

file_name = input("\nEnter the model file name:\n")
net = AlphaNNet(model = "models/" + file_name + ".h5")
snake_cnt = 4 #int(input("Enter the number of snakes:\n"))

f = open("replay.txt", 'w')
f.write('')
f.close()

for _ in range(1):
    g = Game(11, 11, snake_cnt)
    g.run(Agent(net, list(range(snake_cnt))))
net.v_net.summary()
n = input("\nHit Enter to watch replay")
Player().main()
net.v_net.summary()