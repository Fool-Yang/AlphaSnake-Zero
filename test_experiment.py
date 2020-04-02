from utils.alphaNNet import AlphaNNet
from utils.agent import Agent
from utils.game import Game
from time import time

height = 11
width = 11
snake_cnt = 4
competeEps = 64

f = open("results.txt", 'w')
f.write('wall_collision, body_collision, head_collision, starvation, food_eaten, game_length\n')
f.close()

print("Running games...")
t0 = time()
for i in range(1, 155):
    file_name = "models/nn" + str(i) + ".h5"
    try:
        net = AlphaNNet(model = file_name)
    except OSError:
        continue
    agent = Agent(net)
    wall_collision = 0
    body_collision = 0
    head_collision = 0
    starvation = 0
    food_eaten = 0
    game_length = 0
    print("Running", file_name)
    t1 = time()
    for _ in range(competeEps):
        g = Game(height, width, snake_cnt)
        g.run(agent)
        wall_collision += g.wall_collision
        body_collision += g.body_collision
        head_collision += g.head_collision
        starvation += g.starvation
        food_eaten += g.food_eaten
        game_length += g.game_length
    wc = wall_collision/competeEps
    bc = body_collision/competeEps
    hc = head_collision/competeEps
    s = starvation/competeEps
    fe = food_eaten/competeEps
    gl = game_length/competeEps
    log = ', '.join(map(str, [wc, bc, hc, s, fe, gl])) + '\n'
    f = open("results.txt", 'a')
    f.write(log)
    f.close()
    print("Running time", time() - t1, end = '\n\n')
print("Total Running time", time() - t0)
