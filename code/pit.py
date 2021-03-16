from utils.agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.pit_mp_game_runner import MPGameRunner

from time import time, sleep

pit_games = 200
threshold = 0.52
height = 11
width = 11
snake_cnt = 4

model_name = input("Enter the model name (not including the generation number nor \".h5\"):\n")
iteration = int(input("Enter the starting generation (the first champion):\n"))
nnet = AlphaNNet(model_name = "models/" + model_name + str(iteration) + ".h5")
f = open("pit.txt", 'a')
f.write(model_name + str(iteration) + " is set to be the baseline champion.")
f.close()
Alice = Agent(nnet)
Alice_snake_cnt = snake_cnt//2
iteration += 1
new_challenger = False
while True:
    try:
        nnet = AlphaNNet(model_name = "models/" + model_name + str(iteration) + ".h5")
        new_challenger = True
        print("A new challenger,", model_name + str(iteration))
        Bob = Agent(nnet)
        # compare new net with previous net
        win = 0.0
        loss = 0.0
        t0 = time()
        gr = MPGameRunner(height, width, snake_cnt, 1, pit_games)
        print("Running games...")
        winner_ids = gr.run(Alice, Bob, Alice_snake_cnt)
        for winner_id in winner_ids:
            if winner_id is None:
                win += 0.5
                loss += 0.5
            elif winner_id < Alice_snake_cnt:
                loss += 1.0
            else:
                win += 1.0
        score = win/(win + loss)
        if score > threshold:
            Alice = Bob
            f = open("pit.txt", 'a')
            f.write(model_name + str(iteration) + " beats the previouse champion. score = "
                    + str(score) + ". It is the new champion!\n")
            f.close()
        else:
            f = open("pit.txt", 'a')
            f.write(model_name + str(iteration) + " failed to beat the previouse champion. score = "
                    + str(score) + ".\n")
            f.close()
        print("Competing time", time() - t0)
        iteration += 1
    except OSError:
        if new_challenger:
            print("Waiting for", model_name + str(iteration) + "...")
            new_challenger = False
        sleep(10)
