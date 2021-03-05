from utils.agent import Agent
from utils.alpha_nnet import AlphaNNet
from utils.mp_game_runner import MPGameRunner

from time import time, sleep

pit_games = 400
threshold = 0.53
number_of_new_nets = 2
height = 11
width = 11
snake_cnt = 4

model_name = input("Enter the model name (not including the generation number nor \".h5\"):\n")
iteration = int(input("Enter the starting iteration:\n"))
nnet = AlphaNNet(model_name = "models/" + model_name + str(iteration) + ".h5")
Alice = Agent(nnet)
Alice_snake_cnt = snake_cnt//2
f = open("champions.csv", 'w')
f.write("Model name, Score against the previous champion\n")
f.write(model_name + str(iteration) + ", N/A\n")
f.close()
iteration += 1
while True:
    try:
        nnet = AlphaNNet(model_name = "models/" + model_name + str(iteration) + ".h5")
        Bob = Agent(nnet)
        # compare new net with previous net
        win = 0.0
        loss = 0.0
        t0 = time()
        gr = MPGameRunner(height, width, snake_cnt, 1, pit_games)
        winner_ids = gr.run(Alice, Bob, Alice_snake_cnt, True)
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
            print("Iteration", iteration, "beats the previouse version. score =", score, "\nIt is now the new champion!")
            f = open("champions.csv", 'a')
            f.write(model_name + str(iteration) + ", " + str(score) + '\n')
            f.close()
        else:
            print("Iteration", iteration, "failed to beat the previouse one. score =", score)
        print("Competing time", time() - t0, "\n")
        iteration += 1
    except OSError:
        sleep(10)
