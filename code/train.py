import tensorflow as tf

from utils.alpha_nnet import AlphaNNet
from utils.alpha_snake_zero_trainer import AlphaSnakeZeroTrainer

game_board_height = 11
game_board_width = 11
number_of_snakes = 4
self_play_games = 256

try:
    # when running on Google Cloud
    # the TPU must be located in the same area as the CPU
    # pass the TPU's name
    tpu_name = input("Enter the name of the Google Cloud TPU (Leave empty if not using a TPU):\n")
    Resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = tpu_name)
    tf.config.experimental_connect_to_cluster(Resolver)
    tf.tpu.experimental.initialize_tpu_system(Resolver)
    TPU = tf.distribute.experimental.TPUStrategy(Resolver)
    print("Google Cloud TPU online.")
except:
    print("Cannot find the Google Cloud TPU. Using the CPU.")
    TPU = None

name = input("Enter the model name (not including the generation number nor \".h5\"):\n")
start = int(input("Enter the starting generation (0 for creating a new model):\n"))
if start == 0:
    ANNet = AlphaNNet(input_shape = (game_board_height*2 - 1, game_board_width*2 - 1, 3))
    ANNet.save(name + "0")
else:
    ANNet = AlphaNNet(model_name = "models/" + name + str(start) + ".h5")
Trainer = AlphaSnakeZeroTrainer(game_board_height, game_board_width, number_of_snakes, self_play_games, TPU)
Trainer.train(ANNet, name = name, iteration = start)
