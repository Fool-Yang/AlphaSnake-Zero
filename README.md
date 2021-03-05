## Presentation
<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/12Hr78S1oMO34BEdedrI0sOAhaIAdE3jO/preview" width="640" height="480" frameborder="0" allowfullscreen="true"></iframe>
</figure>

**For detailed descriptions of the algorithms, please check the [report](https://github.com/Fool-Yang/AlphaSnake-Zero/blob/master/report.pdf).**

## Abstract
A lot of studies have been done on reinforcement learning recently. Q learning, or DQN tries to solve the single agent vs environment problem, where some other approaches such as AlphaGo attempt the double agent game. In this project, we try to find an algorithm to generate an agent that performs well in a multiagent synchronous strategy game. Although this project is specific to the game called Battlesnake, the method and algorithms we used is not limited to it. The math holds for any synchronous game that has a finite state and action space.

![demo](./demo.gif)

## Requirments:
Python 3.7.6

NumPy 1.18.1

TensorFlow 2.1.0

## Instructions:
Go to the "code" folder and run train.py to start training models. It will ask you to enter things. If you enter a starting iteration number greater than 0, it will try to start using the existing model (e.g. if you enter the model name "MySnake", and generation number "7", it will try to open the file "MySnake7.h5" in the "models" folder and start training). Otherwise, it will create a new model and start training. All models will be stored in the "models" folder as .h5 files. "log.csv" will store historical stats of the models.

If you want to change the neural network structure or the optimizer, modify it in "alpha_nnet.py". You should not modify anything other than "alpha_nnet.py" and "alpha_snake_zero_trainer.py".

Run pit.py to see competition results between different generations. If a generation beats the last champion it will become the new champion. Since now the algorithm is shifted to the AlphaZero's algorithm, the pit phase is no longer part of the training. You don't have to run pit.py. It only gives you information about the training results. The list of champions will be stored in "champions.csv".

Run test_models.py to watch some games played by the model.

Run test_pit.py to run a large number of games between 2 models and observe the stats.
