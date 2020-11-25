# AlphaSnake-Zero
A machine learning AI that learns to play the game BattleSnake

For detailed descriptions, please check the report.pdf.

This is an undergrad course project at the University of Victoria. I (Yang Li) contributed most of the work to it.

Link to the presentation: https://drive.google.com/drive/folders/1Knb5xECKhTKK9vVSAutyHNOFClFaFf8o

## Demo
<img src="/demo.gif" width="250" height="250"/>

## Requirments:
Python 3.7.6

NumPy 1.18.1

TensorFlow 2.1

## Instructions:
Run train.py to start training models. If you enter an existing model name and a generation number > 0, it will try to start using the existing model (e.g. if you enter the model family name "MySnake", and generation number "7", it will try to open the file "MySnake7.h5" in the "models" folder and start training). Otherwise it will create a new model and start training. All models will be stored in the "models" folder as .h5 files.

Run test_models.py to watch some games played by the model.

Run test_pit.py to run a large number of games between 2 models and observe the stats.

Run test_weights.py to see details of the artificial neural network.
