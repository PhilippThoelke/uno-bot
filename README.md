# UnoBot
A reinforcement learning based agent trained to play the card game Uno (https://en.wikipedia.org/wiki/Uno_(card_game)). The project was implemented using Python with various modules for efficient arrays, machine learning and GUIs. Q-learning agents are trained inside the game environment and the resulting model can be analyzed inside a graphical version of the game including AI and human players, as well as a naive baseline algorithm.
## Training
To train the Q-model, simply start the train.py file with `python train.py` (to continue training an existing model, run `python train.py path/to/model.h5`). The model architecture and hyperparameters can be adjusted in the *agent.py* file. Further parameters regarding the Q-learning algorithm can be tuned inside the *train.py* file. Periodical model checkpoints (frequency adjustable in *agent.py*) will be saved under *models/\<timestamp>/model-\<epoch>.h5* and a tensorboard-compatible log file will be stored inside a *logs/\<timestamp>* folder.
## Playing
To run the game with a GUI, use `python play.py <player1> <player2> ...` and replace player arguments with either "AI", "Human" or "Naive". The AI tag will use the model specified inside the *play.py* file, adjust the model path variable to use a different model. If the AI player plays an illegal move, it will immediately be eliminated from the game. Selecting "Human" will allow the user to decide which moves to play in the game and the naive player will always select the first legal move inside the action space. At least two players have to be specified to start a game but player types can be mixed freely.
## Game limitations
Players are not able to choose the colour of the next card after a 4+ or wild card. Instead, the next colour will be determined randomly. This was done to eliminate the need for the AI to choose a preferred colour and in turn keeping the game fair among different player types.
## Library requirements
- NumPy
- Keras
- tensorflow
- pygame
