<h2 align="center"> Robo-Chess </h2>
<p align="center">
<img src="https://github.com/hishamcse/Robo-Chess/blob/main/images/chess.png" width="40%"/>
</p>

Robo-Chess, a comprehensive repository dedicated to developing chess engines using a variety of AI techniques. This repository includes multiple projects, each focusing on different aspects of chess AI, from self-play imitation learning to endgame solving with deep Q-networks. It also adds a solution to a kaggle chess competition.

# Detailed Overview

## Chess-Engine-Self-Play-Imitation
This project implements a chess engine using self-play imitation learning.

**Adapted & Extended From:**
   - [FreeCodeCamp Blog](https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/)
   - [Github Codebase](https://github.com/EivindKjosbakken/ChessEngine)
   - [gym-chess official](https://github.com/iamlucaswolf/gym-chess)
   - [How to play chess](https://www.chess.com/learn-how-to-play-chess)
   - [python-chess](https://python-chess.readthedocs.io/en/latest/)
   - [CrazyAra](https://github.com/QueensGambit/CrazyAra/wiki/Model-architecture)

### Directory Structure
- **data/**: Contains training and validation datasets.
  - **Datasets Used**:
      - [Kaggle Chess Evaluations Dataset](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data)
      - [Lichess Dataset](https://database.lichess.org/#puzzles)
      - [Stockfish](https://stockfishchess.org/download/)
- **savedModels/**: Directory for saving trained models.
- **Robo_Chess-Chess-AI-Self-Play-Imitation.ipynb**: Main notebook for training and evaluating the chess engine using self-play imitation learning.

### Model Architectures
- **Simple Model**:
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/blob/main/images/simple-model.png" width="60%" title="Simple Model"/></p>
- **Complex Model**:
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/blob/main/images/complex-model.png" /></p>

## kaggle-chess-competition
Kaggle competition solutions for training an AI to play chess.

**Adapted & Extended From:**
 - [Kaggle Competition](https://www.kaggle.com/competitions/train-an-ai-to-play-chess)
 - [Kaggle Codebase](https://www.kaggle.com/competitions/train-an-ai-to-play-chess/code)
 - [Youtube Videos](https://www.youtube.com/live/l0bv8IgELfU?si=ijpiOcrPoyq-yrhW)
 - [How to play chess](https://www.chess.com/learn-how-to-play-chess)
 - [python-chess](https://python-chess.readthedocs.io/en/latest/)

### Directory Structure
- **data-train-an-ai-to-play-chess/**: Contains dataset used for the Kaggle competition.
    - **Datasets Used**:
        - [Kaggle Train an AI to Play Chess Dataset](https://www.kaggle.com/competitions/train-an-ai-to-play-chess/data)
- **Model Improvement Suggestions.pdf**: Document with suggestions for model improvements from ChatGPT-4o
- **robo-chess-chess-ai-global-hack-competition.ipynb**: Notebook containing the solution for the Global Hack competition.
  
## Chess-Endgame
This project focuses on endgame solving using DQN (Deep Q-Network) and DDQN (Double DQN) models. This is not tested thoroughly as tianshou updated version still under experimental phase

**Directory Structure**
- **Chess_env_gym.py**: Environment setup for training DQN models.
- **DDQN-tianshou.py**: Implementation of Double DQN using the Tianshou library.
- **DQN-tianshou.py**: Implementation of DQN using the Tianshou library.
- **degree_freedom_king1.py**: Degree of freedom calculations for King piece.
- **degree_freedom_king2.py**: Additional calculations for King piece.
- **degree_freedom_queen.py**: Degree of freedom calculations for Queen piece.
- **generate_game.py**: Script to generate chess game data for training.

## Notebooks on Kaggle
- [Robo-Chess: Chess AI Self Play Imitation](https://www.kaggle.com/code/syedjarullahhisham/robo-chess-chess-ai-self-play-imitation)
- [Robo-Chess: Chess AI Global Hack Competition](https://www.kaggle.com/code/syedjarullahhisham/robo-chess-chess-ai-global-hack-competition)

## Limitations
The **Chess-Engine-Self-Play-Imitation** was very promising but it requires excessive computational power. For e.g: with the complex architecture, even 1 epoch train takes several minutes with the 40% of the data (approximatetly 120K board positions & moves generated from lichess highest rated plays). So, I even couldtn't use whole 465K datas for training. So, making a good model is very challenging. My model at max could give a fight with 600 ELO Stockfish Agent which my trained agent ultimately lost.

