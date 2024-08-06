<h2 align="center"> Robo-Chess </h2>
<p align="center">
<img src="https://github.com/hishamcse/Robo-Chess/blob/main/images/chess.png" width="40%"/>
</p>

**Robo-Chess**, a comprehensive repository dedicated to developing chess engines using a variety of **Deep Reinforcement Learning** techniques. This repository includes multiple projects, each focusing on different solutions of chess AI, from self-play imitation learning or endgame solving with deep Q-networks to Modified AlphaZero Algorithm implementation. It also adds solution to a kaggle chess competition.

# Table of Contents

- [Detailed Overview](#detailed-overview)
   - [Chess-Engine-Self-Play-Imitation](#chess-engine-self-play-imitation)
     - [Directory Structure](#directory-structure)
     - [Model Architectures](#model-architectures)
   - [kaggle-chess-competition](#kaggle-chess-competition)
     - [Directory Structure](#directory-structure-1)
   - [Chess-Endgame](#chess-endgame)
     - [Directory Structure](#directory-structure-2)
- [Notebooks on Kaggle](#notebooks-on-kaggle)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [My DRL Projects](#my-drl-projects)

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
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/tree/main/Chess-Engine-Self-Play-Imitation/images/simple-model.png" width="65%" title="Simple Model"/></p>
- **Complex Model**:
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/tree/main/Chess-Engine-Self-Play-Imitation/images/complex-model.png" /></p>

### Limitation
The **Chess-Engine-Self-Play-Imitation** approach showed great promise but demanded significant computational resources. For example, by using the complex architecture, training just one epoch with 40% of the data (approximately 120K board positions and moves from Lichess's highest-rated games) took several minutes. Consequently, I couldn't utilize the entire 465K dataset for training. This made developing a robust model quite challenging. The best model I trained could compete with a 600 ELO Stockfish agent, but it ultimately lost.

## AlphaZero-Inspired-Chess-Engine
This project implements a chess engine using modified AlphaZero algorithm.

**Adapted & Extended From:**
   - [Original AlphaZero](https://github.com/geochri/AlphaZero_Chess)
   - [AlphaZero With GUI](https://github.com/davnords/chess-engine)
   - [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
   - [python-chess](https://python-chess.readthedocs.io/en/latest/)

### Directory Structure
- datasets - store generated datasets by running MCTS_chess.py
- evaluator_data - store results by running evaluator.py
- model_data - store trained model and graph results by running train.py
- MCTS_chess.py - implements the Monte-Carlo Tree Search (MCTS) algorithm based on Polynomial Upper Confidence Trees (PUCT) method for leaf transversal. This generates datasets (state, policy, value) for neural network training
- alpha_net.py - PyTorch implementation of the modified AlphaGoZero neural network architecture 
- Rests are self-explanatory

### Model Architectures
- **Original Model**:
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/blob/main/AlphaZero-Modified-Chess-Engine/images/alphazero-model.png" width="65%" title="Simple Model"/></p>
- **Modified Model**:
  <p align="center"><img src="https://github.com/hishamcse/Robo-Chess/blob/main/AlphaZero-Modified-Chess-Engine/images/alphazero-complex-model.png" /></p>

### How to Run
- Run pipeline.py to start the MCTS search and neural net training process. Change the folder and net saved names accordingly. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading.

OR

- Run the MCTS_chess.py to generate self-play datasets. Note that for the first time, you will need to create and save a random, initialized alpha_net for loading. 

- Run train.py to train the modified alpha_net with the datasets.

- At predetermined checkpoints, run evaluator.py to evaluate the trained net against the neural net from previous iteration. Saves the neural net that performs better. Multiprocessing is enabled, which shares the PyTorch net model in a single CUDA GPU across 6 CPUs workers each running a MCTS self-play. 


## kaggle-chess-competition
Kaggle competition solutions for training an AI to play chess.

**Adapted & Extended From:**
 - [Kaggle Competition](https://www.kaggle.com/competitions/train-an-ai-to-play-chess)
 - [Kaggle Codebase](https://www.kaggle.com/competitions/train-an-ai-to-play-chess/code)
 - [Youtube Videos](https://www.youtube.com/live/l0bv8IgELfU?si=ijpiOcrPoyq-yrhW)
 - [How to play Chess](https://www.chess.com/learn-how-to-play-chess)
 - [python-chess](https://python-chess.readthedocs.io/en/latest/)

### Directory Structure
- **data-train-an-ai-to-play-chess/**: Contains dataset used for the Kaggle competition.
    - **Datasets Used**:
        - [Kaggle Train an AI to Play Chess Dataset](https://www.kaggle.com/competitions/train-an-ai-to-play-chess/data)
- **Model Improvement Suggestions.pdf**: Document with suggestions for model improvements from ChatGPT-4o
- **robo-chess-chess-ai-global-hack-competition.ipynb**: Notebook containing the solution for the Global Hack competition.
  
## Chess-Endgame
This project focuses on endgame solving using DQN (Deep Q-Network) and DDQN (Double DQN) models. This is not tested thoroughly as tianshou updated version still under experimental phase

**Adapted From:**
 - [Github Codebase](https://github.com/Nishantsgithub/Chess-AI-Development-Using-Reinforcement-Learning)
 - [Report Paper](https://github.com/Nishantsgithub/Chess-AI-Development-Using-Reinforcement-Learning/blob/main/Dissertation.pdf)

### Directory Structure
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

## Acknowledgements
- [Eivind Kjosbakken](https://www.freecodecamp.org/news/author/kjosbakken/)
- [William Lifferth](https://www.kaggle.com/wlifferth)
- [Nishant Panchal](https://github.com/Nishantsgithub)
- [FreeCodeCamp Blog](https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/)
- [Github Codebase](https://github.com/EivindKjosbakken/ChessEngine)
- [gym-chess official](https://github.com/iamlucaswolf/gym-chess)
- [How to play chess](https://www.chess.com/learn-how-to-play-chess)
- [python-chess](https://python-chess.readthedocs.io/en/latest/)
- [CrazyAra](https://github.com/QueensGambit/CrazyAra/wiki/Model-architecture)
- [Kaggle Chess Evaluations Dataset](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data)
- [Lichess Dataset](https://database.lichess.org/#puzzles)
- [Stockfish](https://stockfishchess.org/download/)
- [Kaggle Competition](https://www.kaggle.com/competitions/train-an-ai-to-play-chess)
- [Kaggle Codebase](https://www.kaggle.com/competitions/train-an-ai-to-play-chess/code)
- [Youtube Videos](https://www.youtube.com/live/l0bv8IgELfU?si=ijpiOcrPoyq-yrhW)
- [How to play Chess](https://www.chess.com/learn-how-to-play-chess)
- [python-chess](https://python-chess.readthedocs.io/en/latest/)
- [Github Codebase](https://github.com/Nishantsgithub/Chess-AI-Development-Using-Reinforcement-Learning)
- [Report Paper](https://github.com/Nishantsgithub/Chess-AI-Development-Using-Reinforcement-Learning/blob/main/Dissertation.pdf)
- [GraphViz](https://dreampuf.github.io/GraphvizOnline)

## My DRL Projects
You can view my **Deep Reinforcement Learning** projects
- [DRL-Renegades-Game-Bots](https://github.com/hishamcse/DRL-Renegades-Game-Bots)
- [Advanced-DRL-Renegades-Game-Bots](https://github.com/hishamcse/Advanced-DRL-Renegades-Game-Bots)
- [Robo-Chess](https://github.com/hishamcse/Robo-Chess)
- [Kaggle Notebooks on DRL](https://www.kaggle.com/syedjarullahhisham/code?tags=13313)
