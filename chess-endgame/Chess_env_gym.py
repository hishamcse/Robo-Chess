import numpy as np
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
import gym
from gym import spaces

class Chess_Env_Gym(gym.Env):

    def __init__(self, N_grid):

        self.N_grid = N_grid  # SIZE OF THE BOARD

        self.Board = np.zeros([N_grid,
                               N_grid]) # THE BOARD, THIS WILL BE FILLED BY 0 (NO PIECE), 1 (AGENT'S KING), 2 (AGENT'S QUEEN), 3 (OPPONENT'S KING)

        self.p_k1 = np.zeros([2, 1])    # POSITION OF THE AGENT'S KING AS COORDINATES
        self.p_k2 = np.zeros([2, 1])    # POSITION OF THE OPPOENT'S KING AS COORDINATES
        self.p_q1 = np.zeros([2, 1])    # POSITION OF THE AGENT'S QUEEN AS COORDINATES

        self.dfk1 = np.zeros([N_grid,
                              N_grid])  # ALL POSSIBLE ACTIONS FOR THE AGENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfk2 = np.zeros([N_grid,
                              N_grid])  # ALL POSSIBLE ACTIONS FOR THE OPPONENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfq1 = np.zeros([N_grid,
                              N_grid])  # ALL POSSIBLE ACTIONS FOR THE AGENT'S QUEEN (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)

        self.dfk1_constrain = np.zeros(
            [N_grid, N_grid])           # ALLOWED ACTIONS FOR THE AGENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfk2_constrain = np.zeros(
            [N_grid, N_grid])           # ALLOWED ACTIONS FOT THE OPPONENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfq1_constrain = np.zeros(
            [N_grid, N_grid])           # ALLOWED ACTIONS FOT THE AGENT'S QUEEN CONSIDERING ALSO THE OTHER PIECES

        self.ak1 = np.zeros([8])  # ALLOWED ACTIONS OF THE AGENT'S KING (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.possible_king_a = np.shape(self.ak1)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S KING

        self.aq1 = np.zeros(
            [8 * (self.N_grid - 1)])  # ALLOWED ACTIONS OF THE AGENT'S QUEEN (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.possible_queen_a = np.shape(self.aq1)[0]  # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S QUEEN

        self.check = 0  # 1 (0) IF ENEMY KING (NOT) IN CHECK

        # THIS MAP IS USEFUL FOR US TO UNDERSTAND THE DIRECTION OF MOVEMENT GIVEN THE ACTION MADE (SKIP...)
        self.map = np.array([[1, 0],
                             [-1, 0],
                             [0, 1],
                             [0, -1],
                             [1, 1],
                             [1, -1],
                             [-1, 1],
                             [-1, -1]])
        
        obs, _ = self.reset()
        self.state_shape = obs['obs'].shape[0]
        self.action_shape = obs['mask'].shape[0]
        self.action_space = spaces.Discrete(self.action_shape)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.state_shape,),
            dtype=int
        )

        self.cur_step = 0
        self.max_steps = 20

    def reset(self, seed=None, options=None):
        self.cur_step = 0

        # START THE GAME BY SETTING PIECIES
        self.Board, self.p_k2, self.p_k1, self.p_q1 = generate_game(self.N_grid)

        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board,
                                                                          self.p_k1)

        # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
        allowed_a = np.concatenate([self.a_q1, self.a_k1], 0)

        # FEATURES (INPUT TO NN) AT THIS POSITION
        X = self.Features()

        allowed_a = np.squeeze(allowed_a)
        info = {'board': self.Board}
        # print('\n')
        # print('init board:')
        # print(self.Board)
        # print('init mask')
        # print(allowed_a)
        return {'obs': X, 'mask': allowed_a}, info

    def step(self, a_agent):
        # print(f'action: {a_agent}')

        # SET REWARD TO ZERO IF GAME IS NOT ENDED
        R = 0
        # SET Done TO ZERO (GAME NOT ENDED)
        Done = 0
        self.cur_step += 1

        # PERFORM THE AGENT'S ACTION ON THE CHESS BOARD

        if a_agent < self.possible_queen_a:  # THE AGENT MOVED ITS QUEEN

            # UPDATE QUEEN'S POSITION
            direction = int(np.ceil((a_agent + 1) / (self.N_grid - 1))) - 1
            steps = a_agent - direction * (self.N_grid - 1) + 1

            self.Board[self.p_q1[0], self.p_q1[1]] = 0

            mov = self.map[direction, :] * steps
            self.Board[self.p_q1[0] + mov[0], self.p_q1[1] + mov[1]] = 2
            self.p_q1[0] = self.p_q1[0] + mov[0]
            self.p_q1[1] = self.p_q1[1] + mov[1]

        else:  # THE AGENT MOVED ITS KING

            # UPDATE KING'S POSITION
            direction = a_agent - self.possible_queen_a
            steps = 1

            self.Board[self.p_k1[0], self.p_k1[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k1[0] + mov[0], self.p_k1[1] + mov[1]] = 1
            self.p_k1[0] = self.p_k1[0] + mov[0]
            self.p_k1[1] = self.p_k1[1] + mov[1]

        # COMPUTE THE ALLOWED ACTIONS AFTER AGENT'S ACTION
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)

        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board,
                                                                          self.p_k1)

        # CHECK IF POSITION IS A CHECKMATE, DRAW, OR THE GAME CONTINUES

        # CASE OF CHECKMATE
        if np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 1:

            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            Done = 1  # The epsiode ends
            R = 1  # Reward for checkmate
            allowed_a = np.zeros(self.action_shape)  # Allowed_a set to nothing (end of the episode)
            X = np.zeros(self.state_shape)  # Features set to nothing (end of the episode)

        # CASE OF DRAW
        elif np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 0:

            # King 2 has no freedom but it is not checked
            Done = 1  # The epsiode ends
            R = 0.  # Reward for draw
            allowed_a = np.zeros(self.action_shape)  # Allowed_a set to nothing (end of the episode)
            X = np.zeros(self.state_shape)  # Features set to nothing (end of the episode)

        # THE GAME CONTINUES
        else:

            # THE OPPONENT MOVES THE KING IN A RANDOM SAFE LOCATION
            allowed_enemy_a = np.where(self.a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            self.Board[self.p_k2[0], self.p_k2[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k2[0] + mov[0], self.p_k2[1] + mov[1]] = 3

            self.p_k2[0] = self.p_k2[0] + mov[0]
            self.p_k2[1] = self.p_k2[1] + mov[1]

            # COMPUTE THE ALLOWED ACTIONS AFTER THE OPPONENT'S ACTION
            # Possible actions of the King
            self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1,
                                                                             self.Board)

            # Allowed actions for the agent's king
            self.dfq1_constrain, self.a_q1, self.dfq1 = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1,
                                                                             self.Board)

            # Allowed actions for the enemy's king
            self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1,
                                                                              self.Board, self.p_k1)

            # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
            allowed_a = np.concatenate([self.a_q1, self.a_k1], 0)
            # FEATURES
            X = self.Features()

        allowed_a = np.squeeze(allowed_a)
        terminated = Done
        truncated = self.cur_step >= self.max_steps
        info = {'board': self.Board}
        # print(self.Board)
        # print(f'done: {terminated or truncated}')
        # print('mask')
        # print(allowed_a)
        return {'obs': X, 'mask': allowed_a}, R, terminated, truncated, info

    # DEFINITION OF THE FEATURES (SEE ALSO ASSIGNMENT DESCRIPTION)
    def Features(self):
        s_k1 = np.array(self.Board == 1).astype(float).reshape(-1)  # FEATURES FOR KING POSITION
        s_q1 = np.array(self.Board == 2).astype(float).reshape(-1)  # FEATURES FOR QUEEN POSITION
        s_k2 = np.array(self.Board == 3).astype(float).reshape(-1)  # FEATURE FOR ENEMY'S KING POSITION

        check = np.zeros([2])  # CHECK? FEATURE
        check[self.check] = 1

        K2dof = np.zeros([8])  # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED
        K2dof[np.sum(self.dfk2_constrain).astype(int)] = 1

        # ALL FEATURES...
        x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof], 0)

        return x
