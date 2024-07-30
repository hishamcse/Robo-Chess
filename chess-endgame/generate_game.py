import numpy as np
from degree_freedom_king1 import *
from degree_freedom_queen import *


def generate_game(size_board):
    """
    This function creates a new chess game with three pieces at random locations. The enemy King has no interaction
    with our King and Queen. Positions range from 0 to 4
    :param size_board:
    :return:
    """
    s = np.zeros([size_board, size_board], dtype=int)
    c = np.linspace(1, size_board * size_board, num=size_board * size_board)

    k1 = 1  # King
    q1 = 2  # Queen
    k2 = 3  # Enemy King

    while(1):
        # Spawn Queen at a random location of the board
        p_q1 = [int(np.ceil(np.random.rand() * (size_board - 1))), int(np.ceil(np.random.rand() * (size_board - 1)))]
        s[p_q1[0], p_q1[1]] = q1

        # Spawn King at a random location which is occupied
        while(1):
            p_k1 = [int(np.ceil(np.random.rand() * (size_board - 1))), int(np.ceil(np.random.rand() * (size_board - 1)))]
            if p_k1 != p_q1:
                break
        s[p_k1[0], p_k1[1]] = k1

        # King's location
        dfK1, _, _ = degree_freedom_king1(p_k1, [np.inf, np.inf], p_q1, s)
        dfK1[p_k1[0], p_k1[1]] = 1
        
        # Queen's location
        dfQ1, _, _ = degree_freedom_queen(p_k1, [np.inf, np.inf], p_q1, s)
        dfQ1[p_q1[0], p_q1[1]] = 1

        # Empty locations
        c1 = np.where(dfK1.reshape([-1]) == 0)[0]
        c2 = np.where(dfQ1.reshape([-1]) == 0)[0]
        c = np.intersect1d(c1, c2)

        if c.shape[0] != 0:
            break

    i = int(np.ceil(np.random.rand() * len(c)) - 1)
    s.reshape([-1])[c[i]] = k2
    s = s.reshape([size_board, size_board])
    p_k2 = np.concatenate(np.where(s == k2))

    return np.array(s), np.array(p_k2), np.array(p_k1), np.array(p_q1)