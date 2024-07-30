import numpy as np


def degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1):
    """
    Returns a matrix of ones and zeros where one specify a location in which the oppement king can move to
    :param dfK1: Degrees of freedom for King 1
    :param p_k2: Position of King 2
    :param dfQ1: Degrees of freedom for the Queen
    :param s: board
    :param p_k1: Position of King1
    :return: 
        dfK2: Degrees of freedom for King 2 
        a_k2: allowed actions for King 2 
        check: 1 if it is checked, -1 if not
    """

    size_board = s.shape[0]

    dfK2 = np.zeros([size_board, size_board], dtype=int)
    dfK2[p_k2[0], p_k2[1]] = 1

    # King 2 reach
    k2r = [[p_k2[0] - 1, p_k2[1]],      # up
           [p_k2[0] + 1, p_k2[1]],      # down
           [p_k2[0], p_k2[1] + 1],      # right
           [p_k2[0], p_k2[1] - 1],      # left
           [p_k2[0] + 1, p_k2[1] + 1],  # down-right
           [p_k2[0] + 1, p_k2[1] - 1],  # down-left
           [p_k2[0] - 1, p_k2[1] + 1],  # up-right
           [p_k2[0] - 1, p_k2[1] - 1]]  # up-left
    k2r = np.array(k2r)

    a_k2 = np.zeros([8, 1], dtype=int)

    for i in range(k2r.shape[0]):
        if k2r[i, 0] <= -1 or k2r[i, 0] > size_board - 1 or k2r[i, 1] <= -1 or k2r[i, 1] > size_board - 1:
            continue
        else:
            if (np.abs(k2r[i, 0] - p_k1[0]) > 1 or np.abs(k2r[i, 1] - p_k1[1]) > 1) and dfQ1_[k2r[i, 0], k2r[i, 1]] == 0:
                dfK2[k2r[i, 0], k2r[i, 1]] = 1
                a_k2[i] = 1

    dfK2[p_k2[0], p_k2[1]] = 0

    check = 0
    if dfQ1_[p_k2[0], p_k2[1]] == 1:
        check = 1

    return dfK2, a_k2, check