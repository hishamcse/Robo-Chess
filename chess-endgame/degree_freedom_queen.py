import numpy as np


def degree_freedom_queen(p_k1, p_k2, p_q1, s):
    """
    This function returns a matrix of ones and zeros where 1 specify a location in which the Queen can move to. 
    The king will never choose an unsafe location. White playing with a king(k1) and a queen(q1) 
    and black is playing with only a king(k2)
    :param p_k1: position of King 1
    :param p_k2: position of King 2
    :param p_q1: position of Queen
    :param s: board
    :return: 
        dfQ1: Degrees of Freedom of the Queen 
        a_q1: Allowed actions for the Queen
        dfQ1_: Squares the Queen is threatening. Represents the potential influence of the queen on the board,
               marking squares it could theoretically attack or move to, ignoring certain constraints
    """

    size_board = s.shape[0]

    dfQ1 = np.zeros([size_board, size_board], dtype=int)
    dfQ1[p_q1[0], p_q1[1]] = 1

    dfQ1_ = np.zeros([size_board, size_board], dtype=int)  # King without King 2 reach
    dfQ1_[p_q1[0], p_q1[1]] = 1

    # King 2 reach
    k2r = [[p_k2[0] - 1, p_k2[1]],      # up
           [p_k2[0] + 1, p_k2[1]],      # down
           [p_k2[0], p_k2[1] - 1],      # left
           [p_k2[0], p_k2[1] + 1],      # right
           [p_k2[0] - 1, p_k2[1] - 1],  # up-left
           [p_k2[0] - 1, p_k2[1] + 1],  # up-right
           [p_k2[0] + 1, p_k2[1] - 1],  # down-left
           [p_k2[0] + 1, p_k2[1] + 1]]  # down-right
    k2r = np.array(k2r)

    # King 1 reach
    k1r = [[p_k1[0] - 1, p_k1[1]],      # up
           [p_k1[0] + 1, p_k1[1]],      # down
           [p_k1[0], p_k1[1] - 1],      # left
           [p_k1[0], p_k1[1] + 1],      # right
           [p_k1[0] - 1, p_k1[1] - 1],  # up-left
           [p_k1[0] - 1, p_k1[1] + 1],  # up-right
           [p_k1[0] + 1, p_k1[1] - 1],  # down-left
           [p_k1[0] + 1, p_k1[1] + 1]]  # down-right
    k1r = np.array(k1r)

    # Queen
    blocked = np.zeros(8, dtype=int)
    blocked2 = np.zeros(8, dtype=int)
    a_q1 = np.zeros([8 * (size_board - 1), 1], dtype=int)

    for j in range(size_board):
        # allow_down
        if p_q1[0] + j < size_board - 1:
            if (p_q1[0] + j + 1 == p_k1[0] and p_q1[1] == p_k1[1]) or ((p_q1[0] + j + 1 == p_k2[0] and p_q1[1] == p_k2[1])):
                blocked[0] = 1

            if p_q1[0] + j + 1 == p_k1[0] and p_q1[1] == p_k1[1]:
                blocked2[0] = 1

            if blocked[0] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] + j + 1 != k2r[i, 0] or p_q1[1] != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] + j + 1, p_q1[1]] = 1
                    a_q1[j] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] + j + 1 == k1r[ii, 0] and p_q1[1] == k1r[ii, 1]:
                                    dfQ1[p_q1[0] + j + 1, p_q1[1]] = 1
                                    a_q1[j] = 1
            if blocked2[0] == 0:
                dfQ1_[p_q1[0] + j + 1, p_q1[1]] = 1

        # allow_up
        if p_q1[0] - j > 0:
            if (p_q1[0] - j - 1 == p_k1[0] and p_q1[1] == p_k1[1]) or ((p_q1[0] - j - 1 == p_k2[0] and p_q1[1] == p_k2[1])):
                blocked[1] = 1

            if p_q1[0] - j - 1 == p_k1[0] and p_q1[1] == p_k1[1]:
                blocked2[1] = 1

            if blocked[1] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] - j - 1 != k2r[i, 0] or p_q1[1] != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] - j - 1, p_q1[1]] = 1
                    a_q1[j + (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] - j - 1 == k1r[ii, 0] and p_q1[1] == k1r[ii, 1]:
                                    dfQ1[p_q1[0] - j - 1, p_q1[1]] = 1
                                    a_q1[j + (size_board - 1)] = 1
            if blocked2[1] == 0:
                dfQ1_[p_q1[0] - j - 1, p_q1[1]] = 1

        # allow_right
        if p_q1[1] + j < size_board - 1:
            if (p_q1[0] == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]) or ((p_q1[0] == p_k2[0] and p_q1[1] + j + 1 == p_k2[1])):
                blocked[2] = 1

            if p_q1[0] == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]:
                blocked2[2] = 1

            if blocked[2] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] != k2r[i, 0] or p_q1[1] + j + 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0], p_q1[1] + j + 1] = 1
                    a_q1[j + 2 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] == k1r[ii, 0] and p_q1[1] + j + 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0], p_q1[1] + j + 1] = 1
                                    a_q1[j + 2 * (size_board - 1)] = 1
            if blocked2[2] == 0:
                dfQ1_[p_q1[0], p_q1[1] + j + 1] = 1

        # allow_left
        if p_q1[1] - j > 0:
            if (p_q1[0] == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]) or ((p_q1[0] == p_k2[0] and p_q1[1] - j - 1 == p_k2[1])):
                blocked[3] = 1

            if p_q1[0] == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]:
                blocked2[3] = 1

            if blocked[3] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] != k2r[i, 0] or p_q1[1] - j - 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0], p_q1[1] - j - 1] = 1
                    a_q1[j + 3 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] == k1r[ii, 0] and p_q1[1] - j - 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0], p_q1[1] - j - 1] = 1
                                    a_q1[j + 3 * (size_board - 1)] = 1
            if blocked2[3] == 0:
                dfQ1_[p_q1[0], p_q1[1] - j - 1] = 1

        # allow_down_right
        if p_q1[0] + j < size_board - 1 and p_q1[1] + j < size_board - 1:
            if (p_q1[0] + j + 1 == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]) or ((p_q1[0] + j + 1 == p_k2[0] and p_q1[1] + j + 1 == p_k2[1])):
                blocked[4] = 1

            if p_q1[0] + j + 1 == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]:
                blocked2[4] = 1

            if blocked[4] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] + j + 1 != k2r[i, 0] or p_q1[1] + j + 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] + j + 1, p_q1[1] + j + 1] = 1
                    a_q1[j + 4 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] + j + 1 == k1r[ii, 0] and p_q1[1] + j + 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0] + j + 1, p_q1[1] + j + 1] = 1
                                    a_q1[j + 4 * (size_board - 1)] = 1
            if blocked2[4] == 0:
                dfQ1_[p_q1[0] + j + 1, p_q1[1] + j + 1] = 1

        # allow_down_left
        if p_q1[0] + j < size_board - 1 and p_q1[1] - j > 0:
            if (p_q1[0] + j + 1 == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]) or ((p_q1[0] + j + 1 == p_k2[0] and p_q1[1] - j - 1 == p_k2[1])):
                blocked[5] = 1

            if p_q1[0] + j + 1 == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]:
                blocked2[5] = 1

            if blocked[5] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] + j + 1 != k2r[i, 0] or p_q1[1] - j - 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] + j + 1, p_q1[1] - j - 1] = 1
                    a_q1[j + 5 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] + j + 1 == k1r[ii, 0] and p_q1[1] - j - 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0] + j + 1, p_q1[1] - j - 1] = 1
                                    a_q1[j + 5 * (size_board - 1)] = 1
            if blocked2[5] == 0:
                dfQ1_[p_q1[0] + j + 1, p_q1[1] - j - 1] = 1

        # allow_up_right
        if p_q1[0] - j > 0 and p_q1[1] + j < size_board - 1:
            if (p_q1[0] - j - 1 == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]) or ((p_q1[0] - j - 1 == p_k2[0] and p_q1[1] + j + 1 == p_k2[1])):
                blocked[6] = 1

            if p_q1[0] - j - 1 == p_k1[0] and p_q1[1] + j + 1 == p_k1[1]:
                blocked2[6] = 1

            if blocked[6] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] - j - 1 != k2r[i, 0] or p_q1[1] + j + 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] - j - 1, p_q1[1] + j + 1] = 1
                    a_q1[j + 6 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] - j - 1 == k1r[ii, 0] and p_q1[1] + j + 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0] - j - 1, p_q1[1] + j + 1] = 1
                                    a_q1[j + 6 * (size_board - 1)] = 1
            if blocked2[6] == 0:
                dfQ1_[p_q1[0] - j - 1, p_q1[1] + j + 1] = 1

        # allow_up_left
        if p_q1[0] - j > 0 and p_q1[1] - j > 0:
            if (p_q1[0] - j - 1 == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]) or ((p_q1[0] - j - 1 == p_k2[0] and p_q1[1] - j - 1 == p_k2[1])):
                blocked[7] = 1

            if p_q1[0] - j - 1 == p_k1[0] and p_q1[1] - j - 1 == p_k1[1]:
                blocked2[7] = 1

            if blocked[7] == 0:
                tmp = np.zeros([8], dtype=int)
                for i in range(k2r.shape[0]):
                    if p_q1[0] - j - 1 != k2r[i, 0] or p_q1[1] - j - 1 != k2r[i, 1]:
                        tmp[i] = 1
                if np.all(tmp):
                    dfQ1[p_q1[0] - j - 1, p_q1[1] - j - 1] = 1
                    a_q1[j + 7 * (size_board - 1)] = 1
                else:
                    for i in range(len(tmp)):
                        if tmp[i] == 0:
                            for ii in range(k1r.shape[0]):
                                if p_q1[0] - j - 1 == k1r[ii, 0] and p_q1[1] - j - 1 == k1r[ii, 1]:
                                    dfQ1[p_q1[0] - j - 1, p_q1[1] - j - 1] = 1
                                    a_q1[j + 7 * (size_board - 1)] = 1
            if blocked2[7] == 0:
                dfQ1_[p_q1[0] - j - 1, p_q1[1] - j - 1] = 1

    dfQ1[p_q1[0], p_q1[1]] = 0
    dfQ1[p_k1[0], p_k1[1]] = 0
    if p_k2[0] != np.inf:
        dfQ1[p_k2[0], p_k2[1]] = 0

    return dfQ1, a_q1, dfQ1_
