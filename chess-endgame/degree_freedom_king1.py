import numpy as np


def degree_freedom_king1(p_k1, p_k2, p_q1, s):
    """
    This function returns a matrix of ones and zeros where 1 specify a location in which the King can move to. 
    The king will never choose an unsafe location.
    :param p_k1: position of King 1
    :param p_k2: position of King 2
    :param p_q1: position of Queen
    :param s: board
    :return: 
        dfK1: Degrees of Freedom of King 1
        a_k1: Allowed actions for King 1
        dfK1_: Squares the King1 is threatening
    """

    size_board = s.shape[0]

    dfK1 = np.zeros([size_board, size_board], dtype=int)
    dfK1[p_k1[0], p_k1[1]] = 1

    dfK1_= np.zeros([size_board, size_board], dtype=int)  # King without King 2 reach
    dfK1_[p_q1[0], p_q1[1]] = 1

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

    # King 1
    a_k1 = np.zeros([8, 1], dtype=int)

    # allow_down = 0
    if p_k1[0] < (size_board - 1):
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] != p_q1[1]:
            dfK1_[p_k1[0] + 1, p_k1[1]] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1]] = 1
                a_k1[0] = 1

    # allow_up = 0
    if p_k1[0] > 0:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1]] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1]] = 1
                a_k1[1] = 1

    # allow_right = 0
    if p_k1[1] < (size_board - 1):
        if p_k1[0] != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0], p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0], p_k1[1] + 1] = 1
                a_k1[2] = 1

    # allow_left = 0
    if p_k1[1] > 0:
        if p_k1[0] != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0], p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0], p_k1[1] - 1] = 1
                a_k1[3] = 1

    # allow_down_right = 0
    if p_k1[0] < (size_board - 1) and p_k1[1] < (size_board - 1):
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0] + 1, p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1] + 1] = 1
                a_k1[4] = 1

    # allow_down_left = 0
    if p_k1[0] < (size_board - 1) and p_k1[1] > 0:
        if p_k1[0] + 1 != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0] + 1, p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] + 1 != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] + 1, p_k1[1] - 1] = 1
                a_k1[5] = 1

    # allow_up_right = 0
    if p_k1[0] > 0 and p_k1[1] < size_board - 1:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] + 1 != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1] + 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] + 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1] + 1] = 1
                a_k1[6] = 1

    # allow_up_left = 0
    if p_k1[0] > 0 and p_k1[1] > 0:
        if p_k1[0] - 1 != p_q1[0] or p_k1[1] - 1 != p_q1[1]:
            dfK1_[p_k1[0] - 1, p_k1[1] - 1] = 1
            # It is not the Queen's position
            tmp = np.zeros([k2r.shape[0]], dtype=int)
            for i in range(k2r.shape[0]):
                if p_k1[0] - 1 != k2r[i, 0] or p_k1[1] - 1 != k2r[i, 1]:
                    tmp[i] = 1

            # check if it will be within the reach of King 2
            if np.all(tmp):
                dfK1[p_k1[0] - 1, p_k1[1] - 1] = 1
                a_k1[7] = 1

    # previous location
    dfK1[p_k1[0], p_k1[1]] = 0

    return dfK1, a_k1, dfK1_
