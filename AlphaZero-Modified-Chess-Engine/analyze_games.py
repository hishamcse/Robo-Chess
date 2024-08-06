import os
import numpy as np
import pickle
import encoder_decoder as ed
from visualize_board import view_board as vb
import matplotlib.pyplot as plt

data_path = "./datasets/iter2/"
file = "dataset_cpu1_5"
filename = os.path.join(data_path,file)
with open(filename, 'rb') as fo:
    dataset = pickle.load(fo, encoding='bytes')

last_move = np.argmax(dataset[-1][1])
b = ed.decode_board(dataset[-1][0])
act = ed.decode_action(b,last_move)

b.move_piece(act[0][0],act[1][0],act[2][0])
for i in range(len(dataset)):
    board = ed.decode_board(dataset[i][0])
    fig = vb(board.current_board)
    plt.savefig(os.path.join("images/", f"{file}_{i}.png"))
    
fig = vb(b.current_board)
plt.savefig(os.path.join("images/", f"{file}_{i+1}.png"))