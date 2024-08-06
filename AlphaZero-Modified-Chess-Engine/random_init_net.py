from alpha_net import ChessNet, train
import torch
import os
net_to_train="current_net_trained8_iter1.pth.tar"
save_as="current_net_trained8_iter1.pth.tar"

net = ChessNet()
print()

print(os.path.join("./model_data/", save_as))
torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", save_as))