import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].transpose(2,0,1), self.y_p[idx], self.y_v[idx]


## Original AlphaZero Architecture

# class ConvBlock(nn.Module):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         self.action_size = 8*8*73
#         self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)

#     def forward(self, s):
#         s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
#         s = F.relu(self.bn1(self.conv1(s)))
#         return s

# class ResBlock(nn.Module):
#     def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = F.relu(self.bn1(out))
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = F.relu(out)
#         return out
    
# class OutBlock(nn.Module):
#     def __init__(self):
#         super(OutBlock, self).__init__()
#         self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
#         self.bn = nn.BatchNorm2d(1)
#         self.fc1 = nn.Linear(8*8, 64)
#         self.fc2 = nn.Linear(64, 1)
        
#         self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
#         self.bn1 = nn.BatchNorm2d(128)
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#         self.fc = nn.Linear(8*8*128, 8*8*73)
    
#     def forward(self,s):
#         v = F.relu(self.bn(self.conv(s))) # value head
#         v = v.view(-1, 8*8)  # batch_size X channel X height X width
#         v = F.relu(self.fc1(v))
#         v = F.tanh(self.fc2(v))
        
#         p = F.relu(self.bn1(self.conv1(s))) # policy head
#         p = p.view(-1, 8*8*128)
#         p = self.fc(p)
#         p = self.logsoftmax(p).exp()
#         return p, v
    
# class ChessNet(nn.Module):
#     def __init__(self):
#         super(ChessNet, self).__init__()
#         self.conv = ConvBlock()
#         for block in range(19):
#             setattr(self, "res_%i" % block,ResBlock())
#         self.outblock = OutBlock()
    
#     def forward(self,s):
#         s = self.conv(s)
#         for block in range(19):
#             s = getattr(self, "res_%i" % block)(s)
#         s = self.outblock(s)
#         return s
    

# Modified AlphaZero Architecture

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.res_conv = nn.Conv2d(22, 256, 1)  # To match dimensions of residual and conv output

    def forward(self, s):
        s = s.float()  # Ensure input is of type float
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        residual = self.res_conv(s)  # Adjust residual dimensions to match conv output
        s = F.relu(self.bn1(self.conv1(s)))
        s = self.bn2(self.conv2(s))
        s += residual
        s = F.relu(s)
        return s

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = x.float()  # Ensure input is of type float
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        x = x.float()  # Ensure input is of type float
        b, c, _, _ = x.size()
        y = F.avg_pool2d(x, x.size(2)).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(inplanes, planes, stride=stride)
        self.conv2 = DepthwiseSeparableConv(planes, planes, stride=stride)
        self.se = SEBlock(planes)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = x.float()  # Ensure input is of type float
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self, s):
        s = s.float()  # Ensure input is of type float
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(10)])  # Reduced number of ResBlocks
        self.outblock = OutBlock()
    
    def forward(self, s):
        s = s.float()  # Ensure input is of type float
        s = self.conv(s)
        for block in self.res_blocks:
            s = block(s)
        p, v = self.outblock(s)
        return p, v
        

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
def train(net, dataset, epoch_start=0, epoch_stop=20, cpu=0):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i,data in enumerate(train_loader,0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1)*30, len(train_set), total_loss/10))
                print("Policy:",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Value:",value[0].item(),value_pred[0,0].item())
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break

    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1,epoch_stop+1,1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_%s.png" % datetime.datetime.today().strftime("%Y-%m-%d")))
