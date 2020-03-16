import argparse, os
from arguments import buildParser
import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_karate, create_adjacency
from model import GCN
import numpy
import warnings

# Comment out the imports if visualization not needed
import matplotlib.pyplot as plt
import imageio
from celluloid import Camera
# Set ffmeg path for animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
warnings.filterwarnings("ignore")


parser  = buildParser()
args    = parser.parse_args()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

# Seed for reproducible numbers
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

edge_list = read_karate(args.datapath) 
adj = create_adjacency(edge_list, sparse=False)

# Need only 2 labels for semi-supervised classification
# -1 in target is ignored in the loss function.
target = [-1]*34

# Class Label for Admin (Node 1): 0
target[0] = 0
# Class Label for Instructor(Node 34): 1
target[33] = 1

target = torch.tensor(target)

true_labels = [0, 0, 0, 0 , 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
                1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# Using One-Hot Encoding of the nodes for initializing feature matrix x
feat = torch.eye(adj.size(0), dtype=torch.float32)

n_feat = feat.size(0)  # n_feat = 34
n_hid = args.hid       # n_hid = 10
n_out = args.out       # out = 2

# Initialize the GCN model
model = GCN(adj, n_feat, n_hid, n_out)

#### Training ####
criterion = nn.CrossEntropyLoss(ignore_index = -1)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if not args.no_vis:
    #### Plot animation using celluloid
    fig = plt.figure()
    camera = Camera(fig)

model.train()
for i in range(args.epochs):
    optimizer.zero_grad()
    pred = model(feat)

    loss = criterion(pred, target)
    if i%args.print_freq==0:
        print("Cross Entropy Loss: ", loss.item())
    
    loss.backward()
    optimizer.step()

    if not args.no_vis:
        plt.scatter(pred.detach().numpy()[:,0], pred.detach().numpy()[:,1], c=true_labels)
        for i in range(pred.shape[0]):
            text_plot = plt.text(pred[i,0], pred[i,1], str(i+1))

        camera.snap()

        if i%args.print_freq==0:
            print("Cross Entropy Loss: ", loss.item())

if not args.no_vis:
    animation = camera.animate(blit=False, interval=150)
    animation.save(args.savepath + 'train_karate_animation.mp4', writer='ffmpeg', fps=60)


# Save Parameters
params = list(model.parameters())
print("Learnt Model Parameters:", params)

with open(args.savepath + args.weight_filename + '.txt', 'w') as f:
    for param in params:
        f.write(str(param.shape[0]) + " " + str(param.shape[1]))
        f.write('\n')

        param_str = str(param.data.tolist())
        param_str = param_str.replace('[', '')
        param_str = param_str.replace(']', '')
        param_str = param_str.replace(',', '')

        f.write(param_str)
        f.write('\n')

with open(args.savepath + args.predictions_filename + '.txt', 'w') as f:
    final_pred = model(feat)
    pred_str = str(final_pred.data.tolist())
    pred_str = pred_str.replace('[', '')
    pred_str = pred_str.replace(']', '\n')
    pred_str = pred_str.replace(',', '')
    f.write(pred_str)
    f.write('\n')