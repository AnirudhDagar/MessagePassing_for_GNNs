import argparse, os
from arguments import buildParser
import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_karate, create_adjacency, create_one_hot_feats
from utils import read_enzyme_A, read_enzyme_node_label, read_enzyme_graph_indicator, read_enzyme_graph_label
from model import GCN_graph_level
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

################ Adjacency CYCLE A #########################
adj_enzymes_path = args.datapath
edge_list = read_enzyme_A(adj_enzymes_path + "CYCLE_A.txt" )

adj_enzyme = create_adjacency(edge_list)
################ Adjacency CYCLE A #########################


# ################ Node feat X #########################
feat = torch.eye(adj_enzyme.size(0), dtype=torch.float32)
################ Node feat X #########################


################ Each Node and Corresponding Graph Label ################
graph_indicator_enzymes_path = args.datapath 
graph_indicator = read_enzyme_graph_indicator(graph_indicator_enzymes_path + "CYCLE_graph_indicator.txt")
indicator = torch.tensor(graph_indicator) - 1
################ Each Node and Corresponding Graph Label ################


################ TRUE LABELS graph_labels#########################
graph_label_enzymes_path = args.datapath
graph_labels = read_enzyme_graph_label(graph_label_enzymes_path + "CYCLE_graph_labels.txt")
graph_labels_ohe = torch.tensor(graph_labels)
print(graph_labels_ohe)
################ TRUE LABELS graph_labels#########################



n_feat = feat.size(1)       # n_feat = 60
n_hid  = args.nhid1         # n_hid  = 6
n_hid2 = args.nhid2         # n_hid2 = 4
n_out  = args.out           # n_out  = 2

# Initialize the GCN model
model = GCN_graph_level(adj_enzyme, n_feat, n_hid, n_hid2, n_out, indicator)

#### Training ####
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if not args.no_vis:
    #### Plot animation using celluloid
    fig = plt.figure()
    camera = Camera(fig)

model.train()
for i in range(args.epochs):
    optimizer.zero_grad()
    pred = model(feat)

    loss = criterion(pred, graph_labels_ohe)
    if i%args.print_freq==0:
        print("Cross Entropy Loss: ", loss.item())
    
    loss.backward()
    optimizer.step()

    if not args.no_vis:
        plt.scatter(pred.detach().numpy()[:,0], pred.detach().numpy()[:,1], c=graph_labels)
        for i in range(pred.shape[0]):
            text_plot = plt.text(pred[i,0], pred[i,1], str(i+1))

        camera.snap()

if not args.no_vis:
    animation = camera.animate(blit=False, interval=150)
    animation.save(args.savepath + 'train_CYCLE_animation.mp4', writer='ffmpeg', fps=60)


# Save Parameters
params = list(model.parameters())
print("Learnt Model Parameters:", params)

with open(args.savepath + args.weight_filename + ".txt", 'w') as f:
    f.write(str(len(params))+'\n')
    for param in params:
        if len(param.shape)==2:
            f.write(str(param.shape[0]) + " " + str(param.shape[1]))
        else:
            f.write(str(param.shape[0]))
        f.write('\n')

        param_str = str(param.data.tolist())
        param_str = param_str.replace('[', '')
        param_str = param_str.replace(']', '')
        param_str = param_str.replace(',', '')

        f.write(param_str)
        f.write('\n')

with open(args.savepath + args.predictions_filename + ".txt", 'w') as f:
    final_pred = model(feat)
    pred_str = str(final_pred.data.tolist())
    pred_str = pred_str.replace('[', '')
    pred_str = pred_str.replace(']', '\n')
    pred_str = pred_str.replace(',', '')
    f.write(pred_str)
    f.write('\n')