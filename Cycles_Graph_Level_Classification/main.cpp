// Author: Anirudh Dagar 10/03/20

#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <Eigen>
#include "GCN_Model.h"
#include "data.h"

int main()
{
    // Import Cycles Graph Data
    auto adj_cycle = get_adj_cycle("resources/CYCLE/CYCLE_A.txt");

    //Get Graph Indiacator
    std::vector<int> gi_vec = get_graph_indicator("resources/CYCLE/CYCLE_graph_indicator.txt");

    //Get Graph Labels
    std::vector<int> gl_vec = get_graph_label("resources/CYCLE/CYCLE_graph_labels.txt");

    // Import Weights
    std::vector<Eigen::MatrixXd> weight_vec = getWeights_Cycle("resources/saved/saved_weights_cycle.txt");
    std::cout<<"Imported PyTorch Trained Weights"<<std::endl;


    int num_nodes = adj_cycle.rows();
    std::cout<<"num_nodes: "<<num_nodes<<std::endl;

    Eigen::MatrixXd feats = Eigen::MatrixXd::Identity(num_nodes, num_nodes);
    std::cout<<"Created feature matrix `X` or `feats` as Identity"<<std::endl;

    int n_feat = feats.rows();
    int n_hid  = 6;
    int n_hid2 = 4;
    int n_out  = 2;
    std::cout<<"n_feat:"<<n_feat<<", n_hid1:"<<n_hid<<", n_hid2:"<<n_hid2<<" n_out:"<<n_out<<std::endl<<"\n";

    std::cout<<"===============================================================\n";

    GCN model(adj_cycle, n_feat, n_hid, n_hid2, n_out, weight_vec, gi_vec);
    Eigen::MatrixXd predictions = model.forward(feats);

    std::cout<<"\nFinal Predictions:\n"<<predictions<<std::endl;

    writeTofile("resources/saved/cpp_predicted_cycle.txt", Eigen::MatrixXd (predictions));
    std::cout<<"Final Predictions saved!\n";

    return 0;
}
