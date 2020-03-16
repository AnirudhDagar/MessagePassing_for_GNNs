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
	// Import Karate Club Data
	auto adj_karate = read_karate();

	// Import Weights 
	std::string address = "resources/saved/saved_weights.txt";
	std::vector<Eigen::MatrixXd> weight_vec = getWeights(address);
	std::cout<<"Imported PyTorch Trained Weights"<<std::endl;

	int num_nodes = adj_karate.rows();
	std::cout<<"num_nodes: "<<num_nodes<<std::endl;

  	Eigen::MatrixXd feats = Eigen::MatrixXd::Identity(num_nodes, num_nodes);
  	std::cout<<"Created feature matrix `X` or `feats` as Identity"<<std::endl;

  	int n_feat = feats.rows();
	int n_hid = 10;
	int n_out = 2;
	std::cout<<"n_feat:"<<n_feat<<", n_hid:"<<n_hid<<", n_out:"<<n_out<<std::endl<<"\n";

	std::cout<<"===============================================================\n";

  	GCN model(adj_karate, n_feat, n_hid, n_out, weight_vec);
	Eigen::MatrixXd predictions = model.forward(feats);

  	std::cout<<"\nFinal Predictions:\n"<<predictions<<std::endl;

  	writeTofile("resources/saved/cpp_predicted.txt", Eigen::MatrixXd (predictions));
  	std::cout<<"Final Predictions saved!\n";

	return 0;
}
