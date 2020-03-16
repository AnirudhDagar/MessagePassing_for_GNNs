// Author: Anirudh Dagar 10/03/20

/*//////////////////////////////////////////////////////////////////////////
// C++ Data utils to read in arbitrary sized graph data,   				  //
// eg: Karate Club Dataset. Also provides functionality to read           //
// the trained weight matrices into an Eigen Matrix for the forward       //
// pass of Message Passing Neural Networks eg GCN                         //
//////////////////////////////////////////////////////////////////////////*/

#pragma once

#ifndef DATA_UTILS
#define DATA_UTILS

#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <Eigen>


auto read_karate() {
  std::string line;
  std::ifstream myfile ("resources/out.ucidata-zachary");

	int edges, num_nodes;

	if (getline (myfile, line))
	{
		// Remove the first line
		// pass;
	}
	// Get number of edges and num nodes 
	if (getline (myfile, line))
	{
		std::stringstream temp(line.substr(2,line.length()-2));
		temp>>edges;
		temp>>num_nodes;

	}

	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
	for(int i=0; i<edges; i++){
		int x, y =0;
		myfile>>x;
		myfile>>y;
		data(x-1,y-1) = 1;
		data(y-1,x-1) = 1;
	}

	// std::cout<<data;
	return data;
}

auto getWeights(std::string address){
	std::ifstream myfile(address);
	if(!myfile.is_open()){
		std::cout<<"Empty weights!";
	}
   	int rows,cols;
   	myfile>>rows;
   	myfile>>cols;
   	Eigen::MatrixXd weights1 = Eigen::MatrixXd::Zero(rows, cols);
   	for(int i=0; i<rows; i++){
   		for(int j=0; j<cols; j++){
   			myfile >> weights1(i,j);
   		}
   	}
   	myfile>>rows;
   	myfile>>cols;
   	Eigen::MatrixXd weights2 = Eigen::MatrixXd::Zero(rows, cols);
   	for(int i=0; i<rows; i++){
   		for(int j=0; j<cols; j++){
   			myfile >> weights2(i,j);
   		}
   	}
   	myfile.close();
   	
   	std::vector <Eigen::MatrixXd> out_weights;
   	out_weights.push_back(weights1);
   	out_weights.push_back(weights2);
   	
   	return out_weights;
}


void writeTofile(std::string name, Eigen::MatrixXd matrix)
{
  std::ofstream file(name.c_str());

  for(int i=0; i<matrix.rows(); i++)
  {
      for(int j=0; j<matrix.cols(); j++)
      {
         std::string str = std::to_string(matrix(i,j));
         if(j+1 == matrix.cols()){
             file<<str;
         }else{
             file<<str<<" ";
         }
      }
      file<<'\n';
  }
}


#endif //DATA_UTILS
