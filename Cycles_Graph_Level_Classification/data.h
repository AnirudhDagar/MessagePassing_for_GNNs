// #pragma once
/////////// CYCLE ////////////
#ifndef DATA_UTILS
#define DATA_UTILS

#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <Eigen>
#include <vector>

Eigen::MatrixXd get_adj_cycle(std::string str_name)
{
  std::string line;
  std::ifstream myfile(str_name);

  int edges = 0, num_nodes = 0, maxx = 0;
  while(getline (myfile, line)){
    edges++;
    int x, y =0;
    int ptr = 0;
    for(int i=0; i<line.length(); i++){
      if(line[i]==','){
        ptr = i;
        break;
      }
    }
    x = stoi(line.substr(0,ptr));
    y = stoi(line.substr(ptr+2));
    maxx = (x>y)?x:y;
    num_nodes = (maxx>num_nodes)? maxx: num_nodes;
  }
  std::ifstream again("CYCLE/CYCLE_A.txt");
  Eigen::MatrixXd data = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
  while(getline (again, line))
  {
    int x, y =0;
    int ptr = 0;
    for(int i=0; i<line.length(); i++){
      if(line[i]==','){
        ptr = i;
        break;
      }
    }
    x = stoi(line.substr(0,ptr));
    y = stoi(line.substr(ptr+2));
    data(x-1,y-1) = 1;
    data(y-1,x-1) = 1;
  }
  std::cout<<"Adjacency Matrix Shape for "<<str_name<<":"<<data.rows()<<" "<<data.cols();
  return data;
}

auto getWeights_Cycle(std::string address){
    std::ifstream myfile(address);
    if(!myfile.is_open()){
        std::cout<<"Empty weights!";
    }
    int rows,cols;
    int num_weights;
    myfile>>num_weights;
    std::vector <Eigen::MatrixXd> out_weights;
    while(num_weights--)
    {
      myfile>>rows;
      myfile>>cols;
      Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(rows, cols);
      for(int i=0; i<rows; i++){
          for(int j=0; j<cols; j++){
              myfile >> weights(i,j);
          }
      }
      out_weights.push_back(weights);
    }
    
    std::cout<<std::endl<<out_weights[0]<<"\n"<<out_weights[1]<<"\n"<<out_weights[2]<<"\n"<<out_weights[3]<<"\n";
    std::cout<<"here";
    return out_weights;
}

/////////////////////////////////////

std::vector<int> get_graph_indicator(std::string strname)
{
  std::string line;
  std::ifstream myfile(strname);

  std::vector<int> vec;
  int x;
  while(getline (myfile, line)){
    vec.push_back(stoi(line));
  }
  return vec;
}
////////////////////////////////////

std::vector<int> get_graph_label(std::string strname){
  std::string line;
  std::ifstream myfile(strname);

  std::vector<int> vec;
  int x;
  while(getline (myfile, line)){
    vec.push_back(stoi(line));
  }
  return vec;
}

//////////////////////////////////////
// Uncomment if node labels given

// auto get_nl(){
//   std::string line;
//   std::ifstream myfile("CYCLE/CYCLE_node_labels.txt");

//   std::vector<int> vec;
//   int x;
//   while(getline (myfile, line)){
//     vec.push_back(stoi(line));
//   }
//   int len = vec.size();
//   Eigen::MatrixXd data = Eigen::MatrixXd::Zero(len, 3);
//   for(int i=0; i<len; i++){
//     int t = vec[i];
//     data(i,t-1) = 1;
//   }
//   return data;
// }

/////////////////////////////////////

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