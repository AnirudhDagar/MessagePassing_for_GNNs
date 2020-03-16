// Author: Anirudh Dagar 10/03/20

/*//////////////////////////////////////////////////////////////////////////
// Definition of the GCNConv class and relu method                        //
// for the graph convolutional operator from the paper:                   //
// Semi-supervised Classification with Graph Convolutional Networks"      //
// <https://arxiv.org/abs/1609.02907> paper  used for Message Passing     //
// Neural Networks.                                                       //
//////////////////////////////////////////////////////////////////////////*/

#pragma once

#ifndef LAYERS_MPNN
#define LAYERS_MPNN

#include <iostream>
#include <Eigen>


class GCNConv
{
private:
    Eigen::MatrixXd adj;
    Eigen::MatrixXd weight;
    int in_channels;
    int out_channels;
    int num_nodes;

public:
    // Constructor
    GCNConv(int in_channels, int out_channels, Eigen::MatrixXd adj, Eigen::MatrixXd weight,
        bool normalize=true)
    {
        this->adj       = adj;
        this->weight    = weight;
        num_nodes       = adj.rows();
        in_channels     = in_channels;
        out_channels    = out_channels;

        // Add self-loops to Adjacecny Matrix
        this->adj = this->adj + Eigen::MatrixXd::Identity(num_nodes, num_nodes);

        //Degree Diagonal Matrix D
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
        for(int i=0; i<num_nodes; i++)
        {
            for(int j=0; j<num_nodes; j++)
            {
                if(i==j)
                {
                    D(i,j) = this->adj.rowwise().sum()(i);
                }
            }
        }

        // Symmetric Normalization of Adjacency Matrix
        D = D.inverse();
        D = D.cwiseSqrt();
        this->adj = (D * this->adj) * D;

        // print norm to compare with Pytorch adjacency matrix
        // std::cout<<"Norm of Adjacency Matrix after normalization: "<<this->adj.norm()<<std::endl;
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd x)
    {
        std::cout<<"\n Dimensions of Weight Matrix:\n";
        std::cout<<this->weight.rows()<<", "<<this->weight.cols();

        Eigen::MatrixXd xw  = x * this->weight;
        Eigen::MatrixXd axw = this->adj * xw;

        return axw;
    }

};

Eigen::MatrixXd relu(Eigen::MatrixXd &out)
{ 
    return out.array().cwiseMax(0.0);
}

#endif //LAYERS_MPNN