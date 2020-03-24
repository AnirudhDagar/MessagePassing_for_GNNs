// Author: Anirudh Dagar 10/03/20

/*//////////////////////////////////////////////////////////////////////////
// Definition of the GCN class, which uses the graph Convolutional        //
// operator GCNConv from layers.h to implement                            //
// Semi-supervised Classification with Graph Convolutional Networks"      //
// <https://arxiv.org/abs/1609.02907> paper used for Message Passing      //
// Neural Networks.                                                       //
//////////////////////////////////////////////////////////////////////////*/

#pragma once

#ifndef GCN_MODEL
#define GCN_MODEL

#include <iostream>
#include <Eigen>
#include "layers.h"


class GCN
{

private:
    std::vector<int> gi_vec;
    std::vector<Eigen::MatrixXd> weights;

public:
    // Default Constructor
    GCNConv conv1;
    GCNConv conv2;
    GCN (Eigen::MatrixXd A, int nfeat, int nhid, int nhid2, int nout, std::vector<Eigen::MatrixXd> weights, std::vector<int> gi_vec): 
    conv1(nfeat, nhid, A, weights[0], true), conv2(nhid, nhid2, A,  weights[1], true)
    {
        this->weights    = weights;
        this->gi_vec     = gi_vec;
    }
        
    Eigen::MatrixXd forward(Eigen::MatrixXd x)
    {
        Eigen::MatrixXd h1      = conv1.forward(x);
        h1                      = relu(h1);
        Eigen::MatrixXd h2      = conv2.forward(h1); 
        h2                      = relu(h2);
        Eigen::MatrixXd pooled  = func_pool(h2, this->gi_vec);
        Eigen::MatrixXd fc_out  = pooled * weights[2].transpose();
        
        return softmax(fc_out);
    }
};



#endif //GCN_MODEL