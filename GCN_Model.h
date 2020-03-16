// Author: Anirudh Dagar 10/03/20

/*//////////////////////////////////////////////////////////////////////////
// Definition of the GCN class, which uses the graph Convolutional        //
// operator GCNConv from layers.h to implement                            //
// Semi-supervised Classification with Graph Convolutional Networks"      //
// <https://arxiv.org/abs/1609.02907> paper  used for Message Passing     //
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
public:
    // Default Constructor
    GCNConv conv1;
    GCNConv conv2;
    GCN (Eigen::MatrixXd A, int nfeat, int nhid, int nout, std::vector<Eigen::MatrixXd> weights): 
    conv1(nfeat, nhid, A, weights[0], true), conv2(nhid, nout, A,  weights[1], true)
    {}
        
    Eigen::MatrixXd forward(Eigen::MatrixXd x)
    {
        Eigen::MatrixXd h1  = conv1.forward(x);
        h1                  = relu(h1);
        Eigen::MatrixXd h2  = conv2.forward(h1); 
        h2                  = relu(h2);

        return h2;
    }

};

#endif //GCN_MODEL