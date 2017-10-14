//
//  NN.hpp
//  NeuralNetworkCPP
//
//  Created by Daniel Solovich on 9/25/17.
//  Copyright © 2017 Daniel Solovich. All rights reserved.
//

#ifndef NN_hpp
#define NN_hpp
#include <stdio.h>
#include <iostream>
#include "Matrix.h" 
#include <random>
#include "PutDataFromCsvInMatrix.hpp"

class NeuralNetwork {
private:
    size_t inputNeurons, hiddenNeurons, outputNeurons;
    float learningRate;
    Matrix<double> linksBetweenInputAndHiddenLayers = {hiddenNeurons, inputNeurons};
    Matrix<double> linksBetweenHiddenAndOutputLayers = {outputNeurons, hiddenNeurons};
public:
    NeuralNetwork() = default;
    NeuralNetwork(size_t, size_t, size_t, float);
    void trainNetwork(Matrix<double>& inputData, Matrix<double>& targets);
    Matrix<double> queryNetwork(Matrix<double>&);
};

/*
class neuralNetworkLayer : public NeuralNetwork {
private:
    size_t numberOfHiddenLayers;
public:
    neuralNetworkLayer() = default;
    neuralNetworkLayer(size_t numOfHiddden);
    
};
 */ 


Matrix<double> sigmoidFunction(Matrix<double>&);
Matrix<double> valueMinusAllMatrixValues(Matrix<double>& matrix, int number);
#endif /* NN_hpp */
