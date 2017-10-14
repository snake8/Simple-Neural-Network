//
//  main.cpp
//  NeuralNetworkCPP
//
//  Created by Daniel Solovich on 9/25/17.
//  Copyright © 2017 Daniel Solovich. All rights reserved.
//

#include "NN.hpp"
#include <vector>


// neural network parametrs
size_t inputNeurons = 784;
size_t hiddenNeurons = 100;
size_t outputNeurons = 10;
float learningRate = .1;
size_t trainingCycles = 200;
NeuralNetwork nn = {inputNeurons, hiddenNeurons, outputNeurons, learningRate};
clock_t t1, t2;
std::string pathToTestFile = "PATHTOFILE";
std::string pathToTrainFile = "PATHTOFILE";

void trainNetworkFunction();
Matrix<double> getNeuralNetworkResult(size_t indexOfSpacificNumber);


int main() {
    trainNetworkFunction();
    Matrix<double> neuralNetworkResult = getNeuralNetworkResult(2);
    showNeuralNetworkResult(neuralNetworkResult);
    return 0;
}

void trainNetworkFunction() {
    for (size_t i = 0; i < trainingCycles; i++) {
        Matrix<double> allTrainValues = putDataIntoMatrix(pathToTrainFile);
        Matrix<double> targets(outputNeurons, 1);
        Matrix<double> dataForPushingInNetwork;
        for (size_t row = 0; row < allTrainValues.size1(); row++) {
            dataForPushingInNetwork = scaleMatrixForNetwork(&allTrainValues, row);
            targets = .01;
            targets((size_t)allTrainValues(row, 0), 0) = .99;
            nn.trainNetwork(dataForPushingInNetwork, targets);
        }
    }
}

Matrix<double> getNeuralNetworkResult(size_t indexOfSpacificNumber) {
    Matrix<double> allTestValues = putDataIntoMatrix(pathToTestFile);
    Matrix<double> testInputsData;
    testInputsData = scaleMatrixForNetwork(&allTestValues, indexOfSpacificNumber);
    return nn.queryNetwork(testInputsData);
}
