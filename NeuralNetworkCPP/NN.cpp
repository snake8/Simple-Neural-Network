//
//  NN.cpp
//  NeuralNetworkCPP
//
//  Created by Daniel Solovich on 9/25/17.
//  Copyright © 2017 Daniel Solovich. All rights reserved.
//

#include "NN.hpp"



NeuralNetwork::NeuralNetwork(size_t input, size_t hidden, size_t output, float lr) : inputNeurons(input), hiddenNeurons(hidden), outputNeurons(output), learningRate(lr) {

        fillMatrixWithRandomValues(this->linksBetweenInputAndHiddenLayers, -pow(hiddenNeurons, -.5), pow(hiddenNeurons, -.5));

        fillMatrixWithRandomValues(this->linksBetweenHiddenAndOutputLayers, -pow(hiddenNeurons, -.5), pow(outputNeurons, -.5));

}


Matrix<double> sigmoidFunction(Matrix<double>& matrix) {
    Matrix<double> sigmoidResult(matrix.size1(), matrix.size2());
    for (size_t i = 0; i < sigmoidResult.size1(); i++) {
        for (size_t j = 0; j < matrix.size2(); j++) {
            sigmoidResult(i, j) = 1 / (1 + exp(-matrix(i, j)));
        }
    }
    return sigmoidResult;
}


Matrix<double> oneMinusMatrix(Matrix<double>* matrix) {
    Matrix<double>* resultMatrix = new Matrix<double>(matrix->size1(), matrix->size2());
    for (size_t i = 0; i < matrix->size1(); i++) {
        for (size_t j = 0; j < matrix->size2(); j++)
            resultMatrix->operator()(i, j) = 1 - matrix->operator()(i, j);
    }
    return *resultMatrix;
}

void updateWeight(Matrix<double>& weight, Matrix<double>& inputs, Matrix<double>& outputs, Matrix<double>& error, float learningRate) {
    Matrix<double> transposedInputs = transpose(&inputs);
    Matrix<double> oneMinusOutputs = oneMinusMatrix(&outputs);
    
    Matrix<double> outputError = multiplyMatrixValueByValue(error, outputs);
    Matrix<double> dotProduct1 = multiplyMatrixValueByValue(outputError, oneMinusOutputs);
    
    Matrix<double> matrixWithUpdatingWeight = dotProduct1 * transposedInputs;
    
    for (size_t i = 0; i < weight.size1(); i++) {
        for (size_t j = 0; j < weight.size2(); j++)
            weight(i, j) += learningRate * matrixWithUpdatingWeight(i, j);
    }
}

// backpropagation algorithm
void NeuralNetwork::trainNetwork(Matrix<double>& inputData, Matrix<double>& targets) {
    Matrix<double> inputs = transpose(&inputData);
    
    Matrix<double> hiddenInputs = this->linksBetweenInputAndHiddenLayers * inputs;
    Matrix<double> hiddenOutputs = sigmoidFunction(hiddenInputs);
    
    Matrix<double> finalInputs = this->linksBetweenHiddenAndOutputLayers * hiddenOutputs;
    Matrix<double> finalOutputs = sigmoidFunction(finalInputs);


    Matrix<double> outputError = targets - finalOutputs;
    Matrix<double> transposedLinksBetweenHiddenAndOutputLayers = transpose(&this->linksBetweenHiddenAndOutputLayers);

    Matrix<double> hiddenError = transposedLinksBetweenHiddenAndOutputLayers * outputError;


    updateWeight(this->linksBetweenHiddenAndOutputLayers, hiddenOutputs, finalOutputs, outputError, learningRate);


    updateWeight(this->linksBetweenInputAndHiddenLayers, inputs, hiddenOutputs, hiddenError, learningRate);

    std::cout << "Training in process!" << std::endl;

}


Matrix<double> NeuralNetwork::queryNetwork(Matrix<double>& inputData) {
    Matrix<double> inputs = transpose(&inputData);

    Matrix<double> hiddenInputs = this->linksBetweenInputAndHiddenLayers * inputs;
    Matrix<double> hiddenOutputs = sigmoidFunction(hiddenInputs);

    Matrix<double> finalInputs = this->linksBetweenHiddenAndOutputLayers * hiddenOutputs;
    Matrix<double> finalOutputs = sigmoidFunction(finalInputs);

    return finalOutputs;
}
