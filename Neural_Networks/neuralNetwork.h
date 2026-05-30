#ifndef NNETWORK_H
#define NNETWORK_H

#include <vector>
#include <cublas_v2.h>
#include "dataReader.h"

class nnTrain;

class neuralNetwork
{
private:
    int nInput, nHidden, nOutput, batchSize;

    // Host-side neuron activations (used for data loading and accuracy eval)
    float* inputNeurons;
    float* hiddenNeurons;
    float* outputNeurons;

    // Host-side weight matrices (flat, row-major)
    // wInputHidden: (nInput+1) x nHidden
    // wHiddenOutput: (nHidden+1) x nOutput
    float** wInputHidden;
    float** wHiddenOutput;

    // Device-side buffers
    float* d_input;       // batchSize x (nInput+1)
    float* d_hidden;      // batchSize x (nHidden+1)
    float* d_output1;     // batchSize x nHidden  (layer-1 pre/post activation)
    float* d_output2;     // batchSize x nOutput  (layer-2 pre/post activation)
    float* d_w1;          // (nInput+1)  x nHidden
    float* d_w2;          // (nHidden+1) x nOutput

    cublasHandle_t cublasHandle;
    bool weightsDirty;    // true when CPU weights have been updated and must be re-uploaded

    friend class nnTrain;

public:
    neuralNetwork(int numInput, int numHidden, int numOutput, int batchSize);
    ~neuralNetwork();

    bool saveWeights(char* outputFilename);
    double getSetAccuracy(std::vector<dataEntry*>& set);
    void printCudaInfo();

private:
    void initializeWeights();
    inline float activationFunction(float x);
    void feedForward(float* pattern);
    void feedForwardBatch(std::vector<float*>& patternVector);
};

#endif
