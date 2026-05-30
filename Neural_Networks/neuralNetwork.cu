#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CycleTimer.h"
#include "neuralNetwork.h"

using namespace std;

// ---------------------------------------------------------------------------
// Helper macro for CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",            \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Fused sigmoid activation applied in-place on a flat buffer
// ---------------------------------------------------------------------------
__global__ void sigmoid_inplace(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = 1.f / (1.f + __expf(-data[i]));
}

// ---------------------------------------------------------------------------
// Sets bias neuron slots to -1.0 for every sample in a batch.
// neurons: batchSize x stride (row-major), biasIdx = stride-1
// ---------------------------------------------------------------------------
__global__ void set_bias_neurons(float* neurons, int stride, int batchSize)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batchSize)
        neurons[b * stride + (stride - 1)] = -1.f;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
neuralNetwork::neuralNetwork(int nI, int nH, int nO, int bS)
    : nInput(nI), nHidden(nH), nOutput(nO), batchSize(bS), weightsDirty(true)
{
    // ---- host neuron buffers ----
    inputNeurons  = new float[batchSize * (nInput  + 1)]();
    hiddenNeurons = new float[batchSize * (nHidden + 1)]();
    outputNeurons = new float[batchSize * nOutput]();

    // Set bias neuron slots to -1 on the host
    for (int b = 0; b < batchSize; b++) {
        inputNeurons [b * (nInput  + 1) + nInput ] = -1.f;
        hiddenNeurons[b * (nHidden + 1) + nHidden] = -1.f;
    }

    // ---- host weight matrices (flat, row-major) ----
    // wInputHidden[i] points into a single flat block; only [0] owns the memory.
    wInputHidden    = new float*[nInput  + 1];
    wInputHidden[0] = new float[(nInput  + 1) * nHidden]();
    for (int i = 1; i <= nInput; i++)
        wInputHidden[i] = wInputHidden[i - 1] + nHidden;

    wHiddenOutput    = new float*[nHidden + 1];
    wHiddenOutput[0] = new float[(nHidden + 1) * nOutput]();
    for (int i = 1; i <= nHidden; i++)
        wHiddenOutput[i] = wHiddenOutput[i - 1] + nOutput;

    // ---- device buffers ----
    CUDA_CHECK(cudaMalloc(&d_input,   sizeof(float) * batchSize * (nInput  + 1)));
    CUDA_CHECK(cudaMalloc(&d_hidden,  sizeof(float) * batchSize * (nHidden + 1)));
    CUDA_CHECK(cudaMalloc(&d_output1, sizeof(float) * batchSize * nHidden));
    CUDA_CHECK(cudaMalloc(&d_output2, sizeof(float) * batchSize * nOutput));
    CUDA_CHECK(cudaMalloc(&d_w1,      sizeof(float) * (nInput  + 1) * nHidden));
    CUDA_CHECK(cudaMalloc(&d_w2,      sizeof(float) * (nHidden + 1) * nOutput));

    // ---- cuBLAS ----
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // ---- weight initialisation ----
    initializeWeights();
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
neuralNetwork::~neuralNetwork()
{
    delete[] inputNeurons;
    delete[] hiddenNeurons;
    delete[] outputNeurons;

    // Only [0] owns the flat allocation; others are interior pointers
    delete[] wInputHidden[0];
    delete[] wInputHidden;
    delete[] wHiddenOutput[0];
    delete[] wHiddenOutput;

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_w1);
    cudaFree(d_w2);

    cublasDestroy(cublasHandle);
}

// ---------------------------------------------------------------------------
// Xavier uniform weight initialisation
// ---------------------------------------------------------------------------
void neuralNetwork::initializeWeights()
{
    // Layer 1: fan_in = nInput+1, fan_out = nHidden
    float limit1 = sqrtf(6.f / (float)(nInput + 1 + nHidden));
    for (int i = 0; i <= nInput; i++)
        for (int j = 0; j < nHidden; j++)
            wInputHidden[i][j] = limit1 * (2.f * ((float)rand() / RAND_MAX) - 1.f);

    // Layer 2: fan_in = nHidden+1, fan_out = nOutput
    float limit2 = sqrtf(6.f / (float)(nHidden + 1 + nOutput));
    for (int i = 0; i <= nHidden; i++)
        for (int j = 0; j < nOutput; j++)
            wHiddenOutput[i][j] = limit2 * (2.f * ((float)rand() / RAND_MAX) - 1.f);

    weightsDirty = true;
}

// ---------------------------------------------------------------------------
// Uploads weight matrices to the device when they have been modified.
// Called lazily at the start of every forward pass.
// ---------------------------------------------------------------------------
static inline void uploadWeightsIfDirty(neuralNetwork* nn)
{
    if (!nn->weightsDirty) return;
    CUDA_CHECK(cudaMemcpy(nn->d_w1, nn->wInputHidden[0],
        sizeof(float) * (nn->nInput + 1) * nn->nHidden,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->d_w2, nn->wHiddenOutput[0],
        sizeof(float) * (nn->nHidden + 1) * nn->nOutput,
        cudaMemcpyHostToDevice));
    nn->weightsDirty = false;
}

// ---------------------------------------------------------------------------
// Single-sample forward pass (used for accuracy evaluation)
//
// cuBLAS GEMV: d_output = sigmoid(W^T * input)
//
// Memory layout (row-major on host / as stored in device buffers):
//   d_w1  : (nInput+1) x nHidden   → cuBLAS sees (nHidden, nInput+1) col-major
//   d_input: vector of size (nInput+1)
//   d_output1: vector of size nHidden
// ---------------------------------------------------------------------------
void neuralNetwork::feedForward(float* pattern)
{
    // Pack input (bias slot was pre-set to -1)
    for (int i = 0; i < nInput; i++)
        inputNeurons[i] = pattern[i];

    uploadWeightsIfDirty(this);

    CUDA_CHECK(cudaMemcpy(d_input, inputNeurons,
        sizeof(float) * (nInput + 1), cudaMemcpyHostToDevice));

    const float alpha = 1.f, beta = 0.f;

    // Layer 1: d_output1 = d_w1^T * d_input  (dim: nHidden)
    // cuBLAS col-major: d_w1 is (nHidden x (nInput+1)), CUBLAS_OP_N gives nHidden-dim output
    CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_N,
        nHidden, nInput + 1,
        &alpha, d_w1, nHidden,
        d_input, 1,
        &beta, d_output1, 1));

    int n1 = nHidden;
    sigmoid_inplace<<<(n1 + 255) / 256, 256>>>(d_output1, n1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy hidden activations back and restore bias
    CUDA_CHECK(cudaMemcpy(hiddenNeurons, d_output1,
        sizeof(float) * nHidden, cudaMemcpyDeviceToHost));
    hiddenNeurons[nHidden] = -1.f;

    CUDA_CHECK(cudaMemcpy(d_hidden, hiddenNeurons,
        sizeof(float) * (nHidden + 1), cudaMemcpyHostToDevice));

    // Layer 2: d_output2 = d_w2^T * d_hidden  (dim: nOutput)
    CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_N,
        nOutput, nHidden + 1,
        &alpha, d_w2, nOutput,
        d_hidden, 1,
        &beta, d_output2, 1));

    int n2 = nOutput;
    sigmoid_inplace<<<(n2 + 255) / 256, 256>>>(d_output2, n2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputNeurons, d_output2,
        sizeof(float) * nOutput, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Batched forward pass (used during training)
//
// cuBLAS GEMM: D_output = sigmoid(W^T * D_input)
//
// For a batch of B samples:
//   d_input  : B x (nInput+1)  row-major → cuBLAS (nInput+1, B) col-major
//   d_w1     : (nInput+1) x nHidden → cuBLAS (nHidden, nInput+1) col-major
//   d_output1: B x nHidden result
//
// cuBLAS SGEMM (col-major):  C = alpha * A * B + beta * C
//   A = d_w1   : (nHidden   x (nInput+1))
//   B = d_input: ((nInput+1) x batchSize)
//   C = d_output1: (nHidden x batchSize)
// ---------------------------------------------------------------------------
void neuralNetwork::feedForwardBatch(std::vector<float*>& patternVector)
{
    int B = (int)patternVector.size();

    // Pack input batch (bias slots pre-set to -1 in constructor)
    for (int b = 0; b < B; b++)
        for (int i = 0; i < nInput; i++)
            inputNeurons[b * (nInput + 1) + i] = patternVector[b][i];

    uploadWeightsIfDirty(this);

    CUDA_CHECK(cudaMemcpy(d_input, inputNeurons,
        sizeof(float) * B * (nInput + 1), cudaMemcpyHostToDevice));

    const float alpha = 1.f, beta = 0.f;

    // Layer 1: d_output1 (nHidden x B) = d_w1 (nHidden x (nInput+1)) * d_input ((nInput+1) x B)
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nHidden, B, nInput + 1,
        &alpha, d_w1, nHidden,
        d_input, nInput + 1,
        &beta, d_output1, nHidden));

    int n1 = B * nHidden;
    sigmoid_inplace<<<(n1 + 255) / 256, 256>>>(d_output1, n1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy hidden activations back to host, set bias neurons, re-upload
    CUDA_CHECK(cudaMemcpy(hiddenNeurons, d_output1,
        sizeof(float) * B * nHidden, cudaMemcpyDeviceToHost));

    for (int b = 0; b < B; b++)
        hiddenNeurons[b * (nHidden + 1) + nHidden] = -1.f;

    CUDA_CHECK(cudaMemcpy(d_hidden, hiddenNeurons,
        sizeof(float) * B * (nHidden + 1), cudaMemcpyHostToDevice));

    // Layer 2: d_output2 (nOutput x B) = d_w2 (nOutput x (nHidden+1)) * d_hidden ((nHidden+1) x B)
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nOutput, B, nHidden + 1,
        &alpha, d_w2, nOutput,
        d_hidden, nHidden + 1,
        &beta, d_output2, nOutput));

    int n2 = B * nOutput;
    sigmoid_inplace<<<(n2 + 255) / 256, 256>>>(d_output2, n2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputNeurons, d_output2,
        sizeof(float) * B * nOutput, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Activation function (host-side, used for reference only)
// ---------------------------------------------------------------------------
inline float neuralNetwork::activationFunction(float x)
{
    return 1.f / (1.f + expf(-x));
}

// ---------------------------------------------------------------------------
// Evaluate classification accuracy on a data set
// ---------------------------------------------------------------------------
double neuralNetwork::getSetAccuracy(std::vector<dataEntry*>& set)
{
    double incorrect = 0;

    for (int tp = 0; tp < (int)set.size(); tp++) {
        feedForward(set[tp]->pattern);

        int predicted = (int)distance(outputNeurons,
            max_element(outputNeurons, outputNeurons + nOutput));
        int expected  = (int)distance(set[tp]->target,
            max_element(set[tp]->target, set[tp]->target + nOutput));

        if (predicted != expected)
            incorrect++;
    }

    return 100.0 - (incorrect / set.size() * 100.0);
}

// ---------------------------------------------------------------------------
// Save weights to CSV
// ---------------------------------------------------------------------------
bool neuralNetwork::saveWeights(char* filename)
{
    fstream out;
    out.open(filename, ios::out);
    if (!out.is_open()) {
        cout << "Error: cannot open '" << filename << "'" << endl;
        return false;
    }

    out.precision(10);
    for (int i = 0; i <= nInput; i++)
        for (int j = 0; j < nHidden; j++)
            out << wInputHidden[i][j] << ",";

    for (int i = 0; i <= nHidden; i++)
        for (int j = 0; j < nOutput; j++) {
            out << wHiddenOutput[i][j];
            if (i * nOutput + j + 1 != (nHidden + 1) * nOutput)
                out << ",";
        }

    cout << "Weights saved to '" << filename << "'" << endl;
    out.close();
    return true;
}

// ---------------------------------------------------------------------------
// Print CUDA device info
// ---------------------------------------------------------------------------
void neuralNetwork::printCudaInfo()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA device(s)\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("Device %d: %s\n", i, p.name);
        printf("   SMs:        %d\n", p.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               (float)p.totalGlobalMem / (1024.f * 1024.f));
        printf("   CUDA Cap:   %d.%d\n", p.major, p.minor);
    }
    printf("---------------------------------------------------------\n");
}
