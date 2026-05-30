#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "neuralNetworkTrainer.h"
#include "CycleTimer.h"

using namespace std;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Backward kernel: compute input-hidden weight gradients for a single sample.
//
// One block per hidden unit (blockIdx.x = hidden unit index).
// Threads cover input units (threadIdx.x < nInput+1).
//
// Warp-shuffle reduction replaces the original single-thread sequential loop.
// Works correctly when nOutput <= 32 (MNIST: 10 outputs — fine).
//
// Output: device_output[(linearThreadIndex) * nHidden + unit]
//         = lr * input[linearThreadIndex] * h*(1-h) * weightedSum
// ---------------------------------------------------------------------------
__global__ void back_prop_kernel(
    float* device_output,
    const float* inP,
    const float* m_hidden,
    const float* weights_2,
    const float* o_errG,
    int nInput, int nHidden, int nOutput, float l_R)
{
    int linearThreadIndex = threadIdx.x;
    int unit = blockIdx.x;  // hidden unit index

    // Each thread in the first warp handles one output unit for the weighted sum
    float ws = 0.f;
    if (linearThreadIndex < nOutput)
        ws = weights_2[unit * nOutput + linearThreadIndex] * o_errG[linearThreadIndex];

    // Warp-level reduction (safe for nOutput <= 32)
    for (int mask = 16; mask > 0; mask >>= 1)
        ws += __shfl_xor_sync(0xffffffff, ws, mask);

    float weightedSum = ws;  // broadcast: all threads in warp 0 hold the total

    __syncthreads();

    float h = m_hidden[unit];
    float delta = l_R * h * (1.f - h) * weightedSum;

    if (linearThreadIndex < nInput)
        device_output[linearThreadIndex * nHidden + unit] =
            inP[linearThreadIndex] * delta;
}

// ---------------------------------------------------------------------------
// Backward kernel: compute input-hidden weight gradients for a batch.
//
// One block per (batch, hidden_unit) pair: blockIdx.x = batch*nHidden + unit.
// Uses warp-shuffle reduction for the weighted sum over output units.
// atomicAdd accumulates across the batch dimension into the shared delta buffer.
// ---------------------------------------------------------------------------
__global__ void back_prop_kernel_batch(
    float* device_output,
    const float* inP,
    const float* m_hidden,
    const float* weights_2,
    const float* o_errG,
    int nInput, int nHidden, int nOutput, float l_R, int batchSize)
{
    int linearThreadIndex = threadIdx.x;
    int unit  = blockIdx.x % nHidden;
    int batch = blockIdx.x / nHidden;

    float ws = 0.f;
    if (linearThreadIndex < nOutput)
        ws = weights_2[unit * nOutput + linearThreadIndex]
           * o_errG[batch * (nOutput + 1) + linearThreadIndex];

    for (int mask = 16; mask > 0; mask >>= 1)
        ws += __shfl_xor_sync(0xffffffff, ws, mask);

    float weightedSum = ws;

    __syncthreads();

    float h = m_hidden[batch * (nHidden + 1) + unit];
    float delta = l_R * h * (1.f - h) * weightedSum;

    if (linearThreadIndex < nInput) {
        float grad = inP[batch * (nInput + 1) + linearThreadIndex] * delta;
        atomicAdd(&device_output[linearThreadIndex * nHidden + unit], grad);
    }
}

// ---------------------------------------------------------------------------
// GPU weight update: weights += deltas, then zero deltas
// ---------------------------------------------------------------------------
__global__ void weight_update_kernel(float* weights, float* deltas, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        weights[i] += deltas[i];
        deltas[i] = 0.f;
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
nnTrain::nnTrain(neuralNetwork* nn)
    : NN(nn),
      eP(0),
      l_R(DEFAULT_LR),
      max_eP(DEFAULT_EPOCHS),
      d_acc(DEFAULT_ACCUR),
      u_B(true),
      train_Acc(0), val_Acc(0), gen_Acc(0),
      l_E(false), logR(1), lastLog(-1)
{
    // ---- host delta matrices ----
    d_Inp    = new float*[NN->nInput + 1];
    d_Inp[0] = new float[(NN->nInput + 1) * NN->nHidden]();
    for (int i = 1; i <= NN->nInput; i++)
        d_Inp[i] = d_Inp[i - 1] + NN->nHidden;

    d_Out = new float*[NN->nHidden + 1];
    for (int i = 0; i <= NN->nHidden; i++) {
        d_Out[i] = new float[NN->nOutput]();
    }

    // ---- host error gradient arrays ----
    h_errG = new float[NN->batchSize * (NN->nHidden + 1)]();
    o_errG = new float[NN->batchSize * (NN->nOutput + 1)]();

    // ---- device buffers ----
    CUDA_CHECK(cudaMalloc(&d_dInp,    sizeof(float) * (NN->nInput + 1)  * NN->nHidden));
    CUDA_CHECK(cudaMalloc(&d_dOut,    sizeof(float) * (NN->nHidden + 1) * NN->nOutput));
    CUDA_CHECK(cudaMalloc(&d_inP,     sizeof(float) * NN->batchSize * (NN->nInput  + 1)));
    CUDA_CHECK(cudaMalloc(&d_mHidden, sizeof(float) * NN->batchSize * (NN->nHidden + 1)));
    CUDA_CHECK(cudaMalloc(&d_w2,      sizeof(float) * (NN->nHidden + 1) * NN->nOutput));
    CUDA_CHECK(cudaMalloc(&d_oErrG,   sizeof(float) * NN->batchSize * (NN->nOutput + 1)));

    CUDA_CHECK(cudaMemset(d_dInp, 0, sizeof(float) * (NN->nInput + 1)  * NN->nHidden));
    CUDA_CHECK(cudaMemset(d_dOut, 0, sizeof(float) * (NN->nHidden + 1) * NN->nOutput));
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
nnTrain::~nnTrain()
{
    delete[] d_Inp[0];   // only [0] owns the flat allocation
    delete[] d_Inp;

    for (int i = 0; i <= NN->nHidden; i++)
        delete[] d_Out[i];
    delete[] d_Out;

    delete[] h_errG;
    delete[] o_errG;

    cudaFree(d_dInp);
    cudaFree(d_dOut);
    cudaFree(d_inP);
    cudaFree(d_mHidden);
    cudaFree(d_w2);
    cudaFree(d_oErrG);
}

void nnTrain::setTrain(double learningRate, bool useBatch)
{
    l_R = (float)learningRate;
    u_B = useBatch;
}

void nnTrain::setStop(int maxEpochs, double desiredAccuracy)
{
    max_eP = maxEpochs;
    d_acc  = (float)desiredAccuracy;
}

void nnTrain::e_Log(const char* filename, int resolution)
{
    if (!logFile.is_open()) {
        logFile.open(filename, ios::out);
        if (logFile.is_open()) {
            logFile << "Epoch,TrainingAccuracy,GeneralizationAccuracy" << endl;
            l_E    = true;
            logR   = resolution;
            lastLog = -resolution;
        }
    }
}

// ---------------------------------------------------------------------------
// Compute output-layer error gradient for one sample at batch index batchIdx.
// o_errG[batchIdx * (nOutput+1) + k] = o*(1-o)*(d-o)
// ---------------------------------------------------------------------------
inline float nnTrain::get_oerrG(float dVal, float oVal)
{
    return oVal * (1.f - oVal) * (dVal - oVal);
}

void nnTrain::computeOutputErrorGrads(float* desiredOutputs, int batchIdx)
{
    int base = batchIdx * (NN->nOutput + 1);
    int oBase = batchIdx * NN->nOutput;
    for (int k = 0; k < NN->nOutput; k++)
        o_errG[base + k] = get_oerrG(desiredOutputs[k], NN->outputNeurons[oBase + k]);
}

// ---------------------------------------------------------------------------
// Main training loop
// ---------------------------------------------------------------------------
void nnTrain::netTrain(trainingDataSet* trainSet)
{
    cout << "\n Training Starts:\n----\n"
         << " LR: " << l_R
         << "  MaxEpochs: " << max_eP
         << "  BatchMode: " << u_B << "\n"
         << " Inputs: " << NN->nInput
         << "  Hidden: " << NN->nHidden
         << "  Outputs: " << NN->nOutput
         << "\n----\n\n";

    eP      = 0;
    lastLog = -logR;

    while ((train_Acc < d_acc || gen_Acc < d_acc) && eP < max_eP) {
        double prevT = train_Acc;
        double prevG = gen_Acc;

        r_TrainEP(trainSet->trainingSet, (int)eP);

        gen_Acc = (float)NN->getSetAccuracy(trainSet->generalizationSet);

        if (l_E && logFile.is_open() && (eP - lastLog == logR)) {
            logFile << eP << "," << train_Acc << "," << gen_Acc << "\n";
            lastLog = (int)eP;
        }

        if (ceil(prevT) != ceil(train_Acc) || ceil(prevG) != ceil(gen_Acc))
            cout << "Epoch: " << eP
                 << "  Train: " << train_Acc << "%"
                 << "  Gen: "   << gen_Acc   << "%\n";

        eP++;
    }

    val_Acc = (float)NN->getSetAccuracy(trainSet->validationSet);

    if (logFile.is_open()) {
        logFile << eP << "," << train_Acc << "," << gen_Acc << "\n\n";
        logFile << "Training Complete — Epochs: " << eP
                << "  Validation Accuracy: " << val_Acc << "\n";
    }

    cout << "\nTraining Complete — Elapsed Epochs: " << eP << "\n"
         << " Validation Accuracy: " << val_Acc << "%\n\n";
}

// ---------------------------------------------------------------------------
// Run one training epoch (single-sample or batch mode)
// ---------------------------------------------------------------------------
void nnTrain::r_TrainEP(std::vector<dataEntry*>& trainingSet, int epoch)
{
    double startIter = CycleTimer::currentSeconds();
    double incorrectPatterns = 0;

    // Per-epoch timing accumulators (printed once at end of epoch)
    double totalForward = 0, totalBack = 0;

    vector<float*> largePattern;
    vector<float*> largeTarget;
    largePattern.reserve(NN->batchSize);
    largeTarget.reserve(NN->batchSize);

    for (int tp = 0; tp < (int)trainingSet.size(); tp++) {
        largePattern.push_back(trainingSet[tp]->pattern);
        largeTarget.push_back(trainingSet[tp]->target);

        bool flushBatch = u_B &&
            ((tp == (int)trainingSet.size() - 1) ||
             ((int)largePattern.size() == NN->batchSize));

        if (flushBatch) {
            double t0 = CycleTimer::currentSeconds();
            NN->feedForwardBatch(largePattern);
            double t1 = CycleTimer::currentSeconds();
            backp_B(largeTarget);
            double t2 = CycleTimer::currentSeconds();
            w_Update();
            totalForward += t1 - t0;
            totalBack    += t2 - t1;

            // Accuracy check on last sample in batch
            int B = (int)largePattern.size();
            for (int b = 0; b < B; b++) {
                int predicted = (int)distance(NN->outputNeurons + b * NN->nOutput,
                    max_element(NN->outputNeurons + b * NN->nOutput,
                                NN->outputNeurons + b * NN->nOutput + NN->nOutput));
                int expected = (int)distance(largeTarget[b],
                    max_element(largeTarget[b], largeTarget[b] + NN->nOutput));
                if (predicted != expected) incorrectPatterns++;
            }

            largePattern.clear();
            largeTarget.clear();

        } else if (!u_B) {
            double t0 = CycleTimer::currentSeconds();
            NN->feedForward(trainingSet[tp]->pattern);
            double t1 = CycleTimer::currentSeconds();
            backp(trainingSet[tp]->target);
            double t2 = CycleTimer::currentSeconds();
            totalForward += t1 - t0;
            totalBack    += t2 - t1;

            int predicted = (int)distance(NN->outputNeurons,
                max_element(NN->outputNeurons, NN->outputNeurons + NN->nOutput));
            int expected = (int)distance(trainingSet[tp]->target,
                max_element(trainingSet[tp]->target,
                            trainingSet[tp]->target + NN->nOutput));
            if (predicted != expected) incorrectPatterns++;
        }
    }

    train_Acc = 100.f - (float)(incorrectPatterns / trainingSet.size() * 100.0);

    double timeIter = CycleTimer::currentSeconds() - startIter;
    printf("Epoch %d — Total: %.4fs  Forward: %.4fs  Backprop: %.4fs\n",
           epoch, timeIter, totalForward, totalBack);
}

// ---------------------------------------------------------------------------
// Batch backpropagation
// ---------------------------------------------------------------------------
void nnTrain::backp_B(std::vector<float*>& desiredOutputsVector)
{
    int B = (int)desiredOutputsVector.size();

    // 1. Compute output error gradients on CPU
    for (int b = 0; b < B; b++)
        computeOutputErrorGrads(desiredOutputsVector[b], b);

    // 2. Accumulate hidden-output weight deltas on CPU
    for (int b = 0; b < B; b++) {
        for (int j = 0; j <= NN->nHidden; j++) {
            float h = NN->hiddenNeurons[b * (NN->nHidden + 1) + j];
            for (int k = 0; k < NN->nOutput; k++)
                d_Out[j][k] += l_R * h * o_errG[b * (NN->nOutput + 1) + k];
        }
    }

    // 3. Upload data for input-hidden gradient kernel
    CUDA_CHECK(cudaMemcpy(d_inP, NN->inputNeurons,
        sizeof(float) * B * (NN->nInput + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mHidden, NN->hiddenNeurons,
        sizeof(float) * B * (NN->nHidden + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, NN->wHiddenOutput[0],
        sizeof(float) * (NN->nHidden + 1) * NN->nOutput, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oErrG, o_errG,
        sizeof(float) * B * (NN->nOutput + 1), cudaMemcpyHostToDevice));

    // 4. Launch gradient kernel: one block per (batch x hidden_unit)
    dim3 gridDim(NN->nHidden * B);
    dim3 blockDim(max(NN->nInput + 1, 32));  // at least one warp

    back_prop_kernel_batch<<<gridDim, blockDim>>>(
        d_dInp, d_inP, d_mHidden, d_w2, d_oErrG,
        NN->nInput + 1, NN->nHidden, NN->nOutput, l_R, B);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Copy input-hidden deltas back to host
    CUDA_CHECK(cudaMemcpy(d_Inp[0], d_dInp,
        sizeof(float) * (NN->nInput + 1) * NN->nHidden, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Single-sample backpropagation
// ---------------------------------------------------------------------------
void nnTrain::backp(float* desiredOutputs)
{
    // 1. Output error gradients
    computeOutputErrorGrads(desiredOutputs, 0);

    // 2. Hidden-output weight deltas
    for (int j = 0; j <= NN->nHidden; j++) {
        float h = NN->hiddenNeurons[j];
        for (int k = 0; k < NN->nOutput; k++)
            d_Out[j][k] = l_R * h * o_errG[k];
    }

    // 3. Upload data for input-hidden gradient kernel
    CUDA_CHECK(cudaMemcpy(d_inP, NN->inputNeurons,
        sizeof(float) * (NN->nInput + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mHidden, NN->hiddenNeurons,
        sizeof(float) * (NN->nHidden + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, NN->wHiddenOutput[0],
        sizeof(float) * (NN->nHidden + 1) * NN->nOutput, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oErrG, o_errG,
        sizeof(float) * (NN->nOutput + 1), cudaMemcpyHostToDevice));

    // 4. One block per hidden unit
    back_prop_kernel<<<NN->nHidden, max(NN->nInput + 1, 32)>>>(
        d_dInp, d_inP, d_mHidden, d_w2, d_oErrG,
        NN->nInput + 1, NN->nHidden, NN->nOutput, l_R);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_Inp[0], d_dInp,
        sizeof(float) * (NN->nInput + 1) * NN->nHidden, cudaMemcpyDeviceToHost));

    w_Update();
}

// ---------------------------------------------------------------------------
// Apply accumulated weight deltas and reset them for the next step
// ---------------------------------------------------------------------------
void nnTrain::w_Update()
{
    // Input-hidden weights
    for (int i = 0; i <= NN->nInput; i++) {
        for (int j = 0; j < NN->nHidden; j++) {
            NN->wInputHidden[i][j] += d_Inp[i][j];
            d_Inp[i][j] = 0.f;
        }
    }

    // Hidden-output weights
    for (int j = 0; j <= NN->nHidden; j++) {
        for (int k = 0; k < NN->nOutput; k++) {
            NN->wHiddenOutput[j][k] += d_Out[j][k];
            d_Out[j][k] = 0.f;
        }
    }

    // Mark weights dirty so the next forward pass re-uploads them
    NN->weightsDirty = true;

    // Reset device delta buffer for input-hidden
    CUDA_CHECK(cudaMemset(d_dInp, 0,
        sizeof(float) * (NN->nInput + 1) * NN->nHidden));
}
