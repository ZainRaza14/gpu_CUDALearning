#ifndef NNTRAIN_H
#define NNTRAIN_H

#include <fstream>
#include <vector>
#include "neuralNetwork.h"

#define DEFAULT_LR      0.001f
#define DEFAULT_EPOCHS  1500
#define DEFAULT_ACCUR   90.0f

class nnTrain
{
private:
    neuralNetwork* NN;

    float  l_R;       // learning rate
    long   eP;        // current epoch
    long   max_eP;    // max epochs
    float  d_acc;     // desired accuracy threshold

    // Host-side weight-delta matrices (flat, row-major)
    float** d_Inp;    // (nInput+1)  x nHidden  — input-hidden deltas
    float** d_Out;    // (nHidden+1) x nOutput  — hidden-output deltas

    // Host-side error gradient arrays
    float* h_errG;    // batchSize x (nHidden+1)
    float* o_errG;    // batchSize x (nOutput+1)

    // Device buffers for backward pass
    float* d_dInp;    // device copy of d_Inp  : (nInput+1) x nHidden
    float* d_dOut;    // device copy of d_Out  : (nHidden+1) x nOutput
    float* d_inP;     // device input neurons  : batchSize x (nInput+1)
    float* d_mHidden; // device hidden neurons : batchSize x (nHidden+1)
    float* d_w2;      // device copy of w2     : (nHidden+1) x nOutput
    float* d_oErrG;   // device output error grads : batchSize x (nOutput+1)

    float train_Acc;
    float val_Acc;
    float gen_Acc;

    bool u_B;   // use batch learning
    bool l_E;   // logging enabled

    std::fstream logFile;
    int logR;
    int lastLog;

public:
    nnTrain(neuralNetwork* untrainedNetwork);
    ~nnTrain();

    void setTrain(double learningRate, bool useBatch);
    void setStop(int maxEpochs, double desiredAccuracy);
    void useBatchLearning(bool flag) { u_B = flag; }
    void e_Log(const char* filename, int resolution = 1);
    void netTrain(trainingDataSet* trainSet);

private:
    inline float get_oerrG(float dVal, float oVal);
    void computeOutputErrorGrads(float* desiredOutputs, int batchIdx = 0);
    void r_TrainEP(std::vector<dataEntry*>& trainingSet, int epoch);
    void backp_B(std::vector<float*>& desiredOutputsVec);
    void backp(float* desiredOutputs);
    void w_Update();
};

#endif
