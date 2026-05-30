#include <iostream>
#include <ctime>
#include <stdlib.h>
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"
#include "CycleTimer.h"

using namespace std;

int main()
{
    double timer_Start = CycleTimer::currentSeconds();

    srand((unsigned int)time(0));

    // Load MNIST CSV: 784 pixel inputs, 10 class outputs
    dataReader dR;
    if (!dR.loadDataFile("mnist_train.csv", 784, 10)) {
        cerr << "Failed to load dataset." << endl;
        return 1;
    }
    dR.setNumSets(1);

    // Network: 784 inputs, 128 hidden, 10 outputs, batch size 32
    neuralNetwork network(784, 128, 10, 32);
    network.printCudaInfo();

    nnTrain trainer(&network);
    trainer.setTrain(0.01, true);        // learning rate, use batch mode
    trainer.setStop(100, 90);            // max 100 epochs, stop at 90% accuracy
    trainer.e_Log("training_log.csv", 1);

    for (int i = 0; i < dR.getNumTrainingSets(); i++) {
        trainer.netTrain(dR.getTrainingDataSet());
    }

    double timer_End = CycleTimer::currentSeconds();
    double total_Time = timer_End - timer_Start;

    cout << "Total program time: " << total_Time << "s" << endl;

    return 0;
}
