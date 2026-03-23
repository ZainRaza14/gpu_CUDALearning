#include<iostream>
#include<ctime>
#include<stdlib.h>
#include"NN.h"
#include"NN_Train.h"
#include"CycleTimer.h"

using namespace std;

int main()
{
	double timer_Start = CycleTimer::currentSeconds();

	srand((unsigned int) time(0));

	dataReader dR;

	dR.loadDataFile("mnist_train.csv");

	dR.setNumSets(1);

	NN neuralNetwork(782, 128, 10, 1);

	neuralNetwork.printCudaInfo();

	NN_Train neuralNTraining(&neuralNetwork);

	neuralNTraining.setTrainingParameters(0.5, false);

	neuralNTraining.setStoppingConditions(100, 90);

	for(int i = 0; i < dR.getNumTrainingSets() ; i++)
	{
		neuralNTraining.trainNetwork(dR.getTrainingDataSet());
	}

	double timer_End = CycleTimer::currentSeconds();

	double total_Time = timer_End - timer_Start;

	cout<< "Final Total Time of the Program: " << total_Time << endl;

	return 0;
}