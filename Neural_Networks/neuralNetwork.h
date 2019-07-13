
#ifndef NNetwork

#define NNetwork

//sss

#include "dataReader.h"

class neuralNetworkTrainer;

class neuralNetwork
{
	
private:

	int nInput, nHidden, nOutput, batchSize;
	
	float* inputNeurons;
	
	float* hiddenNeurons;
	
	float* outputNeurons;

	float** wInputHidden;
	
	float** wHiddenOutput;

	float *device_output1;
	
	float *input;
	
	float *w1;

	float *device_output2;
	
	float *hidden;
	
	float *w2;

	
	friend neuralNetworkTrainer;
	
public:

	
	neuralNetwork(int numInput, int numHidden, int numOutput, int batchSize);
	
	~neuralNetwork();

	bool saveWeights(char* outputFilename);
	
	double getSetAccuracy( std::vector<dataEntry*>& set );

	void printCudaInfo();

private: 

	void initializeWeights();
	
	inline float activationFunction( float x );
	
	void feedForward( float* pattern );
	
	void feedForwardBatch(std::vector<float*> patternVector);
	
};

#endif
