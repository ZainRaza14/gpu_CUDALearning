//standard includes
#include <iostream>

#include <fstream>
#include <math.h>

#include <algorithm>

#include <omp.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <driver_functions.h>

#include <cublas_v2.h>

#include "nnTrain.h"

#include "CycleTimer.h"

using namespace std;

__global__ void
back_prop_kernel(float *device_output, float *inP, float *m_hidden, float* weights_2, float* o_errG, int nInput, int nHidden, int nOutput,  float l_R) 
{
	int linearThreadIndex = threadIdx.x;
	
	int unit = blockIdx.x;
   
    __shared__ float weightedSum[1];
    
    if (linearThreadIndex==0) 
    {
        for (int i=0; i<nOutput; i++) 
        {
         
          weightedSum[0] += weights_2[unit*nOutput + i] * o_errG[i];
        
        }
    
    }

    __syncthreads();

    if (linearThreadIndex < nInput) 
    {
    	
        device_output[linearThreadIndex*nHidden + unit] = l_R * inP[linearThreadIndex] * m_hidden[unit]*(1 - m_hidden[unit]) * weightedSum[0];
    
    }

}


__global__ void 
back_prop_kernel_batch(float *device_output, float *inP, float *m_hidden, float* weights_2, float* o_errG, int nInput, int nHidden, int nOutput, float l_R, int batchSize) 
{
    int linearThreadIndex = threadIdx.x;
    
    int unit = blockIdx.x%nHidden;
    
    int batch = blockIdx.x/nHidden; 
    
    __shared__ float weightedSum[1];
    
    float temp = 0.0;
    
    if (linearThreadIndex ==0 && unit<nHidden) 
    {
        for (int i=0; i<nOutput; i++) 
        { 
        
            weightedSum[0] += weights_2[unit*nOutput + i] * o_errG[batch*(nOutput+1) +i];
        
        }
    
    }
    
    __syncthreads();
   
    if (linearThreadIndex < nInput) 
    {
        temp = l_R * inP[batch*(nInput+1) + linearThreadIndex] * m_hidden[batch*(nHidden+1) + unit]*(1 - m_hidden[batch*(nHidden+1) + unit]) * weightedSum[0];
        
        atomicAdd(&device_output[linearThreadIndex*nHidden + unit], temp);
    
    } 


}


nnTrain::nnTrain( neuralNetwork *nn )	:	NN(nn),
																	eP(0),
																	l_R(lR),
																	max_eP(m_epchs),
																	d_acc(accur_d),																	
																	u_B(true),
																	train_Acc(0),
																	val_Acc(0),
																	gen_Acc(0)																	
{


	d_Inp = new( float*[NN->nInput + 1] );
    
    d_Inp[0] = new (float[((NN->nInput) + 1)*(NN->nHidden)]);
    
    for ( int i=1; i <= NN->nInput; i++ ) 
    {
	
		d_Inp[i] = d_Inp[i-1] + NN->nHidden;
	
	}

	for ( int i=0; i <= NN->nInput; i++ ) 
	{
	
		for ( int j=0; j < NN->nHidden; j++ ) d_Inp[i][j] = 0;		
	
	}



	d_Out = new( float*[NN->nHidden + 1] );
	
	for ( int i=0; i <= NN->nHidden; i++ ) 
	{
	
		d_Out[i] = new (float[NN->nOutput]);			
	
		for ( int j=0; j < NN->nOutput; j++ ) d_Out[i][j] = 0;		
	
	}

	
	h_errG = new( float[(NN->batchSize)*(NN->nHidden + 1)] );
	
	for (int b = 0; b<NN->batchSize; b++) 
	{
	    for(int i = 0; i < NN->nHidden+1; i++) 
	    { 
        
            h_errG[b*(NN->nHidden+1) + i] = 0;
        
        }
	
	}
	
	o_errG = new( float[(NN->batchSize)*(NN->nOutput + 1)] );
	
	for (int b = 0; b<NN->batchSize; b++) 
	{
	
	    for(int i = 0; i < NN->nOutput+1; i++) 
	    { 
        
            o_errG[b*(NN->nOutput+1) + i] = 0;
        
        }
	
	}

    cudaMalloc(&d_Out1, sizeof(float) * (NN->batchSize)*((NN->nInput)+1)*(NN->nHidden));
    
    cudaMalloc(&inP, sizeof(float) * (NN->batchSize)*((NN->nInput)+1));
    
    cudaMalloc(&m_hidden, sizeof(float) * (NN->batchSize)*(((NN->nHidden) +1)));
    
    cudaMalloc(&weights_2, sizeof(float) * ((NN->nHidden)+1)*(NN->nOutput));
    
    cudaMalloc(&o_errG1, sizeof(float)*((NN->nOutput) +1));
    

}


nnTrain::~nnTrain()
{
	
	for (int i=0; i <= NN->nInput; i++) delete[] d_Inp[i];
	
	delete[] d_Inp;

	for (int j=0; j <= NN->nHidden; j++) delete[] d_Out[j];
	
	delete[] d_Out;

	cudaFree(d_Out1);
	
	cudaFree(inP);
	
	cudaFree(m_hidden);
	
	cudaFree(weights_2);
	
	cudaFree(o_errG1);
	
}


void nnTrain::setTrain( double learningRate, bool m_batch )
{
	
	l_R = learningRate;
	
	u_B = m_batch;

}

void nnTrain::setStop( int mEP, double dAcc )
{

	max_eP = mEP;

	d_acc = dAcc;	

}

void nnTrain::e_Log(const char* filename, int resolution = 1)
{
	
	if ( ! logFile.is_open() )
	{
		
		logFile.open(filename, ios::out);

		if ( logFile.is_open() )
		{
			
			logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
			
			
			l_E = true;
			
			
			logR = resolution;
			
			lastLog = -resolution;
		
		}
	
	}

}


inline float nnTrain::get_oerrG( float dVal, float oVal)
{
	
	return oVal * ( 1 - oVal ) * ( dVal - oVal );

}


float nnTrain::get_herrG( int j )

{
	
	float weightedSum = 0;
	
	for( int k = 0; k < NN->nOutput; k++ ) 
	{
	
		weightedSum += NN->wHiddenOutput[j][k] * o_errG[k];
	
	}

	
	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;

}

float nnTrain::get_herrGB( int j, int b )
{
	
	float weightedSum = 0;
	
	for( int k = 0; k < NN->nOutput; k++ ) 
	{
	
		weightedSum += NN->wHiddenOutput[j][k] * o_errG[b*k];
	
	}

	
	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;

}

void nnTrain::netTrain( trainingDataSet* trainSet )
{
	cout	<< endl << " Training Starts: " << endl
			<< "----" << endl
			<< " Learning Rate: " << l_R << ", Maximum number of Epochs: " << max_eP << ", Use Batch or Not: " << u_B << endl
			<< " " << NN->nInput << " Number of Input Neurons, " << NN->nHidden << " Number of Hidden Neurons, " << NN->nOutput << "Number of Output Neurons" << endl
			<< "----" << endl << endl;

	
	eP = 0;
	lastLog = -logR;
		
	
	while (	( train_Acc < d_acc || gen_Acc < d_acc ) && epoch < max_eP )				
	{			
		
		double previousTAccuracy = train_Acc;
		
		double previousGAccuracy = gen_Acc;

		
		r_TrainEP( trainSet->trainingSet , eP);

		
		gen_Acc = NN->getSetAccuracy( trainSet->generalizationSet );

	
		if ( l_E && logFile.is_open() && ( eP - lastLog == logR ) ) 
		{
		
			logFile << eP << "," << train_Acc << "," << gen_Acc << endl;
		
			lastLog = eP;
		
		}
		
		
		if ( ceil(previousTAccuracy) != ceil(train_Acc) || ceil(previousGAccuracy) != ceil(gen_Acc) ) 
		{	
		
			cout << "Epoch :" << eP;
		
			cout << " TSet Acc:" << train_Acc << "%" ;
		
			cout << " GSet Acc:" << gen_Acc << "%" << endl;
		
		}
		
		
		eP++;

	}

	val_Acc = NN->getSetAccuracy(trainSet->validationSet);

	logFile << epoch << "," << train_Acc << "," << gen_Acc << endl << endl;

	logFile << "Training Complete!!! - > Elapsed Epochs: " << eP << " Validation Set Accuracy: " << val_Acc << endl;
			
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << eP << endl;

	cout << " Validation Set Accuracy: " << val_Acc << endl << endl;

}

void nnTrain::r_TrainEP( vector<dataEntry*> trainingSet , int eP)

{
	double startIter = CycleTimer::currentSeconds();
	
	double incorrectPatterns = 0;
	
	vector<float*>largePattern;
	
	vector<float*>largeTarget; 

	double startForward;
	
	double endForward;

	double startBack;
	
	double endBack;



	for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
	{
		largePattern.push_back(trainingSet[tp]->pattern);
		
		largeTarget.push_back(trainingSet[tp]->target);
	
		if (u_B && ((tp == (int) trainingSet.size()-1) || (largePattern.size() == NN->batchSize))) 
		{
			startForward = CycleTimer::currentSeconds();
			
			NN->feedForwardBatch( largePattern );
			
			endForward = CycleTimer::currentSeconds();

			startBack = CycleTimer::currentSeconds();
			
			backp_B( largeTarget );
			
			endBack = CycleTimer::currentSeconds();

			w_Update();
			
			largePattern.clear();
			
			largeTarget.clear();
		
		} 

		else 
		{
			startForward = CycleTimer::currentSeconds();
			
			NN->feedForward( trainingSet[tp]->pattern );
			
			endForward = CycleTimer::currentSeconds();

			startBack = CycleTimer::currentSeconds();
			
			backp( trainingSet[tp]->target );
			
			endBack = CycleTimer::currentSeconds();
		
		}

		
	    double timeForward = endForward - startForward;
	    
	    double timeBack = endBack - startBack;
	    
	    double timeBoth = endBack - startForward;

	    if (eP) 
	    {
		 
		     printf("Forward: %f\n", timeForward);
		 
		     printf("Backprop: %f\n", timeBack);
		 
		     printf("Both: %f\n", timeBoth);
	    
	    }

	    
		int predicted = distance(NN->outputNeurons, max_element(NN->outputNeurons, NN->outputNeurons + NN->nOutput));
		
		int expected = distance(trainingSet[tp]->target, max_element(trainingSet[tp]->target, trainingSet[tp]->target + NN->nOutput));
		
		if (predicted != expected) incorrectPatterns++;
			
		
	}

	
	
	train_Acc = 100 - (incorrectPatterns/trainingSet.size() * 100);

	double endIter = CycleTimer::currentSeconds();
    
    double timeIter = endIter - startIter;

    printf("Iteration: %f\n", timeIter);


}

void nnTrain::backp_B(vector<float*> desiredOutputsVector) 

{


	double startCuda = CycleTimer::currentSeconds();
	
	dim3 blockDim(1024,1);
    
    dim3 gridDim(NN->nHidden);
    
    cudaMemcpy(inP, NN->inputNeurons, sizeof(float) * ((NN->nInput)+1) *(NN->batchSize), cudaMemcpyHostToDevice);
    
    cudaMemcpy(hidden, NN->hiddenNeurons, (NN->batchSize)*((NN->nHidden)+1)*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(weights_2, NN->wHiddenOutput[0], ((NN->nHidden)+1)*(NN->nOutput)*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(o_errG1, o_errG, sizeof(float) * (NN->batchSize)*((NN->nOutput)+1), cudaMemcpyHostToDevice);
    
    back_prop_kernel_batch<<<gridDim, blockDim>>>(d_Out1, inP, m_hidden, weights_2, o_errG1, (NN->nInput)+1, NN->nHidden, NN->nOutput, l_R, 8);
    
    cudaMemcpy(d_Inp[0], d_Out1, ((NN->nInput) +1)*(NN->nHidden)*sizeof(float), cudaMemcpyDeviceToHost);


    double endCuda = CycleTimer::currentSeconds();
    
    double timeCuda = endCuda - startCuda;
 
}

void nnTrain::backp( float* desiredOutputs )

{	
	
	dim3 blockDim(1024, 1);
    
    dim3 gridDim(128);
    
    cudaMemcpy(input, NN->inputNeurons, sizeof(float) * ((NN->nInput)+1), cudaMemcpyHostToDevice);
    
    cudaMemcpy(m_hidden, NN->hiddenNeurons, ((NN->nHidden)+1)*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(weights_2, NN->wHiddenOutput[0], ((NN->nHidden)+1)*(NN->nOutput)*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(o_errG1, o_errG, sizeof(float) * ((NN->nOutput)+1), cudaMemcpyHostToDevice);
    
    back_prop_kernel<<<gridDim, blockDim>>>(d_Out1, input, m_hidden, weights_2, o_errG1, (NN->nInput)+1, NN->nHidden, NN->nOutput, l_R);
    
    cudaMemcpy(d_Inp[0], d_Out1, (NN->batchSize)*(NN->nInput +1)*(NN->nHidden)*sizeof(float), cudaMemcpyDeviceToHost);
	
	if ( !u_B ) w_Update();

}

void nnTrain::w_Update()
{
	
	for (int i = 0; i <= NN->nInput; i++)
	{
		for (int j = 0; j < NN->nHidden; j++) 
		{
			
			NN->wInputHidden[i][j] += d_Inp[i][j];	
			
			
			if (u_B) d_Inp[i][j] = 0;				
		}
	}
	
	
	for (int j = 0; j <= NN->nHidden; j++)
	{
		for (int k = 0; k < NN->nOutput; k++) 
		{					
			
			NN->wHiddenOutput[j][k] += d_Out[j][k];
			
			
			if (u_B)d_Out[j][k] = 0;
		}
	}
}
