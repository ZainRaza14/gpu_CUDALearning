
#include <iostream>

#include <vector>

#include <fstream>

#include <math.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <algorithm>

#include <omp.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <driver_functions.h>

#include <curand.h>

#include <curand_kernel.h>

#include <cublas_v2.h>

#include "CycleTimer.h"

#define BLOCKSIZE  1024

#define TILE_WIDTH 32

#define SCAN_BLOCK_DIM  BLOCKSIZE

#include "exclusiveScan.cu_inl"


#include "neuralNetwork.h"

using namespace std;


__global__ void forward_prop_kernel(float *device_output, float *input, float *weights, int num_first, int num_second) 
{

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y; 

	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	float Pvalue = 0; 

	__shared__ float prefixSumOutput[BLOCKSIZE];
    
    __shared__ float prefixSumScratch[2 * BLOCKSIZE]; 

	for (int k = 0; k < num_second; ++k) 
		Pvalue += input[Row*num_second+k] * weights[k*num_second+Col];

	device_output[Row*num_second+Col] = Pvalue; 

	__syncthreads();

 	sharedMemExclusiveScan(threadIdx.x, device_output, prefixSumOutput, 
                            prefixSumScratch, BLOCKSIZE);

	__syncthreads();

    if (threadIdx.x == 0 && blockIdx.x < num_second) 
    {
    	device_output[blockIdx.x] = 1/(1+exp(-1*prefixSumOutput[num_first]));
    	
    }

}



__global__ void
forward_prop_kernel_batch(float *device_output, float *input, float *weights, int num_first, int num_second, int batchSize) 
{
	int linearThreadIndex = threadIdx.x;
	
	int unit = blockIdx.x%num_second;
	
	int batch = blockIdx.x/num_second;

    __shared__ float prefixSumInput[BLOCKSIZE];
    
    __shared__ float prefixSumOutput[BLOCKSIZE];
    
    __shared__ float prefixSumScratch[2 * BLOCKSIZE];

    if (linearThreadIndex < num_first) 
    {
    	prefixSumInput[linearThreadIndex] = input[batch*linearThreadIndex] * weights[linearThreadIndex*num_second + unit];
    }

    __syncthreads();

    sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, 
                            prefixSumScratch, BLOCKSIZE);

    __syncthreads();

    if (linearThreadIndex == 0 && unit < num_second) 
    {
    	device_output[batch*unit] = 1/(1+exp(-1*prefixSumOutput[num_first]));
    }
}

neuralNetwork::neuralNetwork(int nI, int nH, int nO, int bS) : nInput(nI), nHidden(nH), nOutput(nO), batchSize(bS)
{				
	
	inputNeurons = new( float[batchSize*(nInput + 1)] );
        
        for (int b= 0; b<batchSize; b++) 
        {
            for (int i=0; i<nInput+1; i++) 
            {
                if (i==nInput) 
                {
                    inputNeurons[(b+1)*(nInput)] = -1;
                }
                
                else 
                {
                    inputNeurons[b*(nInput+1) + i] = 0;
                } 
            
            }
        }

	hiddenNeurons = new( float[batchSize*(nHidden + 1)] );
        
        for (int b=0; b<batchSize; b++) 
        {
            for (int i=0; i<nHidden+1; i++) 
            {
                if (i==nHidden) 
                {
                    hiddenNeurons[(b+1)*(nHidden)] = -1;
                }
                
                else 
                {
                    hiddenNeurons[b*(nHidden+1) + i] = 0; 
                }
            }
        }

	outputNeurons = new( float[batchSize*(nOutput + 1)] );
	
	for ( int i=0; i < batchSize*(nOutput+1); i++ ) 
	{
		outputNeurons[i] = 0;
	}

	
	wInputHidden = new( float*[nInput + 1] );
	
	wInputHidden[0] = new (float[(nInput + 1)*nHidden]);
	
	for ( int i=1; i <= nInput; i++ ) 
	{
		wInputHidden[i] = wInputHidden[i-1] + nHidden;
	}
	
	for ( int i=0; i <= nInput; i++ ) 
	{
		for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;		
	}

	wHiddenOutput = new( float*[nHidden + 1] );
	
	wHiddenOutput[0] = new (float[(nHidden + 1)*nOutput]);
	
	for ( int i=1; i <= nHidden; i++ ) 
	{
		wHiddenOutput[i] = wHiddenOutput[i-1] + nOutput;
	}
	
	for ( int i=0; i <= nHidden; i++ ) 
	{
		for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;		
	}
	
	
	initializeWeights();		
}


neuralNetwork::~neuralNetwork()
{
	
	delete[] inputNeurons;
	
	delete[] hiddenNeurons;
	
	delete[] outputNeurons;

	for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
	
	delete[] wInputHidden;

	for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
	
	delete[] wHiddenOutput;

	
	cudaFree(device_output1);
	
	cudaFree(input);
	
	cudaFree(w1);

	cudaFree(device_output2);
	
	cudaFree(hidden);
	
	cudaFree(w2);

}

bool neuralNetwork::saveWeights(char* filename)
{
	
	fstream outputFile;
	
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
			}
		}

	
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		
		outputFile.close();
		
		return true;
	}
	
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}


double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		
		feedForward( set[tp]->pattern );

		int predicted = distance(outputNeurons, max_element(outputNeurons, outputNeurons + nOutput));
		i
		int expected = distance(set[tp]->target, max_element(set[tp]->target, set[tp]->target + nOutput));
		
		if (predicted != expected) incorrectResults++;	
		
	}
	

	return 100 - (incorrectResults/set.size() * 100);
}



void neuralNetwork::initializeWeights()
{
	double startTime = CycleTimer::currentSeconds();

	cudaMalloc(&device_output1, sizeof(float) * batchSize*nHidden);
    
    cudaMalloc(&input, sizeof(float) * batchSize*(nInput+1));
    
    cudaMalloc(&w1, sizeof(float) * (nInput+1)*nHidden);

    cudaMalloc(&device_output2, sizeof(float) * batchSize*nOutput);
    
    cudaMalloc(&hidden, sizeof(float) * batchSize*(nHidden+1));
    
    cudaMalloc(&w2, sizeof(float) * (nHidden+1)*nOutput);
    
	for(int i = 0; i <= nInput; i++)
	{		
		for(int j = 0; j < nHidden; j++) 
		{
			
			wInputHidden[i][j] = ( (( (float)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}
	
	
	for(int i = 0; i <= nHidden; i++)
	{		
		for(int j = 0; j < nOutput; j++) 
		{
			wHiddenOutput[i][j] = ( (( (float)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}
	
	double endTime = CycleTimer::currentSeconds();
    
    double overallDuration = endTime - startTime;

    printf("Time Taken Seq:%f\n", overallDuration);
}

inline float neuralNetwork::activationFunction( float x )
{
	return 1/(1+exp(-x));
}	

void neuralNetwork::feedForwardBatch(vector<float*> patternVector) 
{

	for (int b = 0; b<batchSize; b++) 
	{
	    for(int i = 0; i < nInput+1; i++) 
	    { 
                if (i!=nInput) 
                {
                    inputNeurons[b*(nInput+1) + i] = patternVector[b][i];
                }
        
        }
	}

	dim3 blockDim(1024,1);
    
    dim3 gridDim(nHidden*batchSize);
    
    cudaMemcpy(input, inputNeurons, sizeof(float) * batchSize*(nInput+1), cudaMemcpyHostToDevice);
    
    cudaMemcpy(w1, wInputHidden[0], (nInput+1)*nHidden*sizeof(float), cudaMemcpyHostToDevice);
    
    forward_prop_kernel_batch<<<gridDim, blockDim>>>(device_output1, input, w1, nInput+1, nHidden, batchSize);
    
    cudaThreadSynchronize();
    
    cudaMemcpy(hiddenNeurons, device_output1, batchSize*nHidden*sizeof(float), cudaMemcpyDeviceToHost);


    dim3 gridDim2(nOutput*batchSize);
	
	cudaMemcpy(hidden, hiddenNeurons, sizeof(float) * batchSize*(nHidden+1), cudaMemcpyHostToDevice);
	
	cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(float), cudaMemcpyHostToDevice);
	
	forward_prop_kernel_batch<<<gridDim2, blockDim>>>(device_output2, hidden, w2, nHidden+1, nOutput,batchSize);
	
	cudaThreadSynchronize();
	
	cudaMemcpy(outputNeurons, device_output2, batchSize*nOutput*sizeof(float), cudaMemcpyDeviceToHost);


}

void neuralNetwork::feedForward(float* pattern)
{
	
	for(int i = 0; i < nInput; i++) 
	{
		inputNeurons[i] = pattern[i];
	}

	double startTime = CycleTimer::currentSeconds();
	
	
	dim3 blockDim(1024, 1);
    
    dim3 gridDim(128);
	
    cudaMemcpy(input, inputNeurons, sizeof(float) * (nInput+1), cudaMemcpyHostToDevice);
   
    
    cudaMemcpy(w1, wInputHidden[0], (nInput+1)*nHidden*sizeof(float), cudaMemcpyHostToDevice);
    

	forward_prop_kernel<<<gridDim, blockDim>>>(device_output1, input, w1, nInput+1, nHidden);

	cudaThreadSynchronize();
	

    cudaMemcpy(hiddenNeurons, device_output1, nHidden*sizeof(float), cudaMemcpyDeviceToHost);
	

    dim3 gridDim2(10);
	
    cudaMemcpy(hidden, hiddenNeurons, sizeof(float) * (nHidden+1), cudaMemcpyHostToDevice);
   
    
    cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(float), cudaMemcpyHostToDevice);
    

	forward_prop_kernel<<<gridDim2, blockDim>>>(device_output2, hidden, w2, nHidden+1, nOutput);

	cudaThreadSynchronize();


	cudaMemcpy(outputNeurons, device_output2, nOutput*sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(outputNeurons, device_output2, nOutput*sizeof(float), cudaMemcpyDeviceToHost);

	double endTime4 = CycleTimer::currentSeconds();

	double time = endTime4 - startTime;
	
}

void neuralNetwork::printCudaInfo()
{

    int deviceCount = 0;
    
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        
        cudaGetDeviceProperties(&deviceProps, i);
        
        printf("Device %d: %s\n", i, deviceProps.name);
        
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    
    printf("---------------------------------------------------------\n"); 
}

