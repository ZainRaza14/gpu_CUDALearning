/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* This software and the information contained herein is PROPRIETARY and 
* CONFIDENTIAL to NVIDIA and is being provided under the terms and 
* conditions of a Non-Disclosure Agreement.  Any reproduction or 
* disclosure to any third party without the express written consent of 
* NVIDIA is prohibited.     
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


// includes, kernels
#include "vector_reduction_kernel.cu"
#include "vector_reduction_gold.cpp"

// For simplicity, just to get the idea in this MP, we're fixing the problem size to 512 elements.
#define NUM_ELEMENTS 512

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);

float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int myerrorM = 0;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        /* --Make your own--
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        */

        case 1:
            
            myerrorM = ReadFile(h_data, argv[1]);

            if(myerrorM != 1)
            {
                printf("Error reading input ! \n");
                exit(1);
            }


        break;

        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);
    
    // **===-------- Modify the body of this function -----------===**
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

/*--make your own--
int ReadFile(float* M, char* file_name)
{
	unsigned int elements_read = NUM_ELEMENTS;
	if (cutReadFilef(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;
}
*/

int ReadFile(float* M, char* file_name)
{
    printf("%s", "The in");

    FILE *myFile;

    myFile = fopen(file_name , "r");

    //read file into array
    //int numberArray[512];

    int i;

    if (myFile == NULL)
    {
        printf("Error Reading File\n");
        exit (0);
    }

    for (i = 0; i < 512; i++){
        fscanf(myFile, "%f,", &M[i] );
    }

    ////for (i = 0; i < 16; i++){
        //printf("Number is: %d\n\n", numberArray[i]);
    //}

    fclose(myFile);

    return 1;

}


// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
    float *d_data;

    float myResult;

    // Allocating memory for device side
    
    cudaMalloc((void**)&d_data, sizeof(float)*num_elements);

    // Copy data from host to device

    cudaMemcpy(d_data, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice );

    int num_ofThreads = num_elements;

    //int num_ofBlocks = 16;

    // Kernel is being invoked

    reduction<<< 1 , num_ofThreads >>>(d_data , num_elements);

    // Copy data back from device to host 
    
    cudaMemcpy(&myResult, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Freeing data
    cudaFree(d_data);

    cudaDeviceReset();
    
    // Return the final result 
    return myResult;

}
     
