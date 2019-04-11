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

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#define TILE_WIDTH 32

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    float mySum = 0;

	// Allocating Shared Memory for Matrix M 
    __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];

    //Allocating Shared Memory for Matrix N
    __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];

    // Registers for thread Id's
    int myTx = threadIdx.x;

    int myTy =  threadIdx.y;

    // Register for Row check
    int myRow = blockIdx.y * TILE_WIDTH + threadIdx.y;
    
    // Register for Column check
    int myCol = blockIdx.x * TILE_WIDTH + myTx;

    //Main loop for copying to shared memory and calculating product.
    for (int m = 0; m < ceil(M.width/(float)TILE_WIDTH); m++) 
    {
        // Tiles loading from Global to Shared Memory
        
        if ((myRow < M.height) && ((m*TILE_WIDTH + myTx) < M.width))
        {
           
            sharedM[myTy][myTx] = M.elements[myRow*M.width + m*TILE_WIDTH + myTx];
        
        }
        
        // Inserting zero if out of bound
        else
        {

            sharedM[myTy][myTx] = 0;
       
        }
        
        // Tiles loading from Global to Shared Memory
        if ((myCol < N.width) && ((m*TILE_WIDTH + myTy) < N.height ))
        {

            sharedN[myTy][myTx] = N.elements[(m*TILE_WIDTH + myTy)*N.width + myCol];
       
        }

        // Inserting zero if out of bound
        else
        {

            sharedN[myTy][myTx] = 0;
        
        }
        
        // Waiting for threads to synch
        __syncthreads();

        // Dot Product
        for (int k = 0; k < TILE_WIDTH; k++) 
        {
            
            mySum += sharedM[myTy][k] * sharedN[k][myTx];
        
        }
        
        // Waiting for threads to use tiled values
        __syncthreads();

    }

    // Storing mySum back to output Matrix if in bound
    if((myRow < M.height) && (myCol < N.width)) 
    {
       
        P.elements[myRow*N.width + myCol] = mySum;
    
    }

}

#endif
