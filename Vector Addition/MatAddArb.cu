// Cuda Vec Add 

#include <iostream> 
#include <stdio.h>
#include <math.h>

// Each thread produces one element of the output matrix
__global__ void vecAdd_1(double *a, double *b, double *c, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < n)
	{
		c[id] = a[id] + b[id];
	}

}


// Each thread produces one row of the output matrix
__global__ void vecAdd_2(double *a, double *b, double *c, int n)
{
	int id = threadIdx.x;
	int id_1;

	for(int i = 0; i < n; i++)
	{
		id_1 = id * n + i; 

		c[id_1] = a[id_1] + b[id_1];

	}
}


// Each thread produces one column of the output matrix
__global__ void vecAdd_3(double *a, double *b, double *c, int n)
{
	int id = threadIdx.x;
	int id_1;

	for(int i = 0; i < n; i++)
	{
		id_1 = id + i * n; 
		
		c[id_1] = a[id_1] + b[id_1];

	}

}


int main(int argc, char* argv[])
{
	int n, m;
    
    printf( "Enter a value for matrix width :");
    scanf("%d", &n);
    //cin >> n;

    printf( "Enter a value for matrix height :");
    scanf("%d", &m);
    //cin>> m;

    if(n != m)
    {
    	printf( "Matrices dimensions are not entered correctly !\n");
    	exit(0);
    }

    if(n > 1024 || m > 1024)
    {
    	printf( "Matrices dimensions are not entered correctly \n");
    	exit(0);
    }

    // Host vectors
	double *h_a;
	double *h_b;
	double *h_c;

	//Device vectors

	double *d_a;
	double *d_b;
	double *d_c;

	// No of bytes
	size_t n_bytes = n*m*sizeof(double);

	// Allocating memory to host vectors
	h_a = (double*)malloc(n_bytes);
	h_b = (double*)malloc(n_bytes);
	h_c = (double*)malloc(n_bytes);

	// Allocating memory to device vectors
	cudaMalloc(&d_a, n_bytes);
	cudaMalloc(&d_b, n_bytes);
	cudaMalloc(&d_c, n_bytes);

	// Initializing values of host vectors randomly
	for(int i = 0; i < n*m; i++)
	{
		h_a[i] = sin(i) * sin(i);
		h_b[i] = sin(i) * sin(i);
	}

	// Copying values from host to device

	cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	blockSize = n;

	gridSize = (int)ceil(float(n/blockSize));

	// Starting the first kernel
	//vecAdd_1<<< n, blockSize>>>(d_a, d_b, d_c, n);

	// Starting the first kernel
	//vecAdd_2<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

	// Starting the first kernel
	vecAdd_3<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

	// Copying back from Device to Host
	cudaMemcpy(h_c, d_c, n_bytes, cudaMemcpyDeviceToHost);

	double sum = 0;

	for(int j = 0; j < n; j++)
	{
		sum += h_c[j];
	}
	printf("Final result is: %f\n", sum/(double)n);

	cudaFree(d_a);
    cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

}