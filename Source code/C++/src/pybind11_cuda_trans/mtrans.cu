#include <cuda_runtime.h>
#include <stdio.h>
#define BDIMX 16
#define BDIMY 16

__global__ void kernel_mtrans(int* A, int* B, int M, int N);

void cu_mtrans(int* A, int* B, int M, int N)
{
	int *d_a, *d_b;

	dim3 blk;
	blk.x = BDIMX; blk.y = BDIMY;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(unsigned int)*M*N;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_mtrans << < grid, blk >> > (d_a, d_b, M, N);

	cudaMemcpy(B, d_b, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
}

__global__ void kernel_mtrans(int* A, int* B, int M, int N)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int ixi = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iyi = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ti = iyi * M + ixi;
	unsigned int to;
	unsigned int bidx, irow, icol;

	if (ti == 0)
		printf("cuda matrix (%d, %d) transposition\n", N, M);

	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;
	unsigned int ixo = blockIdx.y * blockDim.y + icol;
	unsigned int iyo = blockIdx.x * blockDim.x + irow;
	to = iyo * N + ixo;
	if (ixi < M && iyi < N)
	{
		tile[threadIdx.y][threadIdx.x] = A[ti];
	}
	__syncthreads();
	if (ixo < N && iyo < M)
	{
		B[to] = tile[icol][irow];
	}
}

