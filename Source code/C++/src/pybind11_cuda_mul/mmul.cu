#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void kernel_mmul(int* A, int* B, int* C, int M, int N, int s, dim3 blk);

void cu_mmul(int* A, int* B, int* C, int M, int N, int s)
{
	int *d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = BLOCK_SIZE; blk.y = BLOCK_SIZE;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size_a = sizeof(unsigned int)*s*N;
	int size_b = sizeof(unsigned int)*M*s;
	int size_c = sizeof(unsigned int)*M*N;

	cudaMalloc((void **)&d_a, size_a);
	cudaMalloc((void **)&d_b, size_b);
	cudaMalloc((void **)&d_c, size_c);

	cudaMemcpy(d_a, A, size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size_b, cudaMemcpyHostToDevice);

	kernel_mmul << < grid, blk >> > (d_a, d_b, d_c, M, N, s, blk);

	cudaMemcpy(C, d_c, size_c, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_mmul(int* A, int* B, int* C, int M, int N, int s, dim3 blk)
{
	__shared__ int smem_m[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int smem_n[BLOCK_SIZE][BLOCK_SIZE];
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tmp = 0;

	if (row + col == 0)
		printf("cuda matrix ((%d, %d)(%d, %d)) multiplication\n", N, s, s, M);

	for (int stride = 0; stride <= s / blk.y; stride++) {
		int idm = stride * blk.y + row * s + threadIdx.x;
		if (row < N && blk.y * stride + threadIdx.x < s) {
			smem_m[threadIdx.y][threadIdx.x] = A[idm];
		}
		else {
			smem_m[threadIdx.y][threadIdx.x] = 0;
		}
		int idn = stride * blk.y * M + col + threadIdx.y * M;
		if (col < M && blk.y * stride + threadIdx.y < s) {
			smem_n[threadIdx.y][threadIdx.x] = B[idn];
		}
		else {
			smem_n[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for (int i = 0; i < blk.y; i++) {
			tmp += smem_m[threadIdx.y][i] * smem_n[i][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < N && col < M)
	{
		C[row * M + col] = tmp;
	}
}

