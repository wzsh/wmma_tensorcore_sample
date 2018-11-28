//////////////////////////////////////////////////////////////////////
// Sample code to show how your project works
//    Created by Zong-Sheng Wang @ 2018/11/25


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h> 
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)


//__global__ void WMMAINT8()
using namespace nvcuda;

void InitMatrix(float * A, float *B, half *Ah, half *Bh, float *C)
{
	for (int i = 0; i < M_TOTAL*K_TOTAL; i++) {
		A[i] = rand() % 1000 / 1000.0f;
		Ah[i] = __float2half(A[i]);
	}
	for (int i = 0; i < K_TOTAL*N_TOTAL; i++) {
		B[i] = rand() % 1000 / 1000.0f;
		Bh[i] = __float2half(B[i]);
	}
	for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
		C[i] = rand() % 1000 / 1000.0f;
}


// Tensor core
__global__ void WMMAF16TensorCore(half *A, half *B, float *C, float *D)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);

	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

	wmma::fill_fragment(ab_frag, 0.0f);

	// AB = A*B
	int a_col, a_row, b_col, b_row, c_col, c_row;
	a_row = ix * M;
	b_row = iy * N;
	for (int k = 0; k<K_TOTAL; k += K) {
		a_col = b_col = k;

		if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
			wmma::load_matrix_sync(b_frag, B + b_col + b_col * K_TOTAL, K_TOTAL);

			// Perform the matrix multiplication
			wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	// D = AB + C
	c_col = b_row;
	c_row = a_row;
	if (c_row < M_TOTAL && c_col < N_TOTAL) {
		wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
	}
}

cudaError_t CalcByWMMA(half *A, half *B, float *C, float *D)
{
	cudaError_t cuda_status;
	dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 4 * WARP_SIZE;
	blockDim.y = 4;

	gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
	gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

	// for Performance Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	WMMAF16TensorCore <<<gridDim, blockDim >>>(A, B, C, D);
	cuda_status = cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// for Performance Metrics
	printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
	// references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2) / milliseconds / 1e9);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return cuda_status;
}

// GPU version without Tensor Core
__global__ void cuda_matrix_mul(const half *A, const half *B, float *R)
{
	int bId = blockIdx.y * gridDim.x + blockIdx.x;
	float sum = __half2float(A[blockIdx.y* M_TOTAL + threadIdx.x] * B[threadIdx.x * N_TOTAL + blockIdx.x]);
	__syncthreads();
	//printf("Thread %d In block %d : R = %d, sum = %d\n", threadIdx.x, bId, temp, sum);
	atomicAdd(&R[bId], sum);

}

__global__ void cuda_matrix_add(const float *A, const float *B, float *R)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * blockDim.x + ix;
	R[idx] = A[idx] + B[idx];
}


cudaError_t CalcByCUDA(half *A, half *B, float *C, float *R)
{
	cudaError_t cuda_status;
	dim3 gridDim, blockDim;

	blockDim.x = 4 * WARP_SIZE;
	blockDim.y = 4;

	gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
	gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

	// for Performance Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);


	cuda_matrix_mul <<<gridDim, blockDim >> > (A, B, R);
	cuda_status = cudaDeviceSynchronize();
	cuda_matrix_add<<<gridDim, blockDim >> > (R, C, R);
	cuda_status = cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("[+] GPU(without Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
	printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2) / milliseconds / 1e9);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return cuda_status;
}



// CPU version
void matrix_add(const float *A, const float *B, unsigned int row, unsigned int col, float *R) {
	for (unsigned int c = 0; c < col; c++) {
		for (unsigned int r = 0; r < row; r++) {
			unsigned int i = c*row + r;
			R[i] = A[i] + B[i];
		}
	}
}

void matrix_mul(const float *A, unsigned int a_row, unsigned int a_col, const float *B, unsigned int b_row, unsigned int b_col, float *R) {
	memset(R, 0, a_col*b_row * sizeof(float));
	for (unsigned int c = 0; c < a_col; c++) {
		for (unsigned int r = 0; r < b_row; r++) {
			unsigned int index = c * b_row + r;
			for (unsigned int i = 0; i < a_row; i++) {
				R[index] +=  A[c*a_row + i] * B[i*b_row + r];
			}
		}
	}
}

void CalcByCPU(float *A, float *B, float *C, float *D)
{
	matrix_mul(A, M_TOTAL, K_TOTAL, B, K_TOTAL, N_TOTAL, D);
	matrix_add(D, C, M_TOTAL, N_TOTAL, D);
}


int main()
{
	cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		printf("cudaSetDevice failed! ");
		return 1;
	}

	// float Matrix on host for cpu version
	float *hostA = (float*)malloc(sizeof(float) * M_TOTAL * K_TOTAL);
	float *hostB = (float*)malloc(sizeof(float) * K_TOTAL * N_TOTAL);
	float *hostD = (float*)malloc(sizeof(float) * M_TOTAL * N_TOTAL);

	// Matrix on device
	half *A;
	half *B;
	float *C;
	float *D;
	float *D2;

	// CUDA Unified Memory 
	cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * K_TOTAL);
	cudaMallocManaged((void **)&B, sizeof(half) * K_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&C, sizeof(float) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&D, sizeof(float) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&D2, sizeof(float) * M_TOTAL * N_TOTAL);

	// Init matrix A B C on host
	//InitHostMatrix(host_A, host_B, host_C);
	printf("[*] Initializing Matrix...\n");
	InitMatrix(hostA, hostB, A, B, C);
	printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
	printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
	printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);

	// computing with CUDA
	printf("[*] Computing D = A * B + C on GPU without Tensor Cores...\n");
	cuda_status = CalcByCUDA(A, B, C, D2);

	// computing with tensor core
	printf("[*] Computing D = A * B + C on GPU with Tensor Cores...\n");
	// D = A * B +C, D holds the result after ret
	cuda_status = CalcByWMMA(A, B, C, D);

	
// computing with CPU
	printf("[*] Computing D = A * B + C  on CPU...");
	int begintime, endtime;
	begintime = clock();
	CalcByCPU(hostA, hostB, C, hostD);
	endtime = clock();
	printf("OK\n");
	printf("[*] CPU Elapsed Time: %fs\n", (endtime-begintime)/1000.0f);

// Verification
	printf("[*] Verifying result...\n");
	for (int i = 0; i < M_TOTAL * N_TOTAL; i++) {
		if (fabs(D[i] - hostD[i]) > 0.1f)
			printf("[-] Mismatch index=%d TensorCore=%f HOST=%f\n", i, D[i], hostD[i]);
	}
	printf("[+] Verification End\n");

	cuda_status = cudaDeviceReset();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceReset failed! ");
		return 1;
	}

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	cudaFree(D2);

	free(hostA);
	free(hostB);
	free(hostD);

	return 0;
}
