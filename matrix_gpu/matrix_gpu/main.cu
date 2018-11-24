/////////////////////////////////////////
// Calcuating Matrix A*B+C (CUDA Version)
// Created by Wang Zong-Sheng
// 2018/10/18
#include <iostream>
using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define A_ROW 3
#define A_COL 2
#define B_ROW 2
#define B_COL 3
#define C_ROW 2
#define C_COL 2
#define MP_NUM 15
#define CORES_PER_MP 192

template <typename T>
__global__ void cuda_matrix_mul(const T *A, const T *B,  T *R)
{
	int bId = blockIdx.y * gridDim.x + blockIdx.x;
	T sum = A[blockIdx.y* A_ROW + threadIdx.x] * B[threadIdx.x * B_ROW + blockIdx.x];
	__syncthreads();
	//printf("Thread %d In block %d : R = %d, sum = %d\n", threadIdx.x, bId, temp, sum);
	atomicAdd(&R[bId], sum);
	
}

//template <typename T>
//__global__ void cuda_matrix_mul(const T *A, const T *B, T *R)
//{
//	int bId = blockIdx.y * gridDim.x + blockIdx.x;
//	int sum = 0;
//	for(int i=0; i<A_ROW; i++)
//		sum += A[blockIdx.y* A_ROW + i] * B[i * B_ROW + blockIdx.x];
//	R[bId] = sum;
//
//}

template <typename T>
__global__ void cuda_matrix_add(const T *A, const T *B, T *R)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * blockDim.x + ix;
	R[idx] = A[idx] + B[idx];
}


// using CUDA to implement AxB+C
template <typename T>
cudaError_t matrix_mul_add_cuda(const T *A, unsigned int a_row, unsigned int a_col, 
								const T *B, unsigned int b_row, unsigned int b_col, 
								const T *C, unsigned int c_row, unsigned int c_col,
								T *R, T *AB)
{
	T *dev_a = 0;
	T *dev_b = 0;
	T *dev_c = 0;
	T *dev_ab = 0;
	T *dev_r = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for matrics
	cudaStatus = cudaMalloc((void**)&dev_a, a_row * a_col * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, b_row * b_col * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_c, c_row * c_col * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ab, b_row * a_col * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_r, c_row * c_col * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		goto Error;
	}


	// Copy input matrics from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, A, a_row * a_col * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, B, b_row * b_col * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_c, C, c_row * c_col * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU
	// In our case, K40c GPU has 15MP, and 192cores per MP
	dim3 grids(B_ROW, A_COL);
	//cuda_matrix_mul <T> <<<grids, 1>>> (dev_a, dev_b, dev_ab);
	cuda_matrix_mul <T> <<<grids, A_ROW >>> (dev_a, dev_b, dev_ab);

	cudaDeviceSynchronize();

	dim3 threads(C_ROW, C_COL);
	cuda_matrix_add <T> <<<1, threads>>> (dev_ab, dev_c, dev_r);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(AB, dev_ab, b_row * a_col * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(R, dev_r, c_row * c_col * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_a);
	cudaFree(dev_c);
	cudaFree(dev_ab);
	cudaFree(dev_r);

	return cudaStatus;
}

template <typename T>
void print_matrix(T *M, unsigned int row, unsigned int col) {
	for (unsigned int c = 0; c < col; c++) {
		for (unsigned int r = 0; r < row; r++) {
			cout << M[c*row + r] << ", ";
		}
		cout << endl;
	}
}

int main()
{
	const int A[A_ROW*A_COL] = { 1, 0, -3,
		-2, 4,  1 };
	const int B[B_ROW*B_COL] = { 2, -1,
		3,  0,
		-5,  2 };
	const int C[C_ROW*C_COL] = { 3, -1,
		-2,  2 };
	int AB[A_COL*B_ROW];
	int R[C_ROW*C_COL];

	cudaError_t cudaStatus;
	cudaStatus = matrix_mul_add_cuda<int>(A, A_ROW, A_COL, B, B_ROW, B_COL, C, C_ROW, C_COL, R, AB);


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		//cout << "cudaDeviceReset failed!" << endl;
		return 1;
	}

	// for printing results
	cout << "A = " << endl;
	print_matrix<const int>(A, A_ROW, A_COL);

	cout << endl << "B = " << endl;
	print_matrix<const int>(B, B_ROW, B_COL);

	cout << endl << "C = " << endl;
	print_matrix<const int>(C, C_ROW, C_COL);

	cout << endl << "Result:" << endl;
	cout << "A x B = " << endl;
	print_matrix(AB, B_ROW, A_COL);

	cout << endl << "A x B + C = " << endl;
	print_matrix<int>(R, C_ROW, C_COL);


	return 0;
}