#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDACC_RTC__
#define __CUDACC__
#include <device_functions.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
typedef struct {
	int width;
	int height;
	float* elements;
	int step;
} Matrix;

using namespace std;

#define BLOCK_SIZE 4

__global__ void MatrixMulKernel(const Matrix, const Matrix, Matrix);

void MatrixMul(const Matrix A, const Matrix B, Matrix C) {
	Matrix d_A;
	d_A.width = d_A.step = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	cout << "CUDA malloc A: " << cudaGetErrorString(err) << endl;
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = d_B.step = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	err = cudaMalloc(&d_B.elements, size);
	cout << "CUDA malloc B: " << cudaGetErrorString(err) << endl;
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.width = d_C.step = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	err = cudaMalloc(&d_C.elements, size);
	cout << "CUDA malloc C: " << cudaGetErrorString(err) << endl;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatrixMulKernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	cout << "Run kernel: " << cudaGetErrorString(err) << endl;

	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	cout << "Copy C off of device: " << cudaGetErrorString(err) << endl;

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.step + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.step + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.step = A.step;
	Asub.elements = &A.elements[A.step * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

__global__ void MatrixMulKernel(Matrix A, Matrix B, Matrix C) {
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	float Cvalue = 0.0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		__syncthreads();
	}

	SetElement(Csub, row, col, Cvalue);
}

int main() {
	Matrix A, B, C;
	int a1, a2, b1, b2;
	a1 = BLOCK_SIZE;
	a2 = BLOCK_SIZE;
	b1 = a2;
	b2 = BLOCK_SIZE;

	A.height = a1;
	A.width = a2;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));

	B.height = b1;
	B.width = b2;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));

	C.height = A.height;
	C.width = B.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			A.elements[i * A.width + j] = rand() % 10;

	for (int i = 0; i < B.height; i++)
		for (int j = 0; j < B.width; j++)
			B.elements[i * B.width + j] = rand() % 10;

	MatrixMul(A, B, C);

	cout << "---Result of calculating:" << endl;

	cout << "Martix A: " << endl;
	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < A.width; j++)
			cout << A.elements[i * A.width + j] << " ";
		cout << endl;
	}
	cout << endl;

	cout << "Martix B: " << endl;
	for (int i = 0; i < B.height; i++) {
		for (int j = 0; j < B.width; j++)
			cout << B.elements[i * B.width + j] << " ";
		cout << endl;
	}
	cout << endl;

	cout << "Martix C: " << endl;
	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++)
			cout << C.elements[i * C.width + j] << " ";
		cout << endl;
	}
	cout << endl;

	return 0;
}
