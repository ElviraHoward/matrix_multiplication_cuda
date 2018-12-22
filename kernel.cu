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