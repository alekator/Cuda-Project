#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<math.h>
#include<malloc.h>
#include<time.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

#define SIZE 512
__device__ void editRows(double* data, int matrixOrder, int rowNumber) {

    unsigned int idx = blockIdx.x; //get index of block
    if (idx > rowNumber && idx <= matrixOrder - 1) {
        //get divider
        double divider = data[matrixOrder * idx + rowNumber] / data[rowNumber * matrixOrder + rowNumber];
        for (int j = rowNumber; j < matrixOrder; j++)
            //row edit
            data[matrixOrder * idx + j] -= data[rowNumber * matrixOrder + j] * divider;
    }
}
__device__ void kernel_down(double* A, double* B, int n,
    int number_row, int number_column) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * n + ty;
    int column = bx * n + tx;

    if (number_column < column && number_row < row) {
        double glav = A[row * n + number_column] / A[number_row * n + number_column];

        if (number_column == column) B[row] -= B[number_row] * glav;
        A[row * n + column] -= glav * A[number_row * n + column];
    }
}
__global__ void gaussDeterminant(double* data, double* dev_data,double*dev_data1, int matrixOrder) {
    double determinant = 0;
    unsigned int idx = blockIdx.x; //get index of block
    //matrix to triangle
    for (int i = 0; i < matrixOrder- 1; ++i)
        kernel_down (dev_data, dev_data1, matrixOrder, i, i);
    //calculate determinant
    determinant = data[0];
    for (int i = 1; i < matrixOrder; i++) {
        determinant *= data[i * idx * matrixOrder + i];
    };

}

int main() {
    FILE* f1, * f2;
    
   
    //main cycle
    for (int n = 100; n < 3500; n += 20) {
        //matrix creation
        double* data = (double*)malloc(sizeof(double) * (n * n));
        double* data1 = (double*)malloc(n * sizeof(double));

        srand(time(0));
        for (int i = 0; i < n * n; i++) {
            data[i] = (double)(rand() % 101);
        }

        double determinant = 1;
        //write matrix order to file
        f1 = fopen("Matrix Orders.txt", "a");
        printf("matrix order = %d\n", n);
        fprintf(f1, "%d\n", n);
        fclose(f1);

        double* dev_data;
        double* dev_data1;
        //init events to get time of calculation
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        //allocate memory on device
        cudaMalloc((void**)&dev_data, sizeof(double) * (n * n));
        cudaMalloc((void**)&dev_data1, sizeof(double) * (n));
        cudaMemcpy(dev_data, data, sizeof(double) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data1, data1, sizeof(double) * n , cudaMemcpyHostToDevice);
        float timeToCalculate = 0.0;
       

        gaussDeterminant <<<n, 1 >>> (data, dev_data,dev_data1, n);
        cudaMemcpy(data, dev_data, n * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&timeToCalculate, begin, end);

        //write time to calculate in file
        f2 = fopen("Time in milli seconds GPU.txt", "a");
        fprintf(f2, "%.2f\n", timeToCalculate);
        fclose(f2);

        printf("time to calculate = %.2f milliseconds\n", timeToCalculate);

        //printf("determinant %f\n", determinant);

        //destroy events
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        //free memory
        cudaFree(dev_data);
        free(data);
    }
    return 0;
}

