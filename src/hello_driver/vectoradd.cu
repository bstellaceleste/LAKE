#include <stdio.h>

// Kernel definition: executed on the GPU
__global__ void vectorAdd(unsigned long *A, unsigned long *B, unsigned long *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
    // unsigned long tmp = 1234;
    // C[0] = tmp;
}
