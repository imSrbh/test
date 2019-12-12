#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>


#define N 512
void random_ints(int* a, int M)
{
        int i;
        for (i=0; i<M; i++) {
                a[i] = rand() % 5000;
        }
}
__global__ void add(int *a, int *b, int *c, int n) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < n) c[index] = a[index] + b[index];
}
int main(void) {

        // host copies of a, b, c
        int *a, *b, *c;
        // device copies of a, b, c
        int *d_a, *d_b, *d_c;
        int size = N * sizeof(int);


        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);


        // Alloc space for host copies of a, b, c and setup input values
        a = (int *)malloc(size);
        random_ints(a, N);
        b = (int *)malloc(size);
        random_ints(b, N);
        c = (int *)malloc(size);

        // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        // Launch add() kernel on GPU with N blocks
        add<<<2, N>>>(d_a, d_b, d_c, N);

        // Copy result back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        for(int i=0; i<N; i++) {
                printf("%d", c[i]);
        }

        // Cleanup
        free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return 0;
    }        