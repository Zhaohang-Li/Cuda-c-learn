#include<stdio.h>
#include<cuda_runtime.h>
__global__ void vectorAdd(int *a, int *b, int*c , int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
int main(){
    const int N = 1000;
    const size_t bytes = N*sizeof(int);
    int h_a[N], h_b[N], h_c[N];
    for(int i=0; i<N; i++){
        h_a[i] = i;
        h_b[i] = i*2;
    }
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);
    cudaMalloc((void**)&d_c,bytes);
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (int)ceil((float)N/blockSize);
    vectorAdd<<<gridSize,blockSize>>>(d_a,d_b,d_c,N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}