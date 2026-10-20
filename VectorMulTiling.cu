#include<stdio.h>
#include<cuda_runtime.h>
#define TILE_WIDTH 16
__global__ void matrixMulTiled(const float *A,const float *B,float *c,int N){
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    int tx= threadIdx.x;
    int ty= threadIdx.y;
    int row = blockIdx.y*TILE_WIDTH +ty;
    int col = blockIdx.x*TILE_WIDTH +tx;
    float sum =0.0f;
    int numTiles=(N+TILE_WIDTH -1)/TILE_WIDTH;
    for (int m=0;m<numTiles;m++){
        int a_col=m*TILE_WIDTH +tx;
        if (row < N && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f; // 越界部分补0
        }int b_row = m * TILE_WIDTH + ty;
        if (b_row < N && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f; // 越界部分补0
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        c[row * N + col] = sum;
    }
}
int main() {
    int N = 512; // 512方矩
    size_t bytes = N * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 初始化矩阵 (全设为 1.0f，这样结果矩阵每个元素都应该是 N)可使用generate,rand方法
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 配置执行维度
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // 启动核函数
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 简单验证结果
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != (float)N) {
            printf("Error at index %d: %f != %f\n", i, h_C[i], (float)N);
            correct = false;
            break;
        }
    }
    if (correct) printf("Tiled Matrix Multiplication Successful! All values = %d\n", N);

    // 释放内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}

