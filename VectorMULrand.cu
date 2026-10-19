#include <cstdlib> /*C语言中应该叫做<stdlib.h>*/
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;
/*假设矩阵为XY乘以YZ的格式*/
__global__ void matrixMUl(int *a,int *b , int *c, int X, int Y,int Z){
    int row = blockIdx.y * blockDim.y +threadIdx.y;
    int col = blockIdx.x * blockDim.x +threadIdx.x;
    if(row<X&& col<Z){
        c[row*Z+col]=0;
            for (int i=0;i < Y; i++){
            c[row*Z+col]+=a[row*Y+i]*b[i*Z+col];
        }
    }
}
int main(){
    int X= 1<<(rand()%11);
    int Y= 1<<(rand()%11);
    int Z= 1<<(rand()%11);
    vector<int>h_a (X*Y);
    vector<int>h_b (Y*Z);
    vector<int>h_c (X*Z);
    generate(h_a.begin(),h_a.end(),[](){return rand()%100;});
    generate(h_b.begin(),h_b.end(),[](){return rand()%100;});
    generate(h_c.begin(),h_c.end(),[](){return rand()%100;});
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 4 * X * Y); // 或者写 X * Y * sizeof(int)
    cudaMalloc(&d_b, 4 * Y * Z);
    cudaMalloc(&d_c, 4 * X * Z);
    cudaMemcpy(d_a,h_a.data(),4*X*Y,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),4*Y*Z,cudaMemcpyHostToDevice);
    int tx = 8 << (rand() % 3);
    int ty = 8 << (rand() % 3);
    dim3 threads(tx, ty);
     dim3 blocks(ceil((float)Z / tx), ceil((float)X / ty));
     // 1. 启动核函数进行矩阵乘法计算
    matrixMUl<<<blocks, threads>>>(d_a, d_b, d_c, X, Y, Z);

    // 2. 把算好的结果从设备端 (d_c) 搬回主机端 (h_c.data())
    cudaMemcpy(h_c.data(), d_c, 4 * X * Z, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

  return 0;
}