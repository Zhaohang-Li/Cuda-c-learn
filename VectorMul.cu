#include <vector>
#include <cstdlib>   // 提供 rand()
#include <algorithm> // 提供 std::generate
#include <iostream>
#include <cassert>   // 提供 assert 断言验证
/*这里写比对函数和打印函数 */
 __global__  void matrixMul(int *a,int *b,int *c,int N){
   int row = blockIdx.y*blockDim.y+threadIdx.y;
   int col = blockIdx.x*blockDim.x+threadIdx.x;
   /*这里的row和col是矩阵的行和列，注意这里的blockIdx是一个结构体，里面有三个变量x,y,z
   其中x,y,z分别表示三维空间中的三个维度的索引值，blockDim也是一个结构体，里面有三个变量x,y,z
   其中x,y,z分别表示三维空间中每个线程块中线程的数量，threadIdx也是一个结构体，里面有三个变量x,y,z
   其中x,y,z分别表示三维空间中每个线程块中线程的索引值*/
   if(row<N && col<N){
      int sum = 0;
      for(int i=0;i<N;i++){
         sum += a[row*N+i]*b[i*N+col];
      }
      c[row*N+col] = sum;
   }
}
/*这里我来实现矩阵乘法，用注释来表达我的思维过程*/
/*这里一般是先定义函数随后在主函数中启动，但是我
并不是es变量指的是一个方矩阵的字节大小 */
   /*接下来来创建数据，这里使用vetor方法来创造数组需要先导入哪些库呢*/
 int main() {
   int N = 2 <<10;/*首先这里<<为运算左移动算法，a<<b表示将a左移b位
   2的二进制为00000010,然后向左移动10位，在二进制下等价于2*10=1024*/
   /*同时这里的N指方矩阵的维度*/
   int bytes = N * N *sizeof(int);
   /*这里的byt数)并且这里h_a这个变量依然可以调用函数.begin我并不知道返回的结果到底是什么，那么
   这个方法到底从一个地址到地址来按照字节去自动适配填上数字的还是什么方法，并且rand()%100
   是int类型还是什么类型呢，我先注意这里)*/
   /*接下来来创建数据，这里使用vetor方法来创造数组需要先导入哪些库呢*/
   std::vector<int>h_a(N*N),h_b(N*N),h_c(N*N);
   /*因为这里使用来vector所以我导入来vertor这个库，vertor的使用方法为ver<数据类型>变量名(元素个数)
   直接赋值为{}那么这里建好来N*N各元素的三个数组对象其中两个个对应乘法中的两个矩阵，
   一个用于保存后续的结果*/
   std::generate(h_a.begin(),h_a.end(),[](){return rand()%100;});
   std::generate(h_b.begin(),h_b.end(),[](){return rand()%100;});
   int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);
    /*还有写法是cudaMalloc((void**)&d_a,bytes)这里为什么要加一个(void**)不理解，同时那么这里可以理解为
    cudaMalloc的规则是接受两个变量，一个是指针的地址，以此为开始开辟第二个变量为标量的字节大小的空间
    同时这里很狗的是cudamalloc应该不行，这里的m要大写，编程的这些规则好狗*/
    int Threads = 32;
    /*这里定义好线程数为32，其中的规则是SM操作最小单位为32个线程，我还不知道SM的详细原理*/
    int Blocks = N/Threads;
    /*这里矩阵中一行那么有blocks个线程块，每个线程块中有Threads个线程*/
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice);
    /*cudaMemcpy()中四个参数，第一个参数是目标地址，第二个参数是源地址，第三个参数是字节数，第四个参数是拷贝的方向
    并且这里关键的是地址，然而之前用vector创建出来的h_a等不是变量，而是对象，而对象这个词指的填入这个
    地方的指向该对象的地址，而.data()函数又做了什么呢，这里取出的是地址还是什么，还是做了好多的封装
    明明是C++却弄得这么垃圾*/
    /*并且这里如果只是随机实验的话是否可以直接在cudaMalloc的三个对象上使用generate来生成随机数就可以来*/
    dim3 threads(Threads,Threads);
    dim3 blocks(Blocks, Blocks);
    /*dim3是一个结构体，里面有三个变量用来说明维数那么这里都用三个维度来描述这些变量*/
    /*threads同时那么这里有两连个维度一个维度上有32线程，那么一个本体的threads应该就有1024个线程了
    一个threads正好对应一个矩阵的数据大小可以，那么来到blocks,其中也有两个维度，其实我发现这里用维度去
    说明这个东西语言好像是错误的，这里threds应该是1024维拍成一个维的空间，取决与你的视角，然后现在又开始分成
    blocks,注意这里的N是矩阵的维度，而不知大小，这里的数学关系我后面再看，其实我认为这里的数学关系
    理解起来很乖，没有数学的优美性*/
    matrixMul<<<blocks,threads>>>(d_a,d_b,d_c,N);
    /*用这里来启动这个函数但是我现在还完全没有写其中的内容这里我留意我使用了在主机上创建的两个移动到
    设备上的两个数据和一个保存在设备上的数据和一个保存着要计算的矩阵的一个方阵维数几个变量来作为入参*/
    cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost);
    /*这里是将计算结果从设备上拷贝到主机上来同时这里还有之前描绘的那个特点，这里的两个变量需要是地址
    我还不理解cudaMemcpy的原理，我后续再去学习*/
    /*这里应该把数据扳回host本质上任务已经完成了，只需要清理和释放这些内存空间就可以了
    但是这里用于学习我也写一个比对函数验证和打印一下*/
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    /*这里是否可以使用free函数呢，cudafree和cuda的区别是什么*/
    return 0;
    /*这里如果不写return会怎么样呢*/
 }
 