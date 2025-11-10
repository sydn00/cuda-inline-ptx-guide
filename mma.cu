/*
"h" = .u16 reg
"r" = .u32 reg
"l" = .u64 reg
"q" = .u128 reg
"f" = .f32 reg
"d" = .f64 reg
"n" = immediate integer operands

= : write, output                   (ex. "=r")
+ : read-write, input and output    (ex. "+r")
  : read, input                     (ex. "r")

For double precision 8x8x4 (m n k)
*/

#include <iostream>
#include <numeric>


constexpr int m = 8;
constexpr int n = 8;
constexpr int k = 4;

constexpr int sizeA = m * k;
constexpr int sizeB = k * n;
constexpr int sizeC = m * n;


__global__ void matmul(double *d_A, double *d_B, double *d_C){

    int tid = threadIdx.x;
    int laneid = tid % warpSize;

    extern __shared__ double s[];
    double* s_A = s;
    double* s_B = s_A + sizeA;
    double* s_C = s_B + sizeB; 

    //copy from device memory to shared memory
    for(int i = tid; i < sizeA; i += blockDim.x)
        s_A[i] = d_A[i];

    for(int i = tid; i < sizeB; i += blockDim.x)
        s_B[i] = d_B[i];

    for(int i = tid; i < sizeC; i += blockDim.x)
        s_C[i] = d_C[i];

    __syncthreads();

    //copy from shared memory to register
    double a0;
    double b0;
    double c[2] = {};
    double d[2];
    
    int row = laneid >> 2;
    int col = laneid % 4;
    a0 = s_A[row * k + col];      //row-major

    row = laneid % 4;
    col = laneid >> 2;
    b0 = s_B[col * k + row];      //col-major

    //Perform mma
    asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5}; \n"
                :"=d"(d[0]), "=d"(d[1])
                :"d"(a0),
                 "d"(b0),
                 "d"(c[0]), "d"(c[1])
                );
    
    //copy from register to shared memory
    const int groupID           = laneid >> 2;
    const int threadID_in_group = laneid % 4;

    row = groupID;
    for(int i = 0; i < 2; i++)
    {
        col = (threadID_in_group * 2) + (i & 0x1);
        s_C[row * n + col] = d[i];             //row-major
    }
    __syncthreads();

    //copy from shared memory to device memory
    for(int i = tid; i < sizeC; i += blockDim.x)
        d_C[i] = s_C[i];

}


int main(){
    
    double *h_A = new double[sizeA];
    double *h_B = new double[sizeB];
    double *h_C = new double[sizeC];
    
    double *d_A;
    double *d_B;
    double *d_C;

    std::fill(h_A, h_A + sizeA, 4);
    std::fill(h_B, h_B + sizeB, 3);
    std::fill(h_C, h_C + sizeC, 0);

    cudaMalloc((void**)&d_A, sizeA * sizeof(double));
    cudaMalloc((void**)&d_B, sizeB * sizeof(double));
    cudaMalloc((void**)&d_C, sizeC * sizeof(double));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC * sizeof(double), cudaMemcpyHostToDevice);

    int shmemSize = sizeA + sizeB + sizeC;
    matmul<<<1,32,shmemSize * sizeof(double)>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC * sizeof(double), cudaMemcpyDeviceToHost);

    double result = std::reduce(h_C, h_C + sizeC);
    
    std::cout << "Result:"    << result << std::endl;
    std::cout << "Expected: " << m * n * (12 * k) << std::endl;

    delete[] h_A; delete[] h_B; delete[] h_C; cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}