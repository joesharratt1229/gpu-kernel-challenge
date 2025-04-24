#include <stdio.h>
#include <cuda_bf16.h>

#define BLOCK_SIZE 1024

template <typename scalar_t>
__host__ 
void randomiseInput(scalar_t* h_input, size_t size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float random = static_cast<float>(rand()) / RAND_MAX;
            h_input[i * size + j] = static_cast<scalar_t>(random);
        }
    }
}


template <typename scalar_t>
__host__
void printMatrix(scalar_t* h_input, size_t size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float temp = static_cast<float>(h_input[i * size + j]);
            printf("%f ", temp);
        }
        printf("\n");
    }
}

template<typename scalar_t>
__global__ void vectorAddKernel32BitLoad(scalar_t* __restrict__ d_A, 
                                         scalar_t* __restrict__ d_B, 
                                        scalar_t* __restrict__ d_C, 
                                         size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t size_kernel = sizeof(int4) / sizeof(scalar_t);
    
    if ((i * sizeof(scalar_t) + size_kernel - 1)< size) {
        for (size_t j = 0; j < size_kernel; j++) {
            d_C[i * size_kernel + j] = d_A[i * size_kernel + j] + d_B[i * size_kernel + j];
        }
    }
}

template <typename scalar_t>
__global__ void vectorAddKernel(scalar_t* __restrict__ d_A, 
                                scalar_t* __restrict__ d_B, 
                                scalar_t* __restrict__ d_C, 
                                size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t size_kernel = sizeof(int4) / sizeof(scalar_t);

    int4 Ain = ((int4*)d_A)[i]; //casting pointer to int4 128-bit integer - vectorised load into registers
    int4 Bin = ((int4*)d_B)[i];
    int4 Cout;

    scalar_t* Ax = (scalar_t*)&Ain; //casting int4 to scalar_t
    scalar_t* Bx = (scalar_t*)&Bin;
    scalar_t* Cx = (scalar_t*)&Cout;

    if ((i*size_kernel + size_kernel -1) < size) {
        for (size_t j = 0; j < size_kernel; j++) {
            Cx[j] = Ax[j] + Bx[j];
        }

        //write back to global memory
        int4* Cx4 = (int4*)d_C; //cast output pointer to int4
        Cx4[i] = Cout; //write back to global memory
    }
    else {
        for (size_t j = 0; j < size_kernel; j++) {
            if ((i*size_kernel + j) < size) {
                d_C[i*size_kernel + j] = d_A[i*size_kernel + j] + d_B[i*size_kernel + j];
            }
        }
    }
}

template <typename scalar_t>
__host__ void vectorAdd(scalar_t* h_A, scalar_t* h_B, scalar_t* h_C, size_t size) {
    __nv_bfloat16* d_A; 
    __nv_bfloat16* d_B;
    __nv_bfloat16* d_C;

    //calculate number of blocks
    constexpr size_t elements_per_thread = sizeof(int4) / sizeof(scalar_t);
    const int numBlocks = (size * size + (BLOCK_SIZE * elements_per_thread - 1)) / (BLOCK_SIZE * elements_per_thread);
    printf("numBlocks: %d\n", numBlocks);

    cudaMalloc(&d_A, size * size * sizeof(scalar_t));
    cudaMalloc(&d_B, size * size * sizeof(scalar_t));
    cudaMalloc(&d_C, size * size * sizeof(scalar_t));

    cudaMemcpy(d_A, h_A, size * size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size * size * sizeof(scalar_t));
    
    vectorAddKernel<scalar_t><<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, size*size); //40% reduction in time and 20% increase in memory bandwidth
    //vectorAddKernel32BitLoad<scalar_t><<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, size*size);

    cudaMemcpy(h_C, d_C, size * size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename scalar_t>
__host__ void vectorAddCpu(scalar_t* h_A, scalar_t* h_B, float* h_C, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        float a_temp = static_cast<float>(h_A[i]);
        float b_temp = static_cast<float>(h_B[i]);
        h_C[i] = a_temp + b_temp;
    }
}

template <typename scalar_t>
__host__ void verifyResult(scalar_t* h_C, float* h_C_cpu, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        float c_temp = static_cast<float>(h_C[i]);
        float c_cpu_temp = h_C_cpu[i];
        float epsilon = 1e-2;
        if (fabs(c_temp - c_cpu_temp) > epsilon) {
            printf("Error at position %zu: %f != %f\n", i, c_temp, c_cpu_temp);
        }
    }
}

int main(int argc, char** argv) {
    int shape = atoi(argv[1]);
    int DEBUG = atoi(argv[2]);

    // Allocate host memory
    __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(shape * shape * sizeof(__nv_bfloat16)); //using bfloat16 furhter reduces memory usage 10us - 7us
    __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(shape * shape * sizeof(__nv_bfloat16));
    __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(shape * shape * sizeof(__nv_bfloat16));
    float* h_C_cpu = (float*)malloc(shape * shape * sizeof(float));

    // Randomise input
    randomiseInput<__nv_bfloat16>(h_A, shape);
    randomiseInput<__nv_bfloat16>(h_B, shape);
    if (DEBUG) {
        printf("Matrix A:\n");
        printMatrix<__nv_bfloat16>(h_A, shape);
        printf("Matrix B:\n");
        printMatrix<__nv_bfloat16>(h_B, shape);
    }

    // Vector add
    vectorAdd<__nv_bfloat16>(h_A, h_B, h_C, shape);

    if (DEBUG) {
        printf("Matrix C:\n");
        printMatrix<__nv_bfloat16>(h_C, shape);
    }

    vectorAddCpu<__nv_bfloat16>(h_A, h_B, h_C_cpu, shape);

    verifyResult<__nv_bfloat16>(h_C, h_C_cpu, shape);
}