#include <stdio.h>

#define BLOCK_SIZE 1024

template <typename scalar_t>
__host__ 
void randomiseInput(scalar_t* h_input, size_t size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_input[i * size + j] = static_cast<scalar_t>(rand()) / RAND_MAX;
        }
    }
}


template <typename scalar_t>
__host__
void printMatrix(scalar_t* h_input, size_t size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", h_input[i * size + j]);
        }
        printf("\n");
    }
}

template <typename scalar_t>
__global__ void vectorAddKernel(scalar_t* __restrict__ d_A, 
                                scalar_t* __restrict__ d_B, 
                                scalar_t* __restrict__ d_C, 
                                size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t size_kernel = sizeof(int4) / sizeof(scalar_t);

    int4 Ain = ((int4*)d_A)[i]; //casting pointer to int4 128-bit wide loads
    int4 Bin = ((int4*)d_B)[i];
    int4 Cout;

    scalar_t* Ax = (scalar_t*)&Ain;
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
    // ToDo: unaligned boundaries
}

template <typename scalar_t>
__host__ void vectorAdd(scalar_t* h_A, scalar_t* h_B, scalar_t* h_C, size_t size) {
    float* d_A; 
    float* d_B;
    float* d_C;

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
    
    vectorAddKernel<scalar_t><<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, size*size);

    cudaMemcpy(h_C, d_C, size * size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename scalar_t>
__host__ void vectorAddCpu(scalar_t* h_A, scalar_t* h_B, scalar_t* h_C, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

template <typename scalar_t>
__host__ void verifyResult(scalar_t* h_C, scalar_t* h_C_cpu, size_t size) {
    for (size_t i = 0; i < size * size; i++) {
        if (h_C[i] != h_C_cpu[i]) {
            printf("Error at position %zu: %f != %f\n", i, h_C[i], h_C_cpu[i]);
        }
    }
}

int main(int argc, char** argv) {
    int shape = atoi(argv[1]);
    int DEBUG = atoi(argv[2]);

    // Allocate host memory
    float* h_A = (float*)malloc(shape * shape * sizeof(float));
    float* h_B = (float*)malloc(shape * shape * sizeof(float));
    float* h_C = (float*)malloc(shape * shape * sizeof(float));
    float* h_C_cpu = (float*)malloc(shape * shape * sizeof(float));

    // Randomise input
    randomiseInput<float>(h_A, shape);
    randomiseInput<float>(h_B, shape);
    if (DEBUG) {
        printf("Matrix A:\n");
        printMatrix<float>(h_A, shape);
        printf("Matrix B:\n");
        printMatrix<float>(h_B, shape);
    }

    // Vector add
    vectorAdd<float>(h_A, h_B, h_C, shape);

    if (DEBUG) {
        printf("Matrix C:\n");
        printMatrix<float>(h_C, shape);
    }

    vectorAddCpu<float>(h_A, h_B, h_C_cpu, shape);

    verifyResult<float>(h_C, h_C_cpu, shape);
}