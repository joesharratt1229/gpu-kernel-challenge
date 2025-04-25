#include <stdio.h>
#include <cuda_bf16.h>

#define NUM_THREADS 1024
#define RED_SCALAR   0.299f
#define GREEN_SCALAR 0.587f
#define BLUE_SCALAR  0.114f


template <typename scalar_t>
__host__ 
void randomiseInput(scalar_t* h_input, size_t shape) {
    size_t npix = shape * shape;
    srand(static_cast<unsigned>(time(nullptr)));
    for (size_t i = 0; i < npix; ++i) {
        h_input[3*i    ] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
        h_input[3*i + 1] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
        h_input[3*i + 2] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
    }
}


template <typename scalar_t>
__host__ 
void rgbToGrayscaleCPU(const scalar_t* h_input, float* h_output, size_t shape) {
    size_t npix = shape * shape;
    for (size_t i = 0; i < npix; ++i) {
        float r = static_cast<float>(h_input[3*i]);
        float g = static_cast<float>(h_input[3*i + 1]);
        float b = static_cast<float>(h_input[3*i + 2]);
        h_output[i] = RED_SCALAR * r
                    + GREEN_SCALAR * g
                    + BLUE_SCALAR * b;
    }
}

template <typename scalar_t>
__global__ void rgbToGrayscaleKernel_16bit(scalar_t* __restrict__ d_input, 
                                     scalar_t* __restrict__ d_output, 
                                     size_t shape)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    constexpr int size_kernel = sizeof(int4) / sizeof(scalar_t);

    if ((index*size_kernel + size_kernel -1 < shape))
    {
        int4 d_in = ((int4*)d_input)[3*index];
        int4 d_in1 = ((int4*)d_input)[3*index+1];
        int4 d_in2 = ((int4*)d_input)[3*index+2];

        int4 d_out;

        scalar_t* f_in = (scalar_t*)&d_in;
        scalar_t* f_in1 = (scalar_t*)&d_in1;
        scalar_t* f_in2 = (scalar_t*)&d_in2;
        scalar_t* f_out = (scalar_t*)&d_out;

        f_out[0] = static_cast<scalar_t>(0.299)*f_in[0] + static_cast<scalar_t>(0.587)*f_in[1] + static_cast<scalar_t>(0.114)*f_in[2];
        f_out[1] = static_cast<scalar_t>(0.299)*f_in[3] + static_cast<scalar_t>(0.587)*f_in[4] + static_cast<scalar_t>(0.114)*f_in[5];
        f_out[2] = static_cast<scalar_t>(0.299)*f_in[6] + static_cast<scalar_t>(0.587)*f_in[7] + static_cast<scalar_t>(0.114)*f_in1[0];
        f_out[3] = static_cast<scalar_t>(0.299)*f_in1[1] + static_cast<scalar_t>(0.587)*f_in1[2] + static_cast<scalar_t>(0.114)*f_in1[3];
        f_out[4] = static_cast<scalar_t>(0.299)*f_in1[4] + static_cast<scalar_t>(0.587)*f_in1[5] + static_cast<scalar_t>(0.114)*f_in1[6];
        f_out[5] = static_cast<scalar_t>(0.299)*f_in1[7] + static_cast<scalar_t>(0.587)*f_in2[0] + static_cast<scalar_t>(0.114)*f_in2[1];
        f_out[6] = static_cast<scalar_t>(0.299)*f_in2[2] + static_cast<scalar_t>(0.587)*f_in2[3] + static_cast<scalar_t>(0.114)*f_in2[4];
        f_out[7] = static_cast<scalar_t>(0.299)*f_in2[5] + static_cast<scalar_t>(0.587)*f_in2[6] + static_cast<scalar_t>(0.114)*f_in2[7];

        int4* d_output_cast = (int4*)(d_output);
        d_output_cast[index] = d_out;

    } else {
        for (int i = 0; i < size_kernel; i++)
        {
            if (index*size_kernel + i < shape)
            {
               scalar_t red_element = d_input[3*(size_kernel*index)];
               scalar_t green_element = d_input[3*(size_kernel*index+i)+ 1];
               scalar_t blue_element = d_input[3*(size_kernel*index+i)+ 2];
               scalar_t output_element = static_cast<scalar_t>(RED_SCALAR)*red_element + static_cast<scalar_t>(GREEN_SCALAR)*green_element + static_cast<scalar_t>(BLUE_SCALAR)*blue_element;
               d_output[(size_kernel*index+i)] = output_element; 
            }
        }
    }
}


template <typename scalar_t>
__host__ void rgbToGrayscale(scalar_t* h_input, scalar_t* h_output, size_t shape)
{
    scalar_t* d_input;
    scalar_t* d_output;

    int elements_per_thread = sizeof(int4)/(sizeof(scalar_t));
    int num_blocks = ((shape*shape)+ (NUM_THREADS*elements_per_thread)-1)/(elements_per_thread*NUM_THREADS);

    cudaMalloc(&d_input, shape*shape*3*sizeof(scalar_t));
    cudaMalloc(&d_output, shape*shape*sizeof(scalar_t));

    cudaMemcpy(d_input, h_input, shape*shape*3*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0,  shape*shape*sizeof(scalar_t));

    rgbToGrayscaleKernel_16bit<scalar_t><<<num_blocks, NUM_THREADS>>>(d_input, d_output, shape*shape);

    cudaMemcpy(h_output, d_output, shape*shape*sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
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


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <shape> <DEBUG (0/1)>\n", argv[0]);
        return 1;
    }

    int shape = atoi(argv[1]);
    int DEBUG = atoi(argv[2]);

    __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16)*shape*shape*3);
    __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16)*shape*shape);
    float* h_output_cpu = (float*)malloc(sizeof(float)*shape*shape);

    randomiseInput<__nv_bfloat16>(h_input, shape);
    rgbToGrayscaleCPU<__nv_bfloat16>(h_input, h_output_cpu, shape);
    rgbToGrayscale<__nv_bfloat16>(h_input, h_output, shape);

    if (DEBUG)
    {
        verifyResult(h_output, h_output_cpu, shape);
        //printMatrix<__nv_bfloat16>(h_output, shape);
        //printMatrix<float>(h_output_cpu, shape);
    }

    free(h_input);
    free(h_output);
    free(h_output_cpu);
}