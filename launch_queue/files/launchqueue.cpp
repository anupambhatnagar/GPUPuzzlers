// Builds on https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

#include <iostream>
#include <ostream>
#include <cstdlib>
#include <unistd.h> // for usleep()

#include <curand.h>
#include <cublas_v2.h>

#include <driver_types.h>
#include <helper_cuda.h>

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {

    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

int main() {
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 100;

    int cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    }

    cublasHandle_t handle;
    cudaStatus = cublasCreate(&handle);
    if (cudaStatus != cudaSuccess) {
        printf("cublasCreate failed!  Do you have a CUDA-capable GPU installed?\n");
    }


    // Allocate small arrays on GPU
    int N = 20;
    float *d_A[N], *d_B[N], *d_C[N];
    for(int i = 0; i < N; i++) {
        cudaMalloc(&d_A[i], nr_rows_A * nr_cols_A * sizeof(float));
        cudaMalloc(&d_B[i], nr_rows_B * nr_cols_B * sizeof(float));
        cudaMalloc(&d_C[i], nr_rows_C * nr_cols_C * sizeof(float));

       // Fill the arrays A[i] and B[i] on GPU with random numbers
       GPU_fill_rand(d_A[i], nr_rows_A, nr_cols_A);
       GPU_fill_rand(d_B[i], nr_rows_B, nr_cols_B);
	}

    int nr_rows_A_big, nr_cols_A_big, nr_rows_B_big, nr_cols_B_big, nr_rows_C_big, nr_cols_C_big;
    nr_rows_A_big = nr_cols_A_big = nr_rows_B_big = nr_cols_B_big = nr_rows_C_big = nr_cols_C_big = 1600;

    // Allocate big arrays on GPU
	int M = 2;
    float *d_A_big[M], *d_B_big[M], *d_C_big[M];
	for (int j = 0; j < M; j++) {
        cudaMalloc(&d_A_big[j], nr_rows_A_big * nr_cols_A_big * sizeof(float));
        cudaMalloc(&d_B_big[j], nr_rows_B_big * nr_cols_B_big * sizeof(float));
        cudaMalloc(&d_C_big[j], nr_rows_C_big * nr_cols_C_big * sizeof(float));

        // Fill the arrays A_big[j] and B_big[j] on GPU with random numbers
        GPU_fill_rand(d_A_big[j], nr_rows_A_big, nr_cols_A_big);
        GPU_fill_rand(d_B_big[j], nr_rows_B_big, nr_cols_B_big);
	}

    int SLEEP_US = 1000;
    // Warm up with the first N/2 and M/2 matrices.
    for (int i = 0; i < N/2; i++) {
        gpu_blas_mmul(handle, d_A[i], d_B[i], d_C[i], nr_rows_A, nr_cols_A, nr_cols_B);
	}
    for (int j = 0; j < M/2; j++) {
        gpu_blas_mmul(handle, d_A_big[j], d_B_big[j], d_C_big[j], nr_rows_A_big, nr_cols_A_big, nr_cols_B_big);
    }
    usleep(SLEEP_US);
    for (int j = 0; j < M/2; j++) {
        gpu_blas_mmul(handle, d_A_big[j], d_B_big[j], d_C_big[j], nr_rows_A_big, nr_cols_A_big, nr_cols_B_big);
    }
    for (int i = 0; i < N/2; i++) {
        gpu_blas_mmul(handle, d_A[i], d_B[i], d_C[i], nr_rows_A, nr_cols_A, nr_cols_B);
	}
    usleep(SLEEP_US);

    // Now try the next N/2 and M/2 matrices.
    for (int j = M/2; j < M; j++) {
        gpu_blas_mmul(handle, d_A_big[j], d_B_big[j], d_C_big[j], nr_rows_A_big, nr_cols_A_big, nr_cols_B_big);
    }
    for (int i = N/2; i < N; i++) {
        gpu_blas_mmul(handle, d_A[i], d_B[i], d_C[i], nr_rows_A, nr_cols_A, nr_cols_B);
	}

    usleep(SLEEP_US);
    for (int i = N/2; i < N; i++) {
        gpu_blas_mmul(handle, d_A[i], d_B[i], d_C[i], nr_rows_A, nr_cols_A, nr_cols_B);
	}
    for (int j = M/2; j < M; j++) {
        gpu_blas_mmul(handle, d_A_big[j], d_B_big[j], d_C_big[j], nr_rows_A_big, nr_cols_A_big, nr_cols_B_big);
    }

    cublasDestroy(handle);


    //Free GPU memory
    for (int i = 0; i < N; i++) {
      cudaFree(d_A[i]);
      cudaFree(d_B[i]);
      cudaFree(d_C[i]);
        
    }
    for (int j = 0; j < N; j++) {
      cudaFree(d_A_big[j]);
      cudaFree(d_B_big[j]);
      cudaFree(d_C_big[j]);
    }

    return 0;
}
