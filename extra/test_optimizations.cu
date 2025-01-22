#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
std::string float_to_hex_no_prefix(float value);

// Function to get precise time
std::chrono::steady_clock::time_point hpc_gettime( void )
{
    return std::chrono::high_resolution_clock::now();
}
int hpc_gettime_elapsed(std::chrono::steady_clock::time_point time)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(hpc_gettime()-time).count();
}

// Device-side (GPU) Horner's scheme without debugging
__device__ float horner_scheme_fma(const float* coeffs, int ParamHwDegree, float x, int jump=1) {
    float result = coeffs[0];
    for (int i = 1; i <= ParamHwDegree; ++i) {
        result = __fmaf_rn(result, x, coeffs[i*jump]); // FMA: result = result * x + coeffs[i]
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////////
/// Original kernel
//////////////////////////////////////////////////////////////////////////////


// Kernel function to evaluate piecewise polynomials and calculate partition IDs
__global__ void evaluate_polynomials(float* x_values, float* y_values, int* partition_ids,
                                      const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index * (D+1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma(partition_coeffs, D, x);
    }
}

//////////////////////////////////////////////////////////////////////////////
/// AoS kernels
//////////////////////////////////////////////////////////////////////////////

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template< int UNROLL_SIZE>
__global__ void evaluate_polynomials_AoS_unroll(float* x_values, float* y_values, int* partition_ids,
                                                const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma(partition_coeffs, D, x);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template< int UNROLL_SIZE>
__global__ void evaluate_polynomials_AoS_unroll2(float* x_values, float* y_values, int* partition_ids,
                                                 const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = -1;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            partition_index += (partition_index == -1) * ((i+1) * (x < partition_points[i + 1]));
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma(partition_coeffs, D, x);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template< int UNROLL_SIZE, int DEGREE>
__global__ void evaluate_polynomials_AoS_unroll_reg(float* x_values, float* y_values, int* partition_ids,
                                                    const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float partition_coeffs[DEGREE];  // register-side array
    int DEG = min(D+1,DEGREE);

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Evaluate polynomial using Horner's scheme with FMA
        float result = 0.;
        for (int i = 0; i <= D; i+=DEG) {
            // Get the next DEG coefficients for the corresponding polynomial in the partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                partition_coeffs[(j-i)] = coeffs[partition_index*(D+1)+j]; // AoS
            }
            // Evaluate polynomial using Horner's scheme with FMA and the next DEG coefficients of the corresponding partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                result = __fmaf_rn(result, x, partition_coeffs[(j-i)]);
            }
        }
        y_values[idx] = result;
    }
}

//////////////////////////////////////////////////////////////////////////////
/// SoA kernels
//////////////////////////////////////////////////////////////////////////////

// Kernel function to evaluate piecewise polynomials and calculate partition
template< int UNROLL_SIZE>
__global__ void evaluate_polynomials_SoA_unroll(float* x_values, float* y_values, int* partition_ids,
                                                const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma(partition_coeffs, D, x, P);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition
template< int UNROLL_SIZE>
__global__ void evaluate_polynomials_SoA_unroll2(float* x_values, float* y_values, int* partition_ids,
                                                 const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = -1;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            partition_index += (partition_index == -1) * ((i+1) * (x < partition_points[i + 1]));
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma(partition_coeffs, D, x, P);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template< int UNROLL_SIZE, int DEGREE>
__global__ void evaluate_polynomials_SoA_unroll_reg(float* x_values, float* y_values, int* partition_ids,
                                                    const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float partition_coeffs[DEGREE];  // register-side array
    int DEG = min(D+1,DEGREE);

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Evaluate polynomial using Horner's scheme with FMA
        float result = 0.;
        for (int i = 0; i <= D; i+=DEG) {
            // Get the next DEG coefficients for the corresponding polynomial in the partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                partition_coeffs[(j-i)] = coeffs[partition_index+j*P]; // SoA
            }
            // Evaluate polynomial using Horner's scheme with FMA and the next DEG coefficients of the corresponding partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                result = __fmaf_rn(result, x, partition_coeffs[(j-i)]);
            }
        }
        y_values[idx] = result;
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template< int UNROLL_SIZE, int DEGREE>
__global__ void evaluate_polynomials_SoA_shmem(float* x_values, float* y_values, int* partition_ids,
                                               const float* coeffs, const float* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float partition_coeffs[DEGREE];  // register-side array
    __shared__ float s_next_coeffs[256]; // SharedMemory-side array
    int DEG = min(D+1,DEGREE);

    if (idx < N) {
        float x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Store partition ID
        //        partition_ids[idx] = partition_index;

        // Evaluate polynomial using Horner's scheme with FMA
        float result = 0.;
        for (int i = 0; i <= D; i+=DEG) {
            // Load the next P*DEG coeffients in Shared-Memory - GlobalMem->SharedMem
            __syncthreads();
            if (threadIdx.x < P*DEG && (threadIdx.x+(i*P)) < P*(D+1)) {
                //TODO: even if I don't have enough threads, the next DEG partitions must be filled otherwise s_next_coeffs[partition_index+(j-i)*P] points to previous data
                s_next_coeffs[threadIdx.x] = coeffs[threadIdx.x + (i)*P];
            }
            __syncthreads();
            // Get the next DEG coefficients for the corresponding polynomial in the partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                partition_coeffs[(j-i)] = s_next_coeffs[partition_index+(j-i)*P];//coeffs[partition_index+j*P]; // SoA
            }
            // Evaluate polynomial using Horner's scheme with FMA and the next DEG coefficients of the corresponding partition
            for (int j = i; j < i+DEG && j <= D; j++) {
                result = __fmaf_rn(result, x, partition_coeffs[(j-i)]);
            }
        }
        y_values[idx] = result;
    }
}

void run_checksum(float* x, float* y1, float *y2, int N) {
    for(int i = 0; i<N; i++){
        if ((y1[i]-y2[i]) >= 0.0001) {
            printf("checksum ERROR! idx:[%d] x=%f expected:%f -> got:%f\n", i, x[i], y1[i], y2[i]);
            return;
        }
    }
    printf("checksum OK!\n");
}


// Host function to generate random FP32 values and allocate device memory
void generate_stimuli_and_run(const std::string& folder, int P, int D, int N, float coeff_min, float coeff_max, float x_min, float x_max, bool print_checksum) {
    // 1. Generate partition points (P+1 points)
    float* h_partition_points = (float*)malloc((P + 1) * sizeof(float));
    float partition_step = (x_max - x_min) / P; // Step size for each partition
    for (int i = 0; i <= P; ++i) {
        h_partition_points[i] = x_min + i * partition_step; // Partition point is incremented by step size
    }

    // 2. Generate random coefficients for each polynomial, within the range [coeff_min, coeff_max]
    float* h_coeffs = (float*)malloc(P * (D + 1) * sizeof(float));
    for (int i = 0; i < P * (D + 1); ++i) {
        h_coeffs[i] = coeff_min + static_cast<float>(rand()) / RAND_MAX * (coeff_max - coeff_min);
    }

    // 3. Generate N random x-values, within the range [x_min, x_max]
    float* h_x_values = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_x_values[i] = x_min + static_cast<float>(rand()) / RAND_MAX * (x_max - x_min-1.);
    }

    // 4. Allocate memory on the GPU
    float *d_x_values, *d_y_values, *d_coeffs, *d_partition_points;
    cudaError_t err_0 = cudaMalloc((void**)&d_x_values, N * sizeof(float));
    cudaError_t err_1 = cudaMalloc((void**)&d_y_values, N * sizeof(float));
    cudaError_t err_2 = cudaMalloc((void**)&d_coeffs, P * (D + 1) * sizeof(float));
    cudaError_t err_3 = cudaMalloc((void**)&d_partition_points, (P + 1) * sizeof(float));

    if (err_0 != cudaSuccess) {
        printf("CUDA malloc failed for d_x_values: %s\n", cudaGetErrorString(err_0));
        return;
    }
    if (err_1 != cudaSuccess) {
        printf("CUDA malloc failed for d_y_values: %s\n", cudaGetErrorString(err_1));
        return;
    }
    if (err_2 != cudaSuccess) {
        printf("CUDA malloc failed for d_coeffs: %s\n", cudaGetErrorString(err_2));
        return;
    }
    if (err_3 != cudaSuccess) {
        printf("CUDA malloc failed for d_partition_points: %s\n", cudaGetErrorString(err_3));
        return;
    }

    // 5. Copy data from host to device
    cudaMemcpy(d_x_values, h_x_values, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs, P * (D + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partition_points, h_partition_points, (P + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // 6.1 create host arrays
    float* h_y_values_ref = (float*)malloc(N * sizeof(float));
    float* h_y_values = (float*)malloc(N * sizeof(float));

    // 6. Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 6.0 Launch original kernel
    printf("\nLaunching Original ");
    auto time = hpc_gettime();
    evaluate_polynomials<<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                           d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    cudaDeviceSynchronize();
    if(print_checksum) {
        cudaMemcpy(h_y_values_ref, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values_ref, N);
    }


    /// Launches of AoS kernels
    // 6.11 Launch kernel AoS base
    printf("\nLaunching kernel AoS base ");
    time = hpc_gettime();
    evaluate_polynomials_AoS_unroll<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                             d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.12 Launch kernel AoS w/ basic unrolling
    printf("\nLaunching kernel AoS w/ basic unrolling ");
    time = hpc_gettime();
    if(P>=16)       evaluate_polynomials_AoS_unroll<16><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=8)   evaluate_polynomials_AoS_unroll<8><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else if(P>=4)   evaluate_polynomials_AoS_unroll<4><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else if(P>=2)   evaluate_polynomials_AoS_unroll<2><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else            evaluate_polynomials_AoS_unroll<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.13 Launch kernel AoS w/ base non-divergent unrolling
    printf("\nLaunching kernel AoS w/ base non-divergent unrolling ");
    time = hpc_gettime();
    if(P>=16)       evaluate_polynomials_AoS_unroll2<16><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                 d_coeffs, d_partition_points, P, D, N);
    else if(P>=8)   evaluate_polynomials_AoS_unroll2<8><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=4)   evaluate_polynomials_AoS_unroll2<4><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=2)   evaluate_polynomials_AoS_unroll2<2><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else            evaluate_polynomials_AoS_unroll2<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.14 Launch kernel AoS w/ base unrolling + register-arrays
    printf("\nLaunching kernel AoS w/ base unrolling + register-arrays ");
    time = hpc_gettime();
    if(P>=16)       { if((D+1)>=16)     evaluate_polynomials_AoS_unroll_reg<16,16><<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_AoS_unroll_reg<16,8> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_AoS_unroll_reg<16,4> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_AoS_unroll_reg<16,2> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_AoS_unroll_reg<16,1> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=8) { if((D+1)>=16)     evaluate_polynomials_AoS_unroll_reg<8,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_AoS_unroll_reg<8,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_AoS_unroll_reg<8,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_AoS_unroll_reg<8,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_AoS_unroll_reg<8,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=4) { if((D+1)>=16)     evaluate_polynomials_AoS_unroll_reg<4,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_AoS_unroll_reg<4,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_AoS_unroll_reg<4,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_AoS_unroll_reg<4,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_AoS_unroll_reg<4,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=2) {if((D+1)>=16)      evaluate_polynomials_AoS_unroll_reg<2,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_AoS_unroll_reg<2,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_AoS_unroll_reg<2,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_AoS_unroll_reg<2,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_AoS_unroll_reg<2,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else {        if((D+1)>=16)       evaluate_polynomials_AoS_unroll_reg<1,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_AoS_unroll_reg<1,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_AoS_unroll_reg<1,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_AoS_unroll_reg<1,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_AoS_unroll_reg<1,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
    }
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    /// Launches of SoA kernels
    // 6.21 Launch kernel SoA base
    printf("\nLaunching kernel SoA base ");
    time = hpc_gettime();
    evaluate_polynomials_SoA_unroll<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                             d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values_ref, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values_ref, N);
    }

    // 6.22 Launch kernel SoA w/ basic unrolling
    printf("\nLaunching kernel SoA w/ basic unrolling ");
    time = hpc_gettime();
    if(P>=16)       evaluate_polynomials_SoA_unroll<16><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=8)   evaluate_polynomials_SoA_unroll<8><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else if(P>=4)   evaluate_polynomials_SoA_unroll<4><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else if(P>=2)   evaluate_polynomials_SoA_unroll<2><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    else            evaluate_polynomials_SoA_unroll<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                               d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.23 Launch kernel SoA w/ base non-divergent unrolling
    printf("\nLaunching kernel SoA w/ base non-divergent unrolling ");
    time = hpc_gettime();
    if(P>=16)       evaluate_polynomials_SoA_unroll2<16><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                 d_coeffs, d_partition_points, P, D, N);
    else if(P>=8)   evaluate_polynomials_SoA_unroll2<8><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=4)   evaluate_polynomials_SoA_unroll2<4><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else if(P>=2)   evaluate_polynomials_SoA_unroll2<2><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    else            evaluate_polynomials_SoA_unroll2<1><<<blocksPerGrid, threadsPerBlock>>>(d_x_values, d_y_values, NULL,
                                                                                d_coeffs, d_partition_points, P, D, N);
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.24 Launch kernel SoA w/ base unrolling + register-arrays
    printf("\nLaunching kernel SoA w/ base unrolling + register-arrays ");
    time = hpc_gettime();
    if(P>=16) {     if((D+1)>=16)       evaluate_polynomials_SoA_unroll_reg<16,16><<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_unroll_reg<16,8> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_unroll_reg<16,4> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_unroll_reg<16,2> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_SoA_unroll_reg<16,1> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=8) { if((D+1)>=16)     evaluate_polynomials_SoA_unroll_reg<8,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_unroll_reg<8,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_unroll_reg<8,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_unroll_reg<8,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_SoA_unroll_reg<8,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=4) { if((D+1)>=16)     evaluate_polynomials_SoA_unroll_reg<4,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_unroll_reg<4,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_unroll_reg<4,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_unroll_reg<4,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_SoA_unroll_reg<4,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else if(P>=2) {if((D+1)>=16)      evaluate_polynomials_SoA_unroll_reg<2,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_unroll_reg<2,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_unroll_reg<2,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_unroll_reg<2,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_SoA_unroll_reg<2,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);

    } else {        if((D+1)>=16)       evaluate_polynomials_SoA_unroll_reg<1,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_unroll_reg<1,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_unroll_reg<1,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_unroll_reg<1,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else                evaluate_polynomials_SoA_unroll_reg<1,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
    }
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 6.25 Launch kernel SoA w/ base unrolling + shmem + register-arrays
    printf("\nLaunching kernel SoA w/ base unrolling + shmem + register-arrays ");
    time = hpc_gettime();
    if(N%256 >= P || N%256 == 0) {
    if(P<2)             { if((D+1)>=16)      evaluate_polynomials_SoA_shmem<1,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_shmem<1,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_shmem<1,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_shmem<1,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
//        else                evaluate_polynomials_SoA_shmem<1,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N); // D=0 cannot happen

    } else if(P<4)      { if((D+1)>=16)      evaluate_polynomials_SoA_shmem<2,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_shmem<2,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_shmem<2,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_shmem<2,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
//        else                evaluate_polynomials_SoA_shmem<2,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N); // D=0 cannot happen

    } else if(P<8)      { if((D+1)>=16)      evaluate_polynomials_SoA_shmem<4,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_shmem<4,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_shmem<4,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_shmem<4,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
//        else                evaluate_polynomials_SoA_shmem<4,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N); // D=0 cannot happen

    } else if(P<16)     { if((D+1)>=16)      evaluate_polynomials_SoA_shmem<8,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8)   evaluate_polynomials_SoA_shmem<8,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4)   evaluate_polynomials_SoA_shmem<8,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_shmem<8,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
//        else                evaluate_polynomials_SoA_shmem<8,1>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N); // D=0 cannot happen

    } else if(P<=128)     { if((D+1)>=16 && P<=16)      evaluate_polynomials_SoA_shmem<16,16> <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=8 && P<=32)   evaluate_polynomials_SoA_shmem<16,8>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=4 && P<=64)   evaluate_polynomials_SoA_shmem<16,4>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
        else if((D+1)>=2)   evaluate_polynomials_SoA_shmem<16,2>  <<<blocksPerGrid, threadsPerBlock>>> (d_x_values, d_y_values, NULL, d_coeffs, d_partition_points, P, D, N);
    } else
        printf("Shmem NOT DONE!!", hpc_gettime_elapsed(time));
    } else
        printf("N % 256 must be greater than or equal to P otherwise shmem algorithm don't have enough threads in last block!! (to be resolved)", hpc_gettime_elapsed(time));
    cudaDeviceSynchronize();
    printf("|--> %d ms ||", hpc_gettime_elapsed(time));
    if(print_checksum) {
        cudaMemcpy(h_y_values, d_y_values, N * sizeof(float), cudaMemcpyDeviceToHost);
        run_checksum(h_x_values, h_y_values_ref, h_y_values, N);
    }

    // 7. Clean up memory
    free(h_x_values);
    free(h_y_values);
    free(h_y_values_ref);
    free(h_coeffs);
    free(h_partition_points);
    cudaFree(d_x_values);
    cudaFree(d_y_values);
    cudaFree(d_coeffs);
    cudaFree(d_partition_points);
}

void test(int P, int D, int N, float coeff_min, float coeff_max, float x_min, float x_max, bool print_checksum) {
    // Check if the provided values for P, D, N, min, and max are valid
    if (P <= 0 || D <= 0 || N <= 0 || coeff_min >= coeff_max || x_min >= x_max) {
        printf("Error: Invalid input values.\n");
        return;
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("-----------------------Running Test %d %d %d %f %f %f %f-------------------------------\n",P, D, N, coeff_min, coeff_max, x_min, x_max);
    printf("---------------------------------------------------------------------------------------\n");
    cudaDeviceReset();

    // Generate stimuli and run the CUDA kernel, writing files to the specified folder
    generate_stimuli_and_run("", P, D, N, coeff_min, coeff_max, x_min, x_max, print_checksum);
    cudaDeviceReset();
    fflush(stdout);
}



int main(int argc, char** argv) {
    int N = 409600000;  // Number of input x-values
    float coeff_min = -10;  // Minimum value for coefficients
    float coeff_max = 10;  // Maximum value for coefficients
    float x_min = -5;  // Minimum value for x-values
    float x_max = 5;  // Maximum value for x-values
    bool print_checksum = false; // Whether to print a checksum - requires memory transfers

    // test all configuration from the heaviest to the lightest
    for(int D = 20; D > 0; D--) {
        for(int P = 2048; P > 1; P /= 2) {
            int d = (D<=10) ? D : D*10;
            test(P, d, N, coeff_min, coeff_max, x_min, x_max, print_checksum);
        }
    }
    return 0;

}

