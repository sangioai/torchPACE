#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace torch_pace {
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Device-side (GPU) Horner's scheme
template <typename TYPE=__half>
__device__ __half horner_scheme_fma_gpu(const __half* coeffs, int ParamHwDegree, __half x, int jump=1) {
    __half result = coeffs[0];
    for (int i = 1; i <= ParamHwDegree; ++i) {
        result = __hfma(result, x, coeffs[i*jump]); // FMA: result = result * x + coeffs[i]
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////////
/// AoS kernels
//////////////////////////////////////////////////////////////////////////////

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template<typename TYPE=__half>
__global__ void evaluate_polynomials_AoS_gpu(const __half* x_values, __half* y_values,
                                      const __half* coeffs, const __half* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        __half x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (__hlt(x, partition_points[i + 1])) {
                partition_index = i;
                break;
            }
        }
        
        // Get the coefficients for the corresponding polynomial in the partition
        const __half* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<__half>(partition_coeffs, D, x);
    }
}


// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template<typename TYPE=__half, int UNROLL_SIZE>
__global__ void evaluate_polynomials_AoS_unroll_gpu(const __half* x_values, __half* y_values,
                                                const __half* coeffs, const __half* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        __half x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (__hlt(x, partition_points[i + 1])) {
                partition_index = i;
                break;
            }
        }

        // Get the coefficients for the corresponding polynomial in the partition
        const __half* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<__half>(partition_coeffs, D, x);
    }
}


//////////////////////////////////////////////////////////////////////////////
/// SoA kernels
//////////////////////////////////////////////////////////////////////////////


// Kernel function to evaluate piecewise polynomials and calculate partition
template<typename TYPE=__half>
__global__ void evaluate_polynomials_SoA_gpu(const __half* x_values, __half* y_values,
                                                const __half* coeffs, const __half* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        __half x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (__hlt(x, partition_points[i + 1])) {
                partition_index = i;
                break;
            }
        }

        // Get the coefficients for the corresponding polynomial in the partition
        const __half* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<__half>(partition_coeffs, D, x, P);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition
template<typename TYPE=__half, int UNROLL_SIZE>
__global__ void evaluate_polynomials_SoA_unroll_gpu(const __half* x_values, __half* y_values, 
                                                const __half* coeffs, const __half* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        __half x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
#pragma unroll UNROLL_SIZE
        for (int i = 0; i < P; ++i) {
            if (__hlt(x, partition_points[i + 1])) {
                partition_index = i;
                break;
            }
        }

        // Get the coefficients for the corresponding polynomial in the partition
        const __half* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<__half>(partition_coeffs, D, x, P);
    }
}

}