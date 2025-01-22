#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace torch_pace {
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Device-side (GPU) Horner's scheme
template <typename TYPE=float>
__device__ float horner_scheme_fma_gpu(const float* coeffs, int ParamHwDegree, float x, int jump=1) {
    float result = coeffs[0];
    for (int i = 1; i <= ParamHwDegree; ++i) {
        result = __fmaf_rn(result, x, coeffs[i*jump]); // FMA: result = result * x + coeffs[i]
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////////
/// AoS kernels
//////////////////////////////////////////////////////////////////////////////

// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template<typename TYPE=float>
__global__ void evaluate_polynomials_AoS_gpu(const TYPE* x_values, TYPE* y_values,
                                      const TYPE* coeffs, const TYPE* partition_points, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        TYPE x = x_values[idx];

        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }
        
        // Get the coefficients for the corresponding polynomial in the partition
        const TYPE* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<TYPE>(partition_coeffs, D, x);
    }
}


// Kernel function to evaluate piecewise polynomials and calculate partition IDs
template<typename TYPE=float, int UNROLL_SIZE>
__global__ void evaluate_polynomials_AoS_unroll_gpu(const float* x_values, float* y_values,
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

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index * (D + 1)];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<float>(partition_coeffs, D, x);
    }
}


//////////////////////////////////////////////////////////////////////////////
/// SoA kernels
//////////////////////////////////////////////////////////////////////////////


// Kernel function to evaluate piecewise polynomials and calculate partition
template<typename TYPE=float>
__global__ void evaluate_polynomials_SoA_gpu(const float* x_values, float* y_values,
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

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<float>(partition_coeffs, D, x, P);
    }
}

// Kernel function to evaluate piecewise polynomials and calculate partition
template<typename TYPE=float, int UNROLL_SIZE>
__global__ void evaluate_polynomials_SoA_unroll_gpu(const float* x_values, float* y_values, 
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

        // Get the coefficients for the corresponding polynomial in the partition
        const float* partition_coeffs = &coeffs[partition_index];

        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_gpu<float>(partition_coeffs, D, x, P);
    }
}

}
