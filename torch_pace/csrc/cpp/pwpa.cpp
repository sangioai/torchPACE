#include <torch/extension.h>

#include <vector>

namespace torch_pace {
  
typedef at::Half myFloat16;

// Host-size (CPU) Horner's scheme
template<typename D_TYPE>
D_TYPE horner_scheme_fma_cpu(const D_TYPE* coeffs, int ParamHwDegree, D_TYPE x, int jump=1) {
    D_TYPE result = coeffs[0];
    for (int i = 1; i <= ParamHwDegree; ++i) {
        result = result * x + coeffs[i*jump]; // FMA: result = result * x + coeffs[i]
    }
    return result;
}

// Optimized Kernel function to evaluate piecewise polynomials on SoA of coeffs
template<typename D_TYPE>
void evaluate_polynomials_AoS_cpu(const D_TYPE* x_values, D_TYPE* y_values, 
                                     const D_TYPE* coeffs, const D_TYPE* partition_points, int P, int D, int N) {
    for(int idx=0; idx<N; idx++) {
        D_TYPE x = x_values[idx];
        
        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Get the coefficients for the corresponding polynomial in the partition
        const D_TYPE* partition_coeffs = &coeffs[partition_index * (D + 1)];
        
        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_cpu(partition_coeffs, D, x);
    }
}


// Optimized Kernel function to evaluate piecewise polynomials on SoA of coeffs
template<typename D_TYPE>
void evaluate_polynomials_SoA_cpu(const D_TYPE* x_values, D_TYPE* y_values, 
                                     const D_TYPE* coeffs, const D_TYPE* partition_points, int P, int D, int N) {
    for(int idx=0; idx<N; idx++) {
        D_TYPE x = x_values[idx];
        
        // Determine the partition based on x-value
        int partition_index = 0;
        for (int i = 0; i < P; ++i) {
            if (x < partition_points[i + 1]) {
                partition_index = i;
                break;
            }
        }

        // Get the coefficients for the corresponding polynomial in the partition
        const D_TYPE* partition_coeffs = &coeffs[partition_index];
        
        // Evaluate polynomial using Horner's scheme with FMA
        y_values[idx] = horner_scheme_fma_cpu(partition_coeffs, D, x, P);
    }
}

//////////////////////////////////////////////////////////////////////////////
/// _pwpa_cpu Template Calls
//////////////////////////////////////////////////////////////////////////////

template<typename ATEN_TYPE, typename D_TYPE>
at::Tensor _pwpa_cpu_T(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
  // Retrieve pointers and configurations
  at::Tensor x_contig = x.contiguous();
  at::Tensor coeffs_contig = coeffs.contiguous();
  at::Tensor partition_points_contig = partition_points.contiguous();
  at::Tensor result = torch::empty(x_contig.sizes(), x_contig.options());
  const D_TYPE* x_ptr = (D_TYPE*)x_contig.data_ptr<ATEN_TYPE>();
  const D_TYPE* coeffs_ptr = (D_TYPE*)coeffs_contig.data_ptr<ATEN_TYPE>();
  const D_TYPE* partition_points_ptr = (D_TYPE*)partition_points_contig.data_ptr<ATEN_TYPE>();
  D_TYPE* result_ptr = (D_TYPE*)result.data_ptr<ATEN_TYPE>();
  int P = partition_points_contig.numel()-1;
  int D = (coeffs_contig.numel() / P) - 1;
  int N = x_contig.numel();
  // run CPU Kernel
  if(AoS)   evaluate_polynomials_AoS_cpu<D_TYPE>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  else      evaluate_polynomials_SoA_cpu<D_TYPE>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  return result;
}

at::Tensor _pwpa_cpu(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf);
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_CHECK(partition_points.dtype() == at::kFloat || partition_points.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(partition_points.device().type() == at::DeviceType::CPU);
  if(x.dtype() == at::kHalf) return _pwpa_cpu_T<myFloat16, myFloat16>(x, coeffs, partition_points, AoS);
  return _pwpa_cpu_T<float, float>(x, coeffs, partition_points, AoS);
}

//////////////////////////////////////////////////////////////////////////////
/// pwpa_cpu Template Calls
//////////////////////////////////////////////////////////////////////////////

at::Tensor pwpa_cpu(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf);
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_CHECK(partition_points.dtype() == at::kFloat || partition_points.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(partition_points.device().type() == at::DeviceType::CPU);
  if(x.dtype() == at::kHalf) return _pwpa_cpu_T<myFloat16, myFloat16>(x, coeffs, partition_points, AoS);
  return _pwpa_cpu_T<float, float>(x, coeffs, partition_points, AoS);
}

//////////////////////////////////////////////////////////////////////////////
/// aos2soa Template Calls
//////////////////////////////////////////////////////////////////////////////

template<typename ATEN_TYPE, typename D_TYPE>
at::Tensor aos2soa_cpu_T(const at::Tensor& coeffs, int64_t D) {
  // Retrieve pointers and configurations
  at::Tensor coeffs_contig = coeffs.contiguous();
  const D_TYPE* coeffs_ptr = (D_TYPE*)coeffs_contig.data_ptr<ATEN_TYPE>();
  // (im2col) Transform coeffs from AoS -> to SoA https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
  at::Tensor coeffs_SoA = torch::empty(coeffs_contig.sizes(), coeffs_contig.options());
  coeffs_SoA.copy_(coeffs_contig);
  coeffs_SoA.resize_({1, 1, coeffs_contig.numel()});
  coeffs_SoA = at::im2col(coeffs_SoA, { 1, D+1 }, { 1, 1 }, {0,0}, {1,D+1});
  return coeffs_SoA;
}

at::Tensor aos2soa_cpu(const at::Tensor& coeffs, int64_t D) {
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CPU);
  // type-specific kernel launch
  if(coeffs.dtype() == at::kHalf) return aos2soa_cpu_T<myFloat16, myFloat16>(coeffs, D);
  return aos2soa_cpu_T<float, float>(coeffs, D);
}

//////////////////////////////////////////////////////////////////////////////
/// CPU implementation registration
//////////////////////////////////////////////////////////////////////////////

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(torch_pace, m) {
  m.def("_pwpa(Tensor x, Tensor coeffs, Tensor partition_points, bool AoS) -> Tensor");
  m.def("pwpa(Tensor x, Tensor coeffs, Tensor partition_points, bool AoS) -> Tensor");
  m.def("aos2soa(Tensor coeffs, int D) -> Tensor");
}

// Registers CPU implementations for pwpa
TORCH_LIBRARY_IMPL(torch_pace, CPU, m) {
  m.impl("_pwpa", &_pwpa_cpu);
  m.impl("pwpa", &pwpa_cpu);
  m.impl("aos2soa", &aos2soa_cpu);
}

}
