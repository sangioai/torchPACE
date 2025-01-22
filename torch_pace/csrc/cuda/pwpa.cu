#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "pwpa_half.cuh"
#include "pwpa_float.cuh"

namespace torch_pace {

typedef at::Half myFloat16;

//////////////////////////////////////////////////////////////////////////////
/// _pwpa_cuda Template Calls
//////////////////////////////////////////////////////////////////////////////

template<typename ATEN_TYPE, typename D_TYPE>
at::Tensor _pwpa_cuda_T(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
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
  // run CUDA Kernel
  if(AoS)   evaluate_polynomials_AoS_gpu<D_TYPE><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  else      evaluate_polynomials_SoA_gpu<D_TYPE><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  return result;
}

at::Tensor _pwpa_cuda(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf);
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_CHECK(partition_points.dtype() == at::kFloat || partition_points.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(partition_points.device().type() == at::DeviceType::CUDA);
  if(x.dtype() == at::kHalf) return _pwpa_cuda_T<myFloat16, __half>(x, coeffs, partition_points, AoS);
  return _pwpa_cuda_T<float, float>(x, coeffs, partition_points, AoS);
}

//////////////////////////////////////////////////////////////////////////////
/// pwpa_cuda Template Calls
//////////////////////////////////////////////////////////////////////////////


template<typename ATEN_TYPE, typename D_TYPE>
at::Tensor pwpa_cuda_T(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
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
  // run CUDA Kernel
  if(AoS) {
    if(P>=16)       evaluate_polynomials_AoS_unroll_gpu<D_TYPE,16><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=8)   evaluate_polynomials_AoS_unroll_gpu<D_TYPE,8><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=4)   evaluate_polynomials_AoS_unroll_gpu<D_TYPE,4><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=2)   evaluate_polynomials_AoS_unroll_gpu<D_TYPE,2><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else            evaluate_polynomials_AoS_unroll_gpu<D_TYPE,1><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  } else {
    if(P>=16)       evaluate_polynomials_SoA_unroll_gpu<D_TYPE,16><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=8)   evaluate_polynomials_SoA_unroll_gpu<D_TYPE,8><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=4)   evaluate_polynomials_SoA_unroll_gpu<D_TYPE,4><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else if(P>=2)   evaluate_polynomials_SoA_unroll_gpu<D_TYPE,2><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
    else            evaluate_polynomials_SoA_unroll_gpu<D_TYPE,1><<<(N+255)/256, 256>>>(x_ptr, result_ptr, coeffs_ptr, partition_points_ptr, P, D, N);
  }
  return result;
}

at::Tensor pwpa_cuda(const at::Tensor& x, const at::Tensor& coeffs, const at::Tensor& partition_points, bool AoS = true) {
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf);
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_CHECK(partition_points.dtype() == at::kFloat || partition_points.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(partition_points.device().type() == at::DeviceType::CUDA);
  if(x.dtype() == at::kHalf) return pwpa_cuda_T<myFloat16, __half>(x, coeffs, partition_points, AoS);
  return pwpa_cuda_T<float, float>(x, coeffs, partition_points, AoS);
}

//////////////////////////////////////////////////////////////////////////////
/// aos2soa Template Calls
//////////////////////////////////////////////////////////////////////////////

template<typename ATEN_TYPE, typename D_TYPE>
at::Tensor aos2soa_cuda_T(const at::Tensor& coeffs, int64_t D) {
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

at::Tensor aos2soa_cuda(const at::Tensor& coeffs, int64_t D) {
  TORCH_CHECK(coeffs.dtype() == at::kFloat || coeffs.dtype() == at::kHalf);
  TORCH_INTERNAL_ASSERT(coeffs.device().type() == at::DeviceType::CUDA);
  // type-specific kernel launch
  if(coeffs.dtype() == at::kHalf) return aos2soa_cuda_T<myFloat16, __half>(coeffs, D);
  return aos2soa_cuda_T<float, float>(coeffs, D);
}

//////////////////////////////////////////////////////////////////////////////
/// CUDA implementation registration
//////////////////////////////////////////////////////////////////////////////

// Registers CUDA implementations for pwpa
TORCH_LIBRARY_IMPL(torch_pace, CUDA, m) {
  m.impl("_pwpa", &_pwpa_cuda);
  m.impl("pwpa", &pwpa_cuda);
  m.impl("aos2soa", &aos2soa_cuda);
}

}
