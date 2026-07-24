// IRIS optimizer - Fused CUDA kernel
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int* address_as_uint = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        
        __half tmpres = __hadd(*reinterpret_cast<__half*>(&hsum), 
                               *reinterpret_cast<__half*>(&val));
        __half_raw tmpres_raw;
        tmpres_raw.x = __half_as_ushort(tmpres);
        
        unsigned int new_val = (size_t)address & 2 ? 
            (old & 0xffff) | (tmpres_raw.x << 16) :
            (old & 0xffff0000) | tmpres_raw.x;
        
        old = atomicCAS(address_as_uint, assumed, new_val);
    } while (assumed != old);
}
#else
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<const __half*>(&val));
}
#endif

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp(scalar_t val, scalar_t min_val, scalar_t max_val) {
    return fmin(static_cast<float>(max_val), fmax(static_cast<float>(min_val), static_cast<float>(val)));
}

// ---------------------------------------------------------------------------
// Standard IRIS kernel (no heavy-ball)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void iris_kernel(
    scalar_t* __restrict__ params,
    const scalar_t* __restrict__ grads,
    scalar_t* __restrict__ grad_estimates,
    scalar_t* __restrict__ variance_estimates,
    scalar_t* __restrict__ max_variance_estimates,
    scalar_t* __restrict__ innovation_residuals,
    const int64_t numel,
    const scalar_t lr,
    const scalar_t psi_inv_1,
    const scalar_t psi_inv_2,
    const scalar_t psi_inv_3,
    const scalar_t beta2,
    const scalar_t wd,
    const scalar_t eps,
    const scalar_t rho,
    const bool use_clipping,
    const bool amsgrad
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int64_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
        const scalar_t grad = grads[i];
        scalar_t param    = params[i];
        scalar_t grad_est = grad_estimates[i];
        scalar_t var_est  = variance_estimates[i];
        scalar_t innov_res = innovation_residuals[i];
        const scalar_t scalar_t1 = scalar_t(1);

        // innovation = g_t - g_est_{t-1}
        const scalar_t innovation = grad - grad_est;

        // g_est update via bias-corrected gain psi_inv_1
        grad_est = grad_est + psi_inv_1 * innovation;
        grad_estimates[i] = grad_est;

        // innov_res EMA of (innovation - innov_res)
        innov_res = innov_res + psi_inv_2 * (innovation - innov_res);
        innovation_residuals[i] = innov_res;

        // variance of corrected gradient (g + beta2 * I)^2
        const scalar_t corrected_grad = grad + beta2 * innovation;
        var_est = (scalar_t1 - psi_inv_3) * var_est
                + psi_inv_3 * corrected_grad * corrected_grad;
        variance_estimates[i] = var_est;

        // denom = sqrt(var) + eps
        scalar_t denom;
        if (amsgrad) {
            scalar_t max_var_est = max_variance_estimates[i];
            max_var_est = fmax(static_cast<float>(max_var_est),
                               static_cast<float>(var_est));
            max_variance_estimates[i] = max_var_est;
            denom = sqrt(static_cast<float>(max_var_est)) + eps;
        } else {
            denom = sqrt(static_cast<float>(var_est)) + eps;
        }

        // Decoupled weight decay
        if (wd != scalar_t(0)) {
            param *= (scalar_t1 - lr * wd);
        }

        // Parameter update
        const scalar_t numerator = grad_est + beta2 * innov_res;
        if (use_clipping) {
            const scalar_t denom_clipped = denom * rho;
            scalar_t update = numerator / denom_clipped;
            update = clamp(update, -scalar_t1, scalar_t1);
            param -= lr * update;
        } else {
            param -= lr * (numerator / denom);
        }

        params[i] = param;
    }
}

// ---------------------------------------------------------------------------
// Single-tensor entry points
// ---------------------------------------------------------------------------
std::tuple<double, double, double> iris_fused_cuda(
    at::Tensor params,
    at::Tensor grads,
    at::Tensor grad_estimates,
    at::Tensor variance_estimates,
    at::Tensor max_variance_estimates,
    at::Tensor innovation_residuals,
    at::Tensor step_tensor,
    double psi_1_prev,
    double psi_2_prev,
    double psi_3_prev,
    double lr,
    double beta1,
    double beta2,
    double beta3,
    double wd,
    double eps,
    double rho,
    bool use_clipping,
    bool amsgrad
) {
    TORCH_CHECK(params.is_cuda(), "params must be CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be CUDA tensor");
    TORCH_CHECK(params.numel() == grads.numel(), "params and grads must have same numel");
    
    const auto numel  = params.numel();
    const int threads = 256;
    const int blocks  = (int)std::min((numel + threads - 1) / threads, (int64_t)1024);
    
    step_tensor.add_(1);

    // Bias correction accumulators
    double psi_1_curr = beta1 * psi_1_prev + 1.0;
    double psi_2_curr = beta2 * psi_2_prev + 1.0;
    double psi_3_curr = beta3 * psi_3_prev + 1.0;

    double psi_inv_1 = 1.0 / psi_1_curr;
    double psi_inv_2 = 1.0 / psi_2_curr;
    double psi_inv_3 = 1.0 / psi_3_curr;

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(params.scalar_type(), "iris_kernel", [&] {
        iris_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            params.data_ptr<scalar_t>(),
            grads.data_ptr<scalar_t>(),
            grad_estimates.data_ptr<scalar_t>(),
            variance_estimates.data_ptr<scalar_t>(),
            amsgrad ? max_variance_estimates.data_ptr<scalar_t>() : nullptr,
            innovation_residuals.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(lr),
            static_cast<scalar_t>(psi_inv_1),
            static_cast<scalar_t>(psi_inv_2),
            static_cast<scalar_t>(psi_inv_3),
            static_cast<scalar_t>(beta2),
            static_cast<scalar_t>(wd),
            static_cast<scalar_t>(eps),
            static_cast<scalar_t>(rho),
            use_clipping,
            amsgrad
        );
    });

    return std::make_tuple(psi_1_curr, psi_2_curr, psi_3_curr);
}

// ---------------------------------------------------------------------------
// Multi-tensor entry point (called from Python fused API)
// ---------------------------------------------------------------------------
std::tuple<double, double, double> iris_multi_tensor_fused_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> grad_estimates,
    std::vector<at::Tensor> variance_estimates,
    std::vector<at::Tensor> max_variance_estimates,
    std::vector<at::Tensor> innovation_residuals,
    std::vector<at::Tensor> step_tensors,
    double psi_1_prev,
    double psi_2_prev,
    double psi_3_prev,
    double lr,
    double beta1,
    double beta2,
    double beta3,
    double wd,
    double eps,
    double rho,
    bool use_clipping,
    bool amsgrad
) {
    const size_t n_tensors = params.size();

    // Bias correction accumulators
    double psi_1_curr = beta1 * psi_1_prev + 1.0;
    double psi_2_curr = beta2 * psi_2_prev + 1.0;
    double psi_3_curr = beta3 * psi_3_prev + 1.0;

    double psi_inv_1 = 1.0 / psi_1_curr;
    double psi_inv_2 = 1.0 / psi_2_curr;
    double psi_inv_3 = 1.0 / psi_3_curr;

    for (size_t i = 0; i < n_tensors; i++) {
        step_tensors[i].add_(1);

        const auto numel  = params[i].numel();
        const int threads = 256;
        const int blocks  = (int)std::min((numel + threads - 1) / threads, (int64_t)1024);

        auto stream = at::cuda::getCurrentCUDAStream();

        at::Tensor max_var_est = amsgrad ? max_variance_estimates[i] : at::Tensor();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(params[i].scalar_type(), "iris_kernel", [&] {
            iris_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                params[i].data_ptr<scalar_t>(),
                grads[i].data_ptr<scalar_t>(),
                grad_estimates[i].data_ptr<scalar_t>(),
                variance_estimates[i].data_ptr<scalar_t>(),
                amsgrad ? max_var_est.data_ptr<scalar_t>() : nullptr,
                innovation_residuals[i].data_ptr<scalar_t>(),
                numel,
                static_cast<scalar_t>(lr),
                static_cast<scalar_t>(psi_inv_1),
                static_cast<scalar_t>(psi_inv_2),
                static_cast<scalar_t>(psi_inv_3),
                static_cast<scalar_t>(beta2),
                static_cast<scalar_t>(wd),
                static_cast<scalar_t>(eps),
                static_cast<scalar_t>(rho),
                use_clipping,
                amsgrad
            );
        });
    }

    return std::make_tuple(psi_1_curr, psi_2_curr, psi_3_curr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iris_fused_cuda", &iris_fused_cuda,
          "IRIS fused optimizer single-tensor (CUDA)");
    m.def("iris_multi_tensor_fused_cuda", &iris_multi_tensor_fused_cuda,
          "IRIS multi-tensor fused optimizer (CUDA)");
}