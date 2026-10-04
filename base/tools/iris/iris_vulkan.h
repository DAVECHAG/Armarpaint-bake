/*
 * Iris Vulkan Acceleration
 *
 * GPU-accelerated matrix multiplication using Vulkan compute shaders.
 * Targets cross-platform GPUs (initially Linux). Provides a GEMM-offload
 * backend: the transformer/VAE/text-encoder run their CPU orchestration
 * path, but the heavy matrix multiplies are dispatched to the GPU.
 *
 * Only the Vulkan loader (libvulkan) is required at link time; no other
 * external libraries.
 */

#ifndef IRIS_VULKAN_H
#define IRIS_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize Vulkan acceleration: enumerate a compute-capable device,
 * create the queue, command pool, and GEMM pipelines.
 * Returns 1 on success, 0 if Vulkan is unavailable. Safe to call repeatedly.
 */
int iris_vulkan_init(void);

/*
 * Check whether Vulkan acceleration is available and initialized.
 */
int iris_vulkan_available(void);

/*
 * Human-readable name of the selected GPU (valid after init).
 */
const char *iris_vulkan_device_name(void);

/*
 * Release all Vulkan resources (buffers, caches, pipelines, device).
 */
void iris_vulkan_cleanup(void);

/*
 * Free all GPU-resident cached weight buffers (VRAM) while keeping the device
 * and pipelines alive. Cached weights re-upload lazily on next use. Used to
 * reclaim transformer weights before VAE decode; call only between batches.
 */
void iris_vulkan_release_weight_cache(void);

/*
 * GPU matrix multiply (generic, f32):
 *   C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C[M,N]
 * transpose_a / transpose_b select op(). B is uploaded every call, so this
 * is safe for dynamic matrices (e.g. attention K/V temporaries).
 */
void iris_vulkan_sgemm(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                       float *C, int ldc);

/*
 * Same as iris_vulkan_sgemm() but caches the B matrix on the GPU keyed by
 * its pointer. Use only when B is immutable across calls (model weights).
 */
void iris_vulkan_sgemm_cached(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                              float *C, int ldc);

/*
 * GPU matrix multiply with bf16 weights (B is bfloat16, A and C are f32):
 *   C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C[M,N]
 * B is cached on the GPU by pointer (weights are immutable).
 */
void iris_vulkan_sgemm_bf16(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const uint16_t *B_bf16, int ldb,
                            float beta, float *C, int ldc);

/*
 * GPU matrix multiply with GGML Q8_0 block-quantized weights (B is Q8_0,
 * A and C are f32), dequantized in-shader:
 *   C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C[M,N]
 * B points at the raw 34-byte-per-32-element block stream and is cached on the
 * GPU by pointer (weights are immutable). ldb is in logical elements.
 */
void iris_vulkan_sgemm_q8(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const void *B_q8, int ldb, float beta,
                          float *C, int ldc);

/* ========================================================================
 * Resident VAE-decode tensor API.
 *
 * Mirrors the subset of the Metal iris_gpu_* surface used by vae_decode_gpu
 * in iris_vae.c, so that path compiles and runs GPU-resident under Vulkan.
 * Tensors are device-local (VRAM) on discrete GPUs; data stays on the GPU
 * between ops. Wrap a sequence in iris_gpu_batch_begin/end to fold many
 * dispatches into a single submit.
 * ======================================================================== */

typedef struct iris_gpu_tensor *iris_gpu_tensor_t;

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements);
iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements);
void              iris_gpu_tensor_free(iris_gpu_tensor_t tensor);
void              iris_gpu_tensor_read(iris_gpu_tensor_t tensor, float *out);
void              iris_gpu_tensor_write(iris_gpu_tensor_t tensor, const float *data);

void iris_gpu_copy_f32(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n);

iris_gpu_tensor_t iris_gpu_conv2d_f32(iris_gpu_tensor_t x, const float *weight, const float *bias, int batch, int in_ch, int out_ch, int H, int W, int kH,
                                      int kW, int stride, int padding);

void iris_gpu_group_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *gamma, const float *beta, int batch, int channels, int spatial,
                             int num_groups, float eps);

void iris_gpu_swish_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n);
void iris_gpu_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n);

iris_gpu_tensor_t iris_gpu_upsample_nearest_2x_f32(iris_gpu_tensor_t x, int channels, int H, int W);

/* In-place-safe LeakyReLU: out[i] = x[i] >= 0 ? x[i] : slope*x[i]. */
void iris_gpu_leaky_relu_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n, float slope);

/* Scaled residual add: out[i] = scale*a[i] + b[i]. */
void iris_gpu_scale_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, float scale, int n);

void iris_gpu_batch_begin(void);
void iris_gpu_batch_end(void);

/* ========================================================================
 * Resident Qwen3 text-encoder ops.
 *
 * Fully GPU-resident forward path: activations stay in f32 device tensors
 * across all layers; weights are bf16 (cached by pointer) and applied
 * in-shader. Mirrors the subset of the Metal iris_gpu_*_bf16 surface that
 * qwen3_forward uses, but keeps activations in f32 for simplicity. Wrap a
 * whole forward pass in iris_gpu_batch_begin/end to fold every layer's
 * dispatches into a single submit (one GPU sync instead of ~150).
 *
 * Returns 1 if the resident ops are available (pipelines built), else 0.
 * ======================================================================== */
int iris_vk_qwen_available(void);

/* out[seq, out_dim] = x[seq, in_dim] @ weight[out_dim, in_dim]^T.
 * weight is bf16, cached on the GPU by pointer (immutable model weight). */
void iris_vk_qwen_linear(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *weight_bf16, int seq, int in_dim, int out_dim);

/* Same as iris_vk_qwen_linear but weight is GGML Q8_0 block-quantized,
 * dequantized in-shader. weight_q8 points at the raw 34-byte-per-32-element
 * block stream, cached on the GPU by pointer (immutable model weight). */
void iris_vk_qwen_linear_q8(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const void *weight_q8, int seq, int in_dim, int out_dim);

/* Per-row RMSNorm over hidden, scaled by a bf16 weight[hidden]. */
void iris_vk_qwen_rms_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *weight_bf16, int seq, int hidden, float eps);

/* Per-head RMSNorm (over head_dim) of t in place, bf16 weight[head_dim]. */
void iris_vk_qwen_head_rms_norm(iris_gpu_tensor_t t, const uint16_t *weight_bf16, int seq, int num_heads, int head_dim, float eps);

/* Apply RoPE to q and k in place using f32 cos/sin tables (cached by pointer). */
void iris_vk_qwen_rope(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_table, const float *sin_table, int seq, int num_q_heads, int num_kv_heads,
                       int head_dim);

/* GQA causal attention with padding mask (f32 tensor, 1.0 valid / 0.0 pad). */
void iris_vk_qwen_attention(iris_gpu_tensor_t out, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t mask, int seq,
                            int num_heads, int num_kv_heads, int head_dim, float scale);

/* SwiGLU: gate = silu(gate) * up, in place on gate. */
void iris_vk_qwen_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n);

/* ========================================================================
 * Resident transformer (denoising) ops.
 *
 * Fully GPU-resident forward path: activations stay in f32 device tensors
 * across all blocks; immutable weights are bf16 (cached by pointer), and
 * dynamic per-block f32 vectors (modulation-fused norm weights, gates, AdaLN
 * shift/scale) are uploaded fresh per call. Mirrors the subset of the Metal
 * iris_gpu_* surface the resident transformer path uses, so it compiles and
 * runs under Vulkan.
 *
 * Returns 1 if the resident pipelines are available, else 0.
 * ======================================================================== */
int iris_vk_res_available(void);

/* Allocate an uninitialized bf16 (2 bytes/element) resident tensor. */
iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements);

/* Host pointer to a tensor's f32 contents (valid until the next op touches it
 * or it is freed). On discrete GPUs this downloads into a per-tensor shadow. */
float *iris_gpu_tensor_data(iris_gpu_tensor_t tensor);

/* No-op under Vulkan (tensors are freed explicitly; there is no pool). */
void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t tensor, int persistent);

/* out[seq, out_dim] = x @ W^T + b, with f32 weight W[out_dim, in_dim] and
 * optional f32 bias[out_dim] (NULL for none). Returns a new tensor. */
iris_gpu_tensor_t iris_gpu_linear(iris_gpu_tensor_t x, const float *W, const float *b, int seq_len, int in_dim, int out_dim);

/* out[seq, out_dim] = x @ W^T with bf16 weight W[out_dim, in_dim] (cached). */
iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim);
int               iris_gpu_linear_bf16_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim);

/* Same as iris_gpu_linear_bf16_into but W is a GGML Q8_0 block stream,
 * dequantized in-shader (cached on the GPU by pointer). */
int iris_gpu_linear_q8_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const void *W_q8, int seq_len, int in_dim, int out_dim);

/* Per-row RMSNorm scaled by an f32 weight[hidden]. */
void iris_gpu_rms_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *weight, int seq, int hidden, float eps);

/* Per-head RMSNorm of q and k in place, f32 weights[head_dim]. */
void iris_gpu_qk_rms_norm(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps);

/* Consecutive-pair RoPE applied to q and k in place using f32 cos/sin tables
 * [seq, head_dim] (cached by pointer). */
void iris_gpu_rope_single_pair_f32(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim);

/* Split a fused projection into separate q/k/v and/or gate/up tensors.
 * hidden>0 splits the leading 3*hidden into q,k,v; mlp_hidden>0 splits the
 * trailing 2*mlp_hidden into gate,up. */
void iris_gpu_split_qkv_mlp(iris_gpu_tensor_t fused, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t gate,
                            iris_gpu_tensor_t up, int seq, int hidden, int mlp_hidden);

/* SwiGLU: gate = silu(gate) * up, in place on gate. */
void iris_gpu_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n);

/* AdaLN: out = (1 + scale[i]) * layernorm(x)[s,i] + shift[i]. */
void iris_gpu_adaln_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *shift, const float *scale, int seq, int hidden, float eps);

/* Gated residual: out[s,i] += gate[i] * proj[s,i]. */
void iris_gpu_gated_add(iris_gpu_tensor_t out, const float *gate, iris_gpu_tensor_t proj, int seq, int hidden);

/* Full (bidirectional) f32 self-attention; returns 1 on success. */
int iris_gpu_attention_fused(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                             int head_dim, float scale);

/* bf16 attention paths are not implemented under Vulkan; they return 0 so the
 * caller falls back to the f32 path above. */
int iris_gpu_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                            int head_dim, float scale);
int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                                  int head_dim, float scale);
int iris_gpu_convert_f32_to_bf16_into(iris_gpu_tensor_t bf16_out, iris_gpu_tensor_t f32_in);
int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t f32_out, iris_gpu_tensor_t bf16_in);

/* GPU blit copy for f32 tensors with element offsets. */
void iris_gpu_copy_region_f32(iris_gpu_tensor_t dst, size_t dst_offset, iris_gpu_tensor_t src, size_t src_offset, size_t n);

/* ========================================================================
 * Resident Flux transformer (denoising) ops.
 *
 * The Flux double/single blocks reuse the resident op surface above
 * (adaln/qk-rmsnorm/rope/attention/linear/gated-add/silu) for f32 activations.
 * The one extra primitive they need is a strided row-block copy, used to split
 * the fused [Q,K,V,gate,up] single-block projection into separate streams and
 * to assemble the [attn|mlp] concat feeding the output projection.
 *
 * Returns 1 if the Flux resident pipeline is available (rowcopy + the resident
 * ops it depends on), else 0.
 * ======================================================================== */
int iris_vk_flux_available(void);

/* dst[s*dst_stride + dst_off + e] = src[s*src_stride + src_off + e],
 * for e in [0, w), s in [0, seq). Strides/offsets are in elements (f32). */
void iris_gpu_row_copy_f32(iris_gpu_tensor_t dst, int dst_stride, int dst_off, iris_gpu_tensor_t src, int src_stride, int src_off, int seq, int w);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_VULKAN_H */
