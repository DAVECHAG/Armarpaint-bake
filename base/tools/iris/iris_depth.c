/*
 * iris_depth.c - Depth Anything 3 (DA3MONO-LARGE) monocular depth estimation
 *
 * Pure-C implementation of the Depth Anything 3 "mono" depth estimator
 * (DA3MONO-LARGE). The network is a DINOv2 ViT-L/14 backbone followed by a
 * DPT-style head, in the original (non-HuggingFace) checkpoint layout used by
 * the Depth-Anything-3 release:
 *
 *   backbone (DINOv2 ViT-L/14, "model.backbone.pretrained.*"):
 *     - patch embed: 14x14 stride-14 conv, 3 -> 1024, on a 518x518 input
 *       (37x37 = 1369 patches), prepend cls token, add position embeddings
 *     - 24 transformer blocks (pre-norm, fused-QKV MHSA + GELU MLP, LayerScale)
 *     - the outputs of blocks {4,11,17,23} are taken as multi-scale features,
 *       each passed through the final backbone LayerNorm
 *
 *   head (DPT, "model.head.*"):
 *     - projects: drop cls, reshape to 37x37, 1x1 project to {256,512,1024,1024}
 *       channels, then resize_layers (transpose-conv x4 / x2, identity, conv
 *       stride2) to scales {148,74,37,19}
 *     - scratch.layer{1..4}_rn: 3x3 convs map each to 256 channels
 *     - scratch.refinenet{4..1}: feature-fusion (RefineNet) merges coarse->fine
 *       to 296x296
 *     - scratch.output_conv1: 3x3 conv 256->128, bilinear upsample to 518x518,
 *       output_conv2: 3x3 conv 128->32 + ReLU, 1x1 conv 32->1 + ReLU
 *
 * (The model also carries a parallel "sky_output_conv2" branch predicting a sky
 * mask; it is not needed for the depth map and is ignored here.)
 *
 * The output is resized back to the input resolution and min/max normalized to
 * a single-channel grayscale image (brighter = nearer).
 *
 * All weights are F32 in the memory-mapped safetensors file (zero-copy). The
 * transformer linear/attention matrix multiplies are dispatched to the Metal
 * or Vulkan GEMM backend when available; convolutions (small spatially) run on
 * the CPU.
 *
 * Note: the official image processor resizes with bicubic resampling; here we
 * use bilinear, which yields a visually equivalent relative-depth map.
 */

#include "iris_depth.h"
#include "iris_safetensors.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_METAL)
#include "iris_metal.h"
#elif defined(USE_VULKAN)
#include "iris_vulkan.h"
#endif

/* ------------------------------------------------------------------ */
/* Architecture constants (Depth Anything 3 mono - Large / ViT-L/14)   */
/* ------------------------------------------------------------------ */
#define DA_PATCH       14
#define DA_IMG         518                 /* model input is DA_IMG x DA_IMG */
#define DA_GRID        (DA_IMG / DA_PATCH) /* 37 patches per side    */
#define DA_NPATCH      (DA_GRID * DA_GRID) /* 1369                   */
#define DA_TOKENS      (DA_NPATCH + 1)     /* 1370 (+ cls)           */
#define DA_HIDDEN      1024
#define DA_LAYERS      24
#define DA_HEADS       16
#define DA_HEAD_DIM    (DA_HIDDEN / DA_HEADS) /* 64                    */
#define DA_MLP         4096
#define DA_LN_EPS      1e-6f
#define DA_FUSION_CH   256
#define DA_HEAD_CH     (DA_FUSION_CH / 2) /* head conv1 output (128) */
#define DA_HEAD_HIDDEN 32                 /* head conv2 output       */

/* Tensor-name roots for the DA3 checkpoint layout. */
#define BB   "model.backbone.pretrained"
#define HEAD "model.head"

/* Backbone blocks whose outputs feed the head (0-indexed). */
static const int DA_OUT_LAYERS[4] = {4, 11, 17, 23};
/* Per-feature channel widths after the 1x1 projects. */
static const int DA_PROJ_CH[4] = {256, 512, 1024, 1024};

/* ImageNet normalization (matches the model's image processor). */
static const float DA_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float DA_STD[3]  = {0.229f, 0.224f, 0.225f};

struct iris_depth {
	safetensors_file_t *sf;
	int                 tileable; /* remove the perspective tilt for seamless top-down maps */
};

/* ================================================================== */
/* Weight access (zero-copy; all tensors are F32 in the mmap'd file)  */
/* ================================================================== */

static const float *da_tensor(iris_depth_t *m, const char *name) {
	const safetensor_t *t = safetensors_find(m->sf, name);
	if (!t) {
		fprintf(stderr, "DepthAnything3: missing tensor '%s'\n", name);
		return NULL;
	}
	if (t->dtype != DTYPE_F32) {
		fprintf(stderr, "DepthAnything3: tensor '%s' is not F32\n", name);
		return NULL;
	}
	return (const float *)safetensors_data(m->sf, t);
}

/* printf-style tensor lookup. */
static const float *da_tensorf(iris_depth_t *m, const char *fmt, ...) {
	char    name[256];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(name, sizeof(name), fmt, ap);
	va_end(ap);
	return da_tensor(m, name);
}

/* ================================================================== */
/* GEMM helpers (Vulkan-accelerated when available)                   */
/* ================================================================== */

/* C[M,N] = scale * A[M,K] @ B[N,K]^T   (row-wise dot products).
 * cache_b: weight matrix is immutable -> cache it on the GPU by pointer. */
static void gemm_nt(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, float scale, int cache_b) {
#if defined(USE_METAL)
	if (iris_metal_available()) {
		if (cache_b)
			iris_metal_sgemm_cached(0, 1, M, N, K, scale, A, lda, B, ldb, 0.0f, C, ldc);
		else
			iris_metal_sgemm(0, 1, M, N, K, scale, A, lda, B, ldb, 0.0f, C, ldc);
		return;
	}
#elif defined(USE_VULKAN)
	if (iris_vulkan_available()) {
		if (cache_b)
			iris_vulkan_sgemm_cached(0, 1, M, N, K, scale, A, lda, B, ldb, 0.0f, C, ldc);
		else
			iris_vulkan_sgemm(0, 1, M, N, K, scale, A, lda, B, ldb, 0.0f, C, ldc);
		return;
	}
#endif
	(void)cache_b;
	for (int i = 0; i < M; i++) {
		const float *a = A + (size_t)i * lda;
		float       *c = C + (size_t)i * ldc;
		for (int j = 0; j < N; j++) {
			const float *b = B + (size_t)j * ldb;
			float        s = 0.0f;
			for (int k = 0; k < K; k++)
				s += a[k] * b[k];
			c[j] = s * scale;
		}
	}
}

/* C[M,N] = A[M,K] @ B[K,N]  (B not transposed). */
static void gemm_nn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
#if defined(USE_METAL)
	if (iris_metal_available()) {
		iris_metal_sgemm(0, 0, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
		return;
	}
#elif defined(USE_VULKAN)
	if (iris_vulkan_available()) {
		iris_vulkan_sgemm(0, 0, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
		return;
	}
#endif
	for (int i = 0; i < M; i++) {
		float *c = C + (size_t)i * ldc;
		for (int j = 0; j < N; j++)
			c[j] = 0.0f;
		const float *a = A + (size_t)i * lda;
		for (int k = 0; k < K; k++) {
			float        av = a[k];
			const float *b  = B + (size_t)k * ldb;
			for (int j = 0; j < N; j++)
				c[j] += av * b[j];
		}
	}
}

/* out[M,N] = x[M,K] @ W[N,K]^T + bias[N]  (bias may be NULL). */
static void linear(const float *x, int M, int K, const float *W, const float *bias, int N, float *out) {
	gemm_nt(M, N, K, x, K, W, K, out, N, 1.0f, 1);
	if (bias) {
		for (int i = 0; i < M; i++) {
			float *o = out + (size_t)i * N;
			for (int j = 0; j < N; j++)
				o[j] += bias[j];
		}
	}
}

/* ================================================================== */
/* Elementwise / norm primitives                                       */
/* ================================================================== */

/* Row-wise LayerNorm over `dim` (per token), affine weight+bias. */
static void layer_norm(const float *in, int rows, int dim, const float *gamma, const float *beta, float *out) {
	for (int r = 0; r < rows; r++) {
		const float *x    = in + (size_t)r * dim;
		float       *o    = out + (size_t)r * dim;
		float        mean = 0.0f;
		for (int i = 0; i < dim; i++)
			mean += x[i];
		mean /= dim;
		float var = 0.0f;
		for (int i = 0; i < dim; i++) {
			float d = x[i] - mean;
			var += d * d;
		}
		var /= dim;
		float inv = 1.0f / sqrtf(var + DA_LN_EPS);
		for (int i = 0; i < dim; i++)
			o[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
	}
}

/* Exact GELU (erf form), in place. */
static void gelu(float *buf, size_t n) {
	const float inv_sqrt2 = 0.70710678118654752440f;
	for (size_t i = 0; i < n; i++)
		buf[i] = 0.5f * buf[i] * (1.0f + erff(buf[i] * inv_sqrt2));
}

static void relu_inplace(float *buf, size_t n) {
	for (size_t i = 0; i < n; i++)
		if (buf[i] < 0.0f)
			buf[i] = 0.0f;
}

/* ================================================================== */
/* Convolutions (CPU)                                                  */
/* ================================================================== */

/* General 2D convolution, planar CHW.
 *   in:     [Cin, H, W]
 *   weight: [Cout, Cin, kH, kW]
 *   bias:   [Cout] or NULL
 *   out:    [Cout, OH, OW] with OH=(H+2*pad-kH)/stride+1
 */
static void conv2d(const float *in, int Cin, int H, int W, const float *weight, const float *bias, int Cout, int kH, int kW, int stride, int pad, float *out) {
	int          OH        = (H + 2 * pad - kH) / stride + 1;
	int          OW        = (W + 2 * pad - kW) / stride + 1;
	const size_t in_plane  = (size_t)H * W;
	const size_t out_plane = (size_t)OH * OW;

	for (int oc = 0; oc < Cout; oc++) {
		float *op = out + (size_t)oc * out_plane;
		float  b  = bias ? bias[oc] : 0.0f;
		for (size_t i = 0; i < out_plane; i++)
			op[i] = b;

		for (int ic = 0; ic < Cin; ic++) {
			const float *ip = in + (size_t)ic * in_plane;
			const float *wk = weight + (((size_t)oc * Cin + ic) * kH) * kW;
			for (int ky = 0; ky < kH; ky++) {
				for (int kx = 0; kx < kW; kx++) {
					float wv = wk[ky * kW + kx];
					if (wv == 0.0f)
						continue;
					for (int oy = 0; oy < OH; oy++) {
						int sy = oy * stride + ky - pad;
						if (sy < 0 || sy >= H)
							continue;
						float       *orow = op + (size_t)oy * OW;
						const float *irow = ip + (size_t)sy * W;
						for (int ox = 0; ox < OW; ox++) {
							int sx = ox * stride + kx - pad;
							if (sx < 0 || sx >= W)
								continue;
							orow[ox] += wv * irow[sx];
						}
					}
				}
			}
		}
	}
}

/* Transposed convolution with stride == kernel and no padding (the only
 * configuration the DPT resize stage uses). Upsamples HxW -> (H*k)x(W*k).
 *   weight: [Cin, Cout, k, k]   bias: [Cout] or NULL
 *   out[oc, oh, ow] = bias[oc] + sum_ic in[ic, oh/k, ow/k] * W[ic,oc, oh%k, ow%k]
 */
static void conv_transpose_block(const float *in, int Cin, int H, int W, const float *weight, const float *bias, int Cout, int k, float *out) {
	int          OH = H * k, OW = W * k;
	const size_t in_plane  = (size_t)H * W;
	const size_t out_plane = (size_t)OH * OW;

	for (int oc = 0; oc < Cout; oc++) {
		float *op = out + (size_t)oc * out_plane;
		float  b  = bias ? bias[oc] : 0.0f;
		for (int oy = 0; oy < OH; oy++) {
			int iy = oy / k, ky = oy % k;
			for (int ox = 0; ox < OW; ox++) {
				int   ix = ox / k, kx = ox % k;
				float s = b;
				for (int ic = 0; ic < Cin; ic++) {
					float iv = in[(size_t)ic * in_plane + (size_t)iy * W + ix];
					float wv = weight[(((size_t)ic * Cout + oc) * k + ky) * k + kx];
					s += iv * wv;
				}
				op[(size_t)oy * OW + ox] = s;
			}
		}
	}
}

/* ================================================================== */
/* Bilinear resize (planar CHW)                                        */
/* ================================================================== */

static void bilinear_chw(const float *in, int C, int H, int W, int OH, int OW, int align_corners, float *out) {
	const size_t in_plane  = (size_t)H * W;
	const size_t out_plane = (size_t)OH * OW;
	float        sy = 0.0f, sx = 0.0f;
	if (align_corners) {
		sy = (OH > 1) ? (float)(H - 1) / (OH - 1) : 0.0f;
		sx = (OW > 1) ? (float)(W - 1) / (OW - 1) : 0.0f;
	}
	else {
		sy = (float)H / OH;
		sx = (float)W / OW;
	}
	for (int oy = 0; oy < OH; oy++) {
		float fy = align_corners ? oy * sy : (oy + 0.5f) * sy - 0.5f;
		if (fy < 0.0f)
			fy = 0.0f;
		int y0 = (int)fy;
		if (y0 > H - 1)
			y0 = H - 1;
		int   y1 = (y0 + 1 < H) ? y0 + 1 : y0;
		float wy = fy - y0;
		for (int ox = 0; ox < OW; ox++) {
			float fx = align_corners ? ox * sx : (ox + 0.5f) * sx - 0.5f;
			if (fx < 0.0f)
				fx = 0.0f;
			int x0 = (int)fx;
			if (x0 > W - 1)
				x0 = W - 1;
			int   x1 = (x0 + 1 < W) ? x0 + 1 : x0;
			float wx = fx - x0;
			for (int c = 0; c < C; c++) {
				const float *ip                                   = in + (size_t)c * in_plane;
				float        v00                                  = ip[(size_t)y0 * W + x0];
				float        v01                                  = ip[(size_t)y0 * W + x1];
				float        v10                                  = ip[(size_t)y1 * W + x0];
				float        v11                                  = ip[(size_t)y1 * W + x1];
				float        top                                  = v00 + (v01 - v00) * wx;
				float        bot                                  = v10 + (v11 - v10) * wx;
				out[(size_t)c * out_plane + (size_t)oy * OW + ox] = top + (bot - top) * wy;
			}
		}
	}
}

/* ================================================================== */
/* Backbone (DINOv2 ViT-L/14)                                          */
/* ================================================================== */

/* Multi-head self-attention for one transformer block, fused QKV.
 * x is [DA_TOKENS, DA_HIDDEN]; result written to `out` (same shape). */
static int da_attention(iris_depth_t *m, int layer, const float *x, float *out, float *scratch) {
	const int T = DA_TOKENS, Hd = DA_HIDDEN;

	/* Fused QKV: weight [3*Hd, Hd], bias [3*Hd]; output projection. */
	const float *wqkv = da_tensorf(m, BB ".blocks.%d.attn.qkv.weight", layer);
	const float *bqkv = da_tensorf(m, BB ".blocks.%d.attn.qkv.bias", layer);
	const float *wo   = da_tensorf(m, BB ".blocks.%d.attn.proj.weight", layer);
	const float *bo   = da_tensorf(m, BB ".blocks.%d.attn.proj.bias", layer);
	if (!wqkv || !bqkv || !wo || !bo)
		return -1;

	/* scratch layout: QKV (T*3*Hd), attn output A (T*Hd), scores S (T*T). */
	float *QKV = scratch;
	float *A   = QKV + (size_t)T * 3 * Hd; /* attention output (concat heads) */
	float *S   = A + (size_t)T * Hd;       /* per-head scores [T,T]           */

	/* QKV[T,3*Hd] = x @ Wqkv^T + bqkv. Within a row: [q(Hd) | k(Hd) | v(Hd)],
	 * each Hd block laid out as heads*head_dim. */
	linear(x, T, Hd, wqkv, bqkv, 3 * Hd, QKV);
	const float *Qbase = QKV;
	const float *Kbase = QKV + Hd;
	const float *Vbase = QKV + 2 * Hd;

	const float scale = 1.0f / sqrtf((float)DA_HEAD_DIM);
	for (int h = 0; h < DA_HEADS; h++) {
		int off = h * DA_HEAD_DIM;
		/* S[T,T] = scale * Qh @ Kh^T  (strided per-head views, stride 3*Hd). */
		gemm_nt(T, T, DA_HEAD_DIM, Qbase + off, 3 * Hd, Kbase + off, 3 * Hd, S, T, scale, 0);
		/* row softmax */
		for (int i = 0; i < T; i++) {
			float *row = S + (size_t)i * T;
			float  mx  = row[0];
			for (int j = 1; j < T; j++)
				if (row[j] > mx)
					mx = row[j];
			float sum = 0.0f;
			for (int j = 0; j < T; j++) {
				row[j] = expf(row[j] - mx);
				sum += row[j];
			}
			float inv = 1.0f / sum;
			for (int j = 0; j < T; j++)
				row[j] *= inv;
		}
		/* Ah[T,head_dim] = S @ Vh  -> write into A at head offset (stride Hd). */
		gemm_nn(T, DA_HEAD_DIM, T, S, T, Vbase + off, 3 * Hd, A + off, Hd);
	}

	/* output projection: out = A @ Wo^T + bo */
	linear(A, T, Hd, wo, bo, Hd, out);
	return 0;
}

/* Run the backbone, capturing the four feature maps (after the final
 * LayerNorm) into feats[k] = [DA_TOKENS, DA_HIDDEN]. Returns 0 on success. */
static int da_backbone(iris_depth_t *m, const float *pixels /* [3,518,518] */, float *feats[4]) {
	const int T = DA_TOKENS, Hd = DA_HIDDEN;

	/* ---- Patch embedding (14x14 stride-14 conv, 3 -> 1024) ---- */
	const float *pw  = da_tensor(m, BB ".patch_embed.proj.weight");
	const float *pb  = da_tensor(m, BB ".patch_embed.proj.bias");
	const float *cls = da_tensor(m, BB ".cls_token");
	const float *pos = da_tensor(m, BB ".pos_embed");
	if (!pw || !pb || !cls || !pos)
		return -1;

	float *x = malloc((size_t)T * Hd * sizeof(float));
	if (!x)
		return -1;

	/* token 0 = cls */
	memcpy(x, cls, Hd * sizeof(float));
	/* tokens 1.. = patches (row-major), conv with stride=kernel, no overlap */
	for (int r = 0; r < DA_GRID; r++) {
		for (int c = 0; c < DA_GRID; c++) {
			float *tok = x + (size_t)(1 + r * DA_GRID + c) * Hd;
			for (int oc = 0; oc < Hd; oc++) {
				float        s  = pb[oc];
				const float *wk = pw + (size_t)oc * 3 * DA_PATCH * DA_PATCH;
				for (int ic = 0; ic < 3; ic++) {
					const float *ip  = pixels + (size_t)ic * DA_IMG * DA_IMG;
					const float *wic = wk + (size_t)ic * DA_PATCH * DA_PATCH;
					for (int ky = 0; ky < DA_PATCH; ky++) {
						const float *irow = ip + (size_t)(r * DA_PATCH + ky) * DA_IMG + c * DA_PATCH;
						const float *wrow = wic + ky * DA_PATCH;
						for (int kx = 0; kx < DA_PATCH; kx++)
							s += irow[kx] * wrow[kx];
					}
				}
				tok[oc] = s;
			}
		}
	}
	/* add position embeddings (exact size match, no interpolation) */
	for (size_t i = 0; i < (size_t)T * Hd; i++)
		x[i] += pos[i];

	/* ---- work buffers ---- */
	float *norm = malloc((size_t)T * Hd * sizeof(float));
	float *tmp  = malloc((size_t)T * Hd * sizeof(float));
	float *mlp  = malloc((size_t)T * DA_MLP * sizeof(float));
	/* attention scratch: QKV (3*T*Hd) + A (T*Hd) + scores (T*T) */
	float *attn_scratch = malloc(((size_t)4 * T * Hd + (size_t)T * T) * sizeof(float));
	if (!norm || !tmp || !mlp || !attn_scratch) {
		free(x);
		free(norm);
		free(tmp);
		free(mlp);
		free(attn_scratch);
		return -1;
	}

	int next_capture = 0;
	int ok           = 1;

	for (int l = 0; l < DA_LAYERS && ok; l++) {
		const float *n1w = da_tensorf(m, BB ".blocks.%d.norm1.weight", l);
		const float *n1b = da_tensorf(m, BB ".blocks.%d.norm1.bias", l);
		const float *n2w = da_tensorf(m, BB ".blocks.%d.norm2.weight", l);
		const float *n2b = da_tensorf(m, BB ".blocks.%d.norm2.bias", l);
		const float *ls1 = da_tensorf(m, BB ".blocks.%d.ls1.gamma", l);
		const float *ls2 = da_tensorf(m, BB ".blocks.%d.ls2.gamma", l);
		const float *f1w = da_tensorf(m, BB ".blocks.%d.mlp.fc1.weight", l);
		const float *f1b = da_tensorf(m, BB ".blocks.%d.mlp.fc1.bias", l);
		const float *f2w = da_tensorf(m, BB ".blocks.%d.mlp.fc2.weight", l);
		const float *f2b = da_tensorf(m, BB ".blocks.%d.mlp.fc2.bias", l);
		if (!n1w || !n1b || !n2w || !n2b || !ls1 || !ls2 || !f1w || !f1b || !f2w || !f2b) {
			ok = 0;
			break;
		}

		/* --- attention block --- */
		layer_norm(x, T, Hd, n1w, n1b, norm);
		if (da_attention(m, l, norm, tmp, attn_scratch) != 0) {
			ok = 0;
			break;
		}
		/* x = x + ls1 * attn */
		for (int i = 0; i < T; i++) {
			float       *xr = x + (size_t)i * Hd;
			const float *tr = tmp + (size_t)i * Hd;
			for (int j = 0; j < Hd; j++)
				xr[j] += ls1[j] * tr[j];
		}

		/* --- MLP block --- */
		layer_norm(x, T, Hd, n2w, n2b, norm);
		linear(norm, T, Hd, f1w, f1b, DA_MLP, mlp);
		gelu(mlp, (size_t)T * DA_MLP);
		linear(mlp, T, DA_MLP, f2w, f2b, Hd, tmp);
		/* x = x + ls2 * mlp_out */
		for (int i = 0; i < T; i++) {
			float       *xr = x + (size_t)i * Hd;
			const float *tr = tmp + (size_t)i * Hd;
			for (int j = 0; j < Hd; j++)
				xr[j] += ls2[j] * tr[j];
		}

		/* capture selected block outputs (post final LayerNorm) */
		if (next_capture < 4 && l == DA_OUT_LAYERS[next_capture]) {
			const float *lnw = da_tensor(m, BB ".norm.weight");
			const float *lnb = da_tensor(m, BB ".norm.bias");
			if (!lnw || !lnb) {
				ok = 0;
				break;
			}
			layer_norm(x, T, Hd, lnw, lnb, feats[next_capture]);
			next_capture++;
		}
	}

	free(x);
	free(norm);
	free(tmp);
	free(mlp);
	free(attn_scratch);
	return ok ? 0 : -1;
}

/* ================================================================== */
/* DPT head                                                            */
/* ================================================================== */

/* Reshape a captured feature [DA_TOKENS, DA_HIDDEN] (cls dropped) into a
 * planar CHW map [DA_HIDDEN, DA_GRID, DA_GRID]. */
static void da_tokens_to_chw(const float *feat, float *out) {
	for (int r = 0; r < DA_GRID; r++)
		for (int c = 0; c < DA_GRID; c++) {
			const float *tok = feat + (size_t)(1 + r * DA_GRID + c) * DA_HIDDEN;
			for (int ch = 0; ch < DA_HIDDEN; ch++)
				out[(size_t)ch * DA_GRID * DA_GRID + (size_t)r * DA_GRID + c] = tok[ch];
		}
}

/* ResidualConvUnit (in place into `buf`):
 *   buf = buf + conv2(relu(conv1(relu(buf)))) , all 3x3 pad-1, C channels. */
static int da_res_conv_unit(iris_depth_t *m, const char *prefix, float *buf, int C, int H, int W, float *t1, float *t2) {
	const size_t plane = (size_t)C * H * W;
	const float *w1    = da_tensorf(m, "%s.conv1.weight", prefix);
	const float *b1    = da_tensorf(m, "%s.conv1.bias", prefix);
	const float *w2    = da_tensorf(m, "%s.conv2.weight", prefix);
	const float *b2    = da_tensorf(m, "%s.conv2.bias", prefix);
	if (!w1 || !b1 || !w2 || !b2)
		return -1;

	memcpy(t1, buf, plane * sizeof(float));
	relu_inplace(t1, plane);
	conv2d(t1, C, H, W, w1, b1, C, 3, 3, 1, 1, t2);
	relu_inplace(t2, plane);
	conv2d(t2, C, H, W, w2, b2, C, 3, 3, 1, 1, t1);
	for (size_t i = 0; i < plane; i++)
		buf[i] += t1[i];
	return 0;
}

/* One RefineNet feature-fusion block (scratch.refinenet{N}).
 *   in:        [C, H, W]  current path (consumed)
 *   residual:  [C, H, W]  or NULL (skip connection at same resolution)
 *   out:       [C, OH, OW] caller-allocated, OH/OW are the upsample target
 * Returns 0 on success. */
static int da_refinenet(iris_depth_t *m, int n, float *in, const float *residual, int C, int H, int W, int OH, int OW, float *out) {
	char prefix[64];
	snprintf(prefix, sizeof(prefix), HEAD ".scratch.refinenet%d", n);
	const size_t plane = (size_t)C * H * W;

	float *t1  = malloc(plane * sizeof(float));
	float *t2  = malloc(plane * sizeof(float));
	float *res = malloc(plane * sizeof(float));
	if (!t1 || !t2 || !res) {
		free(t1);
		free(t2);
		free(res);
		return -1;
	}

	int rc = 0;
	if (residual) {
		char rp[96];
		memcpy(res, residual, plane * sizeof(float));
		snprintf(rp, sizeof(rp), "%s.resConfUnit1", prefix);
		if (da_res_conv_unit(m, rp, res, C, H, W, t1, t2) != 0) {
			rc = -1;
			goto done;
		}
		for (size_t i = 0; i < plane; i++)
			in[i] += res[i];
	}

	{
		char rp[96];
		snprintf(rp, sizeof(rp), "%s.resConfUnit2", prefix);
		if (da_res_conv_unit(m, rp, in, C, H, W, t1, t2) != 0) {
			rc = -1;
			goto done;
		}
	}

	/* upsample to (OH,OW) (align_corners=True), then 1x1 out_conv (with bias). */
	{
		float *up = malloc((size_t)C * OH * OW * sizeof(float));
		if (!up) {
			rc = -1;
			goto done;
		}
		bilinear_chw(in, C, H, W, OH, OW, 1, up);
		const float *pw = da_tensorf(m, "%s.out_conv.weight", prefix);
		const float *pb = da_tensorf(m, "%s.out_conv.bias", prefix);
		if (!pw || !pb) {
			free(up);
			rc = -1;
			goto done;
		}
		conv2d(up, C, OH, OW, pw, pb, C, 1, 1, 1, 0, out);
		free(up);
	}

done:
	free(t1);
	free(t2);
	free(res);
	return rc;
}

/* Run the DPT head. `feats` are the four captured backbone features.
 * Writes a [DA_IMG, DA_IMG] depth map into `depth_out`. Returns 0 on success. */
static int da_head(iris_depth_t *m, float *feats[4], float *depth_out) {
	int rc = -1;
	/* Per-stage "_rn" features at 256 channels. */
	float *rn[4] = {0};
	int    rdim[4][2]; /* [k] = {H, W} of the _rn output */
	float *chw = malloc((size_t)DA_HIDDEN * DA_GRID * DA_GRID * sizeof(float));
	if (!chw)
		return -1;

	for (int k = 0; k < 4; k++) {
		da_tokens_to_chw(feats[k], chw);

		/* projects.k : 1x1 projection 1024 -> DA_PROJ_CH[k] */
		int          pc  = DA_PROJ_CH[k];
		const float *prw = da_tensorf(m, HEAD ".projects.%d.weight", k);
		const float *prb = da_tensorf(m, HEAD ".projects.%d.bias", k);
		if (!prw || !prb)
			goto done;
		float *proj = malloc((size_t)pc * DA_GRID * DA_GRID * sizeof(float));
		if (!proj)
			goto done;
		conv2d(chw, DA_HIDDEN, DA_GRID, DA_GRID, prw, prb, pc, 1, 1, 1, 0, proj);

		/* resize_layers.k */
		float *res = NULL;
		int    rh = 0, rw = 0;
		if (k == 0 || k == 1) {
			int          s   = (k == 0) ? 4 : 2;
			const float *rwt = da_tensorf(m, HEAD ".resize_layers.%d.weight", k);
			const float *rbs = da_tensorf(m, HEAD ".resize_layers.%d.bias", k);
			if (!rwt || !rbs) {
				free(proj);
				goto done;
			}
			rh  = DA_GRID * s;
			rw  = DA_GRID * s;
			res = malloc((size_t)pc * rh * rw * sizeof(float));
			if (!res) {
				free(proj);
				goto done;
			}
			conv_transpose_block(proj, pc, DA_GRID, DA_GRID, rwt, rbs, pc, s, res);
		}
		else if (k == 2) {
			/* identity (no resize_layers.2 in the checkpoint) */
			rh   = DA_GRID;
			rw   = DA_GRID;
			res  = proj;
			proj = NULL;
		}
		else { /* k == 3: 3x3 stride-2 pad-1 downsample */
			const float *rwt = da_tensorf(m, HEAD ".resize_layers.%d.weight", k);
			const float *rbs = da_tensorf(m, HEAD ".resize_layers.%d.bias", k);
			if (!rwt || !rbs) {
				free(proj);
				goto done;
			}
			rh  = (DA_GRID + 2 - 3) / 2 + 1;
			rw  = rh;
			res = malloc((size_t)pc * rh * rw * sizeof(float));
			if (!res) {
				free(proj);
				goto done;
			}
			conv2d(proj, pc, DA_GRID, DA_GRID, rwt, rbs, pc, 3, 3, 2, 1, res);
		}
		free(proj);

		/* scratch.layer{k+1}_rn : 3x3 pad-1, no bias, pc -> 256 */
		const float *cw = da_tensorf(m, HEAD ".scratch.layer%d_rn.weight", k + 1);
		if (!cw) {
			free(res);
			goto done;
		}
		rn[k] = malloc((size_t)DA_FUSION_CH * rh * rw * sizeof(float));
		if (!rn[k]) {
			free(res);
			goto done;
		}
		conv2d(res, pc, rh, rw, cw, NULL, DA_FUSION_CH, 3, 3, 1, 1, rn[k]);
		free(res);
		rdim[k][0] = rh;
		rdim[k][1] = rw;
	}

	/* ---- feature fusion (coarse -> fine): refinenet 4,3,2,1 ---- */
	{
		float *path = NULL;
		int    ph = 0, pw = 0;
		for (int step = 0; step < 4; step++) {
			int k = 3 - step;
			int n = 4 - step; /* refinenet index (4..1) */
			int H = rdim[k][0], W = rdim[k][1];
			/* target upsample size: next-finer feature's size, or x2 for the last. */
			int OH, OW;
			if (step < 3) {
				OH = rdim[k - 1][0];
				OW = rdim[k - 1][1];
			}
			else {
				OH = H * 2;
				OW = W * 2;
			}

			float *out = malloc((size_t)DA_FUSION_CH * OH * OW * sizeof(float));
			if (!out) {
				free(path);
				goto done;
			}

			if (step == 0) {
				/* deepest (refinenet4): no residual; rn[k] is the path input. */
				if (da_refinenet(m, n, rn[k], NULL, DA_FUSION_CH, H, W, OH, OW, out) != 0) {
					free(out);
					free(path);
					goto done;
				}
			}
			else {
				if (da_refinenet(m, n, path, rn[k], DA_FUSION_CH, ph, pw, OH, OW, out) != 0) {
					free(out);
					free(path);
					goto done;
				}
			}
			free(path);
			path = out;
			ph   = OH;
			pw   = OW;
		}

		/* ---- output convs ---- */
		const float *c1w = da_tensor(m, HEAD ".scratch.output_conv1.weight");
		const float *c1b = da_tensor(m, HEAD ".scratch.output_conv1.bias");
		const float *c2w = da_tensor(m, HEAD ".scratch.output_conv2.0.weight");
		const float *c2b = da_tensor(m, HEAD ".scratch.output_conv2.0.bias");
		const float *c3w = da_tensor(m, HEAD ".scratch.output_conv2.2.weight");
		const float *c3b = da_tensor(m, HEAD ".scratch.output_conv2.2.bias");
		if (!c1w || !c1b || !c2w || !c2b || !c3w || !c3b) {
			free(path);
			goto done;
		}

		/* output_conv1: DA_FUSION_CH -> DA_HEAD_CH, 3x3 pad-1 */
		float *h1 = malloc((size_t)DA_HEAD_CH * ph * pw * sizeof(float));
		if (!h1) {
			free(path);
			goto done;
		}
		conv2d(path, DA_FUSION_CH, ph, pw, c1w, c1b, DA_HEAD_CH, 3, 3, 1, 1, h1);
		free(path);

		/* upsample to (518,518), bilinear align_corners=True */
		float *h1u = malloc((size_t)DA_HEAD_CH * DA_IMG * DA_IMG * sizeof(float));
		if (!h1u) {
			free(h1);
			goto done;
		}
		bilinear_chw(h1, DA_HEAD_CH, ph, pw, DA_IMG, DA_IMG, 1, h1u);
		free(h1);

		/* output_conv2.0: DA_HEAD_CH -> DA_HEAD_HIDDEN, 3x3 pad-1, ReLU */
		float *h2 = malloc((size_t)DA_HEAD_HIDDEN * DA_IMG * DA_IMG * sizeof(float));
		if (!h2) {
			free(h1u);
			goto done;
		}
		conv2d(h1u, DA_HEAD_CH, DA_IMG, DA_IMG, c2w, c2b, DA_HEAD_HIDDEN, 3, 3, 1, 1, h2);
		free(h1u);
		relu_inplace(h2, (size_t)DA_HEAD_HIDDEN * DA_IMG * DA_IMG);

		/* output_conv2.2: DA_HEAD_HIDDEN -> 1, 1x1 -> depth.
		 * (No terminal ReLU: the DA3 mono head emits a signed depth field.) */
		conv2d(h2, DA_HEAD_HIDDEN, DA_IMG, DA_IMG, c3w, c3b, 1, 1, 1, 1, 0, depth_out);
		free(h2);
		rc = 0;
	}

done:
	for (int k = 0; k < 4; k++)
		free(rn[k]);
	free(chw);
	return rc;
}

/* ================================================================== */
/* Tileable detrending                                                 */
/* ================================================================== */

/* Remove the dominant low-frequency tilt from a depth map so that the
 * result tiles seamlessly. Monocular models assume a perspective camera and
 * bake a smooth front-to-back ramp into the depth (darker far / brighter
 * near). For a top-down, tileable texture that ramp is spurious and makes
 * opposite edges mismatch. Under perspective the disparity of a flat ground
 * plane is affine in pixel coordinates, so we least-squares fit a plane
 * a*(x-cx) + b*(y-cy) + c and subtract its tilt, which equalizes opposite
 * borders while preserving genuine surface relief. The coordinate grid is
 * regular and centered, so the design matrix is orthogonal and the fit is a
 * pair of independent 1D slopes. */
static void detrend_plane(float *img, int H, int W) {
	if (H < 2 && W < 2)
		return;
	const double cx = (W - 1) * 0.5, cy = (H - 1) * 0.5;
	double       Sxx = 0.0, Syy = 0.0, Sxz = 0.0, Syz = 0.0;
	for (int x = 0; x < W; x++) {
		double dx = x - cx;
		Sxx += dx * dx;
	}
	Sxx *= H;
	for (int y = 0; y < H; y++) {
		double dy = y - cy;
		Syy += dy * dy;
	}
	Syy *= W;
	for (int y = 0; y < H; y++) {
		double       dy  = y - cy;
		const float *row = img + (size_t)y * W;
		for (int x = 0; x < W; x++) {
			double v = row[x];
			Sxz += v * (x - cx);
			Syz += v * dy;
		}
	}
	double a = (Sxx > 0.0) ? Sxz / Sxx : 0.0;
	double b = (Syy > 0.0) ? Syz / Syy : 0.0;
	for (int y = 0; y < H; y++) {
		double dy  = y - cy;
		float *row = img + (size_t)y * W;
		for (int x = 0; x < W; x++)
			row[x] -= (float)(a * (x - cx) + b * dy);
	}
}

/* ------------------------------------------------------------------ */
/* Seamless tiling via periodic+smooth decomposition (Moisan 2011)      */
/* ------------------------------------------------------------------ */
/* Equalizing opposite edges with a separable per-row/per-column shift injects a
 * visible streak on any row or column whose two edges differ sharply (e.g. black
 * on one side, white on the other): that line gets shifted hard relative to its
 * neighbours. The cure is to stop treating rows and columns independently and
 * instead subtract the single smoothest 2D field that makes the image periodic.
 * That field is Moisan's "smooth component": the solution of a periodic Poisson
 * equation whose source is the jump between opposite borders. A localized edge
 * mismatch is then diffused into a gentle 2D bump rather than smeared along a
 * whole scanline, so no streak remains and the result tiles exactly. Solved in
 * the Fourier domain, so it requires power-of-two dimensions (always the case
 * for textures); other sizes are left unchanged. */

#ifndef IRIS_TWO_PI
#define IRIS_TWO_PI 6.28318530717958647692
#endif

static int is_pow2(int n) {
	return n > 0 && (n & (n - 1)) == 0;
}

/* In-place iterative radix-2 Cooley-Tukey FFT. inv=0 forward (e^-i), inv=1
 * inverse (e^+i, scaled by 1/n). n must be a power of two. */
static void fft1d(double *re, double *im, int n, int inv) {
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;
		if (i < j) {
			double tr = re[i];
			re[i]     = re[j];
			re[j]     = tr;
			double ti = im[i];
			im[i]     = im[j];
			im[j]     = ti;
		}
	}
	for (int len = 2; len <= n; len <<= 1) {
		double ang = IRIS_TWO_PI / len * (inv ? 1.0 : -1.0);
		double wr = cos(ang), wi = sin(ang);
		for (int i = 0; i < n; i += len) {
			double cwr = 1.0, cwi = 0.0;
			for (int k = 0; k < len / 2; k++) {
				int    a = i + k, b = i + k + len / 2;
				double vr = re[b] * cwr - im[b] * cwi;
				double vi = re[b] * cwi + im[b] * cwr;
				re[b]     = re[a] - vr;
				im[b]     = im[a] - vi;
				re[a] += vr;
				im[a] += vi;
				double ncwr = cwr * wr - cwi * wi;
				cwi         = cwr * wi + cwi * wr;
				cwr         = ncwr;
			}
		}
	}
	if (inv) {
		for (int i = 0; i < n; i++) {
			re[i] /= n;
			im[i] /= n;
		}
	}
}

/* 2D FFT: transform every row (length W), then every column (length H), using
 * caller-provided column scratch of length >= H. */
static void fft2d(double *re, double *im, int H, int W, int inv, double *cr, double *ci) {
	for (int i = 0; i < H; i++)
		fft1d(re + (size_t)i * W, im + (size_t)i * W, W, inv);
	for (int j = 0; j < W; j++) {
		for (int i = 0; i < H; i++) {
			cr[i] = re[(size_t)i * W + j];
			ci[i] = im[(size_t)i * W + j];
		}
		fft1d(cr, ci, H, inv);
		for (int i = 0; i < H; i++) {
			re[(size_t)i * W + j] = cr[i];
			im[(size_t)i * W + j] = ci[i];
		}
	}
}

/* Subtract the smooth component so img becomes seamlessly tileable. Only acts on
 * power-of-two dimensions (textures always qualify); other sizes are untouched. */
static void make_seamless_poisson(float *img, int H, int W) {
	if (!is_pow2(H) || !is_pow2(W) || H < 2 || W < 2)
		return;

	size_t  n  = (size_t)H * W;
	double *re = calloc(n, sizeof(double));
	double *im = calloc(n, sizeof(double));
	double *cr = malloc((size_t)H * sizeof(double));
	double *ci = malloc((size_t)H * sizeof(double));
	if (!re || !im || !cr || !ci) {
		free(re);
		free(im);
		free(cr);
		free(ci);
		return;
	}

	/* Boundary jump field: the wrap-around difference across each border,
	 * accumulated (corners receive both a row and a column contribution). */
	for (int i = 0; i < H; i++) {
		double *row = re + (size_t)i * W;
		double  l = img[(size_t)i * W + 0], r = img[(size_t)i * W + (W - 1)];
		row[0] += r - l;
		row[W - 1] += l - r;
	}
	for (int j = 0; j < W; j++) {
		double t = img[(size_t)0 * W + j], b = img[(size_t)(H - 1) * W + j];
		re[(size_t)0 * W + j] += b - t;
		re[(size_t)(H - 1) * W + j] += t - b;
	}

	fft2d(re, im, H, W, 0, cr, ci);

	/* Divide by the periodic-Laplacian eigenvalues to solve the Poisson eq;
	 * the DC term (constant offset) is undetermined, so pin it to zero. */
	for (int q = 0; q < H; q++) {
		double cq = 2.0 * cos(IRIS_TWO_PI * q / H);
		for (int j = 0; j < W; j++) {
			size_t idx = (size_t)q * W + j;
			if (q == 0 && j == 0) {
				re[idx] = 0.0;
				im[idx] = 0.0;
				continue;
			}
			double denom = cq + 2.0 * cos(IRIS_TWO_PI * j / W) - 4.0;
			re[idx] /= denom;
			im[idx] /= denom;
		}
	}

	fft2d(re, im, H, W, 1, cr, ci);

	for (size_t i = 0; i < n; i++)
		img[i] -= (float)re[i];

	free(re);
	free(im);
	free(cr);
	free(ci);
}

/* ================================================================== */
/* Public API                                                          */
/* ================================================================== */

iris_depth_t *iris_depth_load(const char *path) {
	safetensors_file_t *sf = safetensors_open(path);
	if (!sf) {
		fprintf(stderr, "DepthAnything3: failed to open '%s'\n", path);
		return NULL;
	}
	if (!safetensors_find(sf, BB ".cls_token") || !safetensors_find(sf, HEAD ".scratch.output_conv2.2.weight")) {
		fprintf(stderr, "DepthAnything3: '%s' does not look like a Depth Anything 3 mono model\n", path);
		safetensors_close(sf);
		return NULL;
	}
	iris_depth_t *m = calloc(1, sizeof(iris_depth_t));
	if (!m) {
		safetensors_close(sf);
		return NULL;
	}
	m->sf = sf;
	return m;
}

void iris_depth_set_tileable(iris_depth_t *m, int on) {
	if (m)
		m->tileable = on ? 1 : 0;
}

void iris_depth_free(iris_depth_t *m) {
	if (!m)
		return;
	safetensors_close(m->sf);
	free(m);
}

iris_image *iris_depth_estimate(iris_depth_t *m, const iris_image *input) {
	if (!m || !input || !input->data)
		return NULL;

	int W = input->width, H = input->height, ic = input->channels;

	/* ---- Preprocess: resize to 518x518 (bilinear) + ImageNet normalize ----
	 * Build a planar CHW float source at the original resolution, then resize. */
	float *src = malloc((size_t)3 * H * W * sizeof(float));
	if (!src)
		return NULL;
	const size_t src_plane = (size_t)H * W;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			const uint8_t *px       = input->data + ((size_t)y * W + x) * ic;
			size_t         pi       = (size_t)y * W + x;
			src[0 * src_plane + pi] = px[0] / 255.0f;
			src[1 * src_plane + pi] = (ic > 1 ? px[1] : px[0]) / 255.0f;
			src[2 * src_plane + pi] = (ic > 2 ? px[2] : px[0]) / 255.0f;
		}
	}

	float *pixels = malloc((size_t)3 * DA_IMG * DA_IMG * sizeof(float));
	if (!pixels) {
		free(src);
		return NULL;
	}
	bilinear_chw(src, 3, H, W, DA_IMG, DA_IMG, 0, pixels);
	free(src);

	/* normalize per channel */
	for (int c = 0; c < 3; c++) {
		float *p    = pixels + (size_t)c * DA_IMG * DA_IMG;
		float  mean = DA_MEAN[c], istd = 1.0f / DA_STD[c];
		for (size_t i = 0; i < (size_t)DA_IMG * DA_IMG; i++)
			p[i] = (p[i] - mean) * istd;
	}

	/* ---- Backbone ---- */
	float *feats[4] = {0};
	for (int k = 0; k < 4; k++) {
		feats[k] = malloc((size_t)DA_TOKENS * DA_HIDDEN * sizeof(float));
		if (!feats[k]) {
			for (int j = 0; j < k; j++)
				free(feats[j]);
			free(pixels);
			return NULL;
		}
	}

#ifdef USE_VULKAN
	/* Fold the whole forward into one GPU batch when resident ops exist. */
	if (iris_vulkan_available())
		iris_gpu_batch_begin();
#endif

	int ok = (da_backbone(m, pixels, feats) == 0);
	free(pixels);

	float *depth = NULL;
	if (ok) {
		depth = malloc((size_t)DA_IMG * DA_IMG * sizeof(float));
		if (!depth || da_head(m, feats, depth) != 0) {
			free(depth);
			depth = NULL;
			ok    = 0;
		}
	}

#ifdef USE_VULKAN
	if (iris_vulkan_available())
		iris_gpu_batch_end();
#endif

	for (int k = 0; k < 4; k++)
		free(feats[k]);
	if (!ok) {
		free(depth);
		return NULL;
	}

	/* ---- Resize depth back to input resolution (bilinear) ---- */
	float *depth_full = malloc((size_t)H * W * sizeof(float));
	if (!depth_full) {
		free(depth);
		return NULL;
	}
	bilinear_chw(depth, 1, DA_IMG, DA_IMG, H, W, 0, depth_full);
	free(depth);

	/* ---- Optional: make the map tile seamlessly ----
	 * First remove the perspective tilt (keeps the border correction small),
	 * then feather opposite edges together so no seam remains. */
	if (m->tileable) {
		detrend_plane(depth_full, H, W);
		make_seamless_poisson(depth_full, H, W);
	}

	/* ---- Min/max normalize to grayscale (inverted: brighter = farther) ---- */
	float mn = depth_full[0], mx = depth_full[0];
	for (size_t i = 1; i < (size_t)H * W; i++) {
		if (depth_full[i] < mn)
			mn = depth_full[i];
		if (depth_full[i] > mx)
			mx = depth_full[i];
	}
	float range = (mx - mn) > 1e-8f ? (mx - mn) : 1.0f;

	iris_image *out = iris_image_create(W, H, 1);
	if (!out) {
		free(depth_full);
		return NULL;
	}
	for (size_t i = 0; i < (size_t)H * W; i++) {
		float v  = 1.0f - (depth_full[i] - mn) / range;
		int   iv = (int)(v * 255.0f + 0.5f);
		if (iv < 0)
			iv = 0;
		if (iv > 255)
			iv = 255;
		out->data[i] = (uint8_t)iv;
	}
	free(depth_full);
	return out;
}
