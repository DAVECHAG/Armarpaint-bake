/*
 * Iris Vulkan Acceleration - Implementation
 *
 * GEMM-offload backend: dispatches matrix multiplies to the GPU via Vulkan
 * compute shaders. Two pipelines are provided, both tiled 16x16:
 *   - f32:  C = alpha * op(A) @ op(B) + beta * C   (A, B, C all f32)
 *   - bf16: same, but B is bfloat16 (weight-bound linear layers)
 *
 * Model weights (the cached/bf16 paths) are uploaded to GPU buffers once and
 * keyed by their CPU pointer. Activations are streamed each call. Each GEMM
 * is a single dispatch followed by a fence wait, which keeps the contract
 * identical to a synchronous BLAS call.
 */

#include "iris_vulkan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

/* Circular (seamless tiling) padding flag, defined in iris_kernels.c. */
extern int iris_circular;

/* SPIR-V generated at build time by glslc (-mfmt=c). Each header expands to a
 * brace-enclosed uint32 initializer list assigned below. */
static const uint32_t gemm_f32_spv[] =
#include "iris_vulkan_gemm_spv.h"
    ;
static const uint32_t gemm_bf16_spv[] =
#include "iris_vulkan_gemm_bf16_spv.h"
    ;
static const uint32_t gemm_q8_spv[] =
#include "iris_vulkan_gemm_q8_spv.h"
    ;
static const uint32_t conv2d_spv[] =
#include "iris_vulkan_conv2d_spv.h"
    ;
static const uint32_t groupnorm_spv[] =
#include "iris_vulkan_groupnorm_spv.h"
    ;
static const uint32_t swish_spv[] =
#include "iris_vulkan_swish_spv.h"
    ;
static const uint32_t add_spv[] =
#include "iris_vulkan_add_spv.h"
    ;
static const uint32_t upsample_spv[] =
#include "iris_vulkan_upsample_spv.h"
    ;
static const uint32_t leakyrelu_spv[] =
#include "iris_vulkan_leakyrelu_spv.h"
    ;
static const uint32_t scaleadd_spv[] =
#include "iris_vulkan_scaleadd_spv.h"
    ;
static const uint32_t qwen_rmsnorm_spv[] =
#include "iris_vulkan_qwen_rmsnorm_spv.h"
    ;
static const uint32_t qwen_headrms_spv[] =
#include "iris_vulkan_qwen_headrms_spv.h"
    ;
static const uint32_t qwen_rope_spv[] =
#include "iris_vulkan_qwen_rope_spv.h"
    ;
static const uint32_t qwen_attn_spv[] =
#include "iris_vulkan_qwen_attn_spv.h"
    ;
static const uint32_t qwen_silumul_spv[] =
#include "iris_vulkan_qwen_silumul_spv.h"
    ;
static const uint32_t res_rmsnorm_spv[] =
#include "iris_vulkan_res_rmsnorm_spv.h"
    ;
static const uint32_t res_qkrmsnorm_spv[] =
#include "iris_vulkan_res_qkrmsnorm_spv.h"
    ;
static const uint32_t res_rope_spv[] =
#include "iris_vulkan_res_rope_spv.h"
    ;
static const uint32_t res_attn_spv[] =
#include "iris_vulkan_res_attn_spv.h"
    ;
static const uint32_t res_split_spv[] =
#include "iris_vulkan_res_split_spv.h"
    ;
static const uint32_t res_gatedadd_spv[] =
#include "iris_vulkan_res_gatedadd_spv.h"
    ;
static const uint32_t res_adaln_spv[] =
#include "iris_vulkan_res_adaln_spv.h"
    ;
static const uint32_t res_linear_f32_spv[] =
#include "iris_vulkan_res_linear_f32_spv.h"
    ;
static const uint32_t flux_rowcopy_spv[] =
#include "iris_vulkan_flux_rowcopy_spv.h"
    ;

extern int iris_verbose;

/* ---------------------------------------------------------------------- */

#define VK_OK(expr)                                                                     \
	do {                                                                                \
		VkResult _r = (expr);                                                           \
		if (_r != VK_SUCCESS) {                                                         \
			fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
			return 0;                                                                   \
		}                                                                               \
	} while (0)

typedef struct {
	VkBuffer       buffer;
	VkDeviceMemory mem;
	VkDeviceSize   capacity; /* bytes */
	void          *map;
} vk_buffer_t;

typedef struct {
	const void *key;         /* CPU pointer of the cached matrix */
	uint64_t    fingerprint; /* content hash: detects pointer reuse with new data */
	size_t      bytes;       /* uploaded size */
	vk_buffer_t buf;
} weight_entry_t;

static struct {
	int initialized;
	int available;

	VkInstance       instance;
	VkPhysicalDevice phys;
	VkDevice         device;
	uint32_t         queue_family;
	VkQueue          queue;

	VkCommandPool   cmd_pool;
	VkCommandBuffer cmd;
	VkFence         fence;

	/* Dedicated pool/buffer/fence for staging copies, so uploads can run while
	 * a resident-decode batch is mid-recording on cmd_pool's buffers. */
	VkCommandPool   xfer_pool;
	VkCommandBuffer xfer_cmd;
	VkFence         xfer_fence;

	VkDescriptorSetLayout dset_layout;
	VkPipelineLayout      pipe_layout;
	VkDescriptorPool      dpool;
	VkDescriptorSet       dset;

	VkPipeline pipe_f32;
	VkPipeline pipe_bf16;
	VkPipeline pipe_q8; /* Q8_0-weight GEMM offload (gemm_q8 shader) */

	/* Resident VAE-decode path: a shared 4-binding descriptor layout, a pool of
	 * descriptor sets cycled per op, a dedicated command buffer that can batch
	 * many dispatches, and one pipeline per elementwise/conv/norm kernel. */
	int vae_available;
	int vae_recording; /* inside iris_gpu_batch_begin/end */

	/* Tensors freed mid-batch cannot have their buffers destroyed immediately:
	 * they are still bound to the recording command buffer, and destroying a
	 * bound resource invalidates it (unlike Metal, which retains resources).
	 * Defer destruction until the batch is submitted and has completed. */
	struct iris_gpu_tensor **vae_pending_free;
	int                      vae_pending_count;
	int                      vae_pending_cap;

	VkDescriptorSetLayout vae_dset_layout;
	VkPipelineLayout      vae_pipe_layout;
	VkDescriptorPool      vae_dpool;
	VkCommandBuffer       vae_cmd;
	VkPipeline            pipe_conv2d;
	VkPipeline            pipe_groupnorm;
	VkPipeline            pipe_swish;
	VkPipeline            pipe_add;
	VkPipeline            pipe_upsample;
	VkPipeline            pipe_leakyrelu; /* RealESRGAN upscaler */
	VkPipeline            pipe_scaleadd;  /* RealESRGAN upscaler */

	/* Resident Qwen3 text-encoder pipelines (share the resident layout). */
	int        qwen_available;
	VkPipeline pipe_qwen_linear;    /* bf16-weight GEMM (gemm_bf16 shader) */
	VkPipeline pipe_qwen_linear_q8; /* Q8_0-weight GEMM (gemm_q8 shader) */
	VkPipeline pipe_qwen_rmsnorm;
	VkPipeline pipe_qwen_headrms;
	VkPipeline pipe_qwen_rope;
	VkPipeline pipe_qwen_attn;
	VkPipeline pipe_qwen_silumul;

	/* Resident transformer (denoising) op pipelines, reused by the Flux blocks. */
	int        res_available;
	VkPipeline pipe_res_rmsnorm;
	VkPipeline pipe_res_qkrmsnorm;
	VkPipeline pipe_res_rope;
	VkPipeline pipe_res_attn;
	VkPipeline pipe_res_split;
	VkPipeline pipe_res_gatedadd;
	VkPipeline pipe_res_adaln;
	VkPipeline pipe_res_linear_f32;

	/* Resident Flux transformer: one extra strided-copy pipeline on top of the
	 * resident ops above (which the Flux blocks reuse). */
	int        flux_available;
	VkPipeline pipe_flux_rowcopy;

	/* Per-call device buffers for dynamic f32 vectors (norm weights, gates,
	 * AdaLN shift/scale). Each upload gets a fresh buffer via vkCmdUpdateBuffer
	 * so values recorded earlier in a deferred batch are not aliased; the index
	 * resets at the start of each op/batch and the buffers are reused. */
	vk_buffer_t *dyn_bufs;
	int          dyn_count; /* allocated buffers */
	int          dyn_next;  /* next free index within the current op/batch */

	/* Compute buffers bound to the shader. On discrete GPUs these are
	 * device-local (VRAM) and fed through the host-visible staging buffers
	 * below; on unified-memory GPUs they are host-visible and written directly. */
	vk_buffer_t buf_a;
	vk_buffer_t buf_b; /* scratch for non-cached B */
	vk_buffer_t buf_c;

	/* Host-visible staging buffers (discrete path only). */
	int         discrete; /* 1 -> stage host memory through VRAM */
	vk_buffer_t stage_a;
	vk_buffer_t stage_b;
	vk_buffer_t stage_c;
	vk_buffer_t stage_w; /* scratch for one-time weight uploads */

	/* Cached weight buffers. */
	weight_entry_t *weights;
	int             weights_count;
	int             weights_cap;

	char device_name[256];
} g = {0};

/* Push constants must match the shader layout exactly. */
typedef struct {
	uint32_t M, N, K;
	uint32_t lda, ldb, ldc;
	uint32_t ta, tb;
	float    alpha, beta;
} gemm_pc_t;

typedef struct {
	uint32_t batch, in_ch, out_ch, H, W, kH, kW, stride, pad, outH, outW, has_bias, circular;
} conv_pc_t;

typedef struct {
	uint32_t batch, channels, spatial, num_groups;
	float    eps;
} gnorm_pc_t;

typedef struct {
	uint32_t n;
} unary_pc_t;
typedef struct {
	uint32_t channels, H, W;
} upsample_pc_t;
typedef struct {
	uint32_t n;
	float    slope;
} leaky_pc_t;
typedef struct {
	uint32_t n;
	float    scale;
} scaleadd_pc_t;

typedef struct {
	uint32_t rows, hidden;
	float    eps;
} rmsnorm_pc_t;
typedef struct {
	uint32_t rows, num_heads, head_dim;
	float    eps;
} headrms_pc_t;
typedef struct {
	uint32_t seq, num_q_heads, num_kv_heads, head_dim;
} rope_pc_t;
typedef struct {
	uint32_t seq, num_heads, num_kv_heads, head_dim;
	float    scale;
} attn_pc_t;

/* Resident-op push constants. */
typedef struct {
	uint32_t seq, heads, head_dim;
	float    eps;
} res_qkrms_pc_t;
typedef struct {
	uint32_t seq, heads, head_dim;
} res_rope_pc_t;
typedef struct {
	uint32_t seq, num_heads, head_dim;
	float    scale;
} res_attn_pc_t;
typedef struct {
	uint32_t seq, width, fused_dim, n_streams;
} res_split_pc_t;
typedef struct {
	uint32_t seq, hidden;
} res_gatedadd_pc_t;
typedef struct {
	uint32_t seq, in_dim, out_dim, has_bias;
} res_linf32_pc_t;
typedef struct {
	uint32_t seq, w, src_stride, src_off, dst_stride, dst_off;
} flux_rowcopy_pc_t;

/* GPU-resident tensor: a device-local (or host-visible on unified) buffer.
 * Holds f32 elements unless is_f16 is set (bf16, 2 bytes/element). */
struct iris_gpu_tensor {
	vk_buffer_t buf;
	size_t      n;      /* element count */
	int         is_f16; /* 1 -> 2 bytes/element (bf16); else 4 (f32) */
	float      *shadow; /* lazy host readback for iris_gpu_tensor_data (discrete) */
};

/* ---------------------------------------------------------------------- */

static int find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want, uint32_t *out_index) {
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
		if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
			*out_index = i;
			return 1;
		}
	}
	return 0;
}

/* Create or grow a host-visible, persistently mapped storage buffer. */
/* Create or grow a buffer. host_visible buffers are persistently mapped and
 * used for CPU staging; otherwise the buffer is device-local (VRAM). */
static int buffer_ensure(vk_buffer_t *b, VkDeviceSize bytes, VkBufferUsageFlags usage, int host_visible) {
	if (bytes == 0)
		bytes = 16;
	if (b->buffer != VK_NULL_HANDLE && b->capacity >= bytes)
		return 1;

	/* Grow geometrically to avoid frequent reallocation as sizes ratchet up. */
	if (b->buffer != VK_NULL_HANDLE && b->capacity * 2 > bytes)
		bytes = b->capacity * 2;

	if (b->buffer != VK_NULL_HANDLE) {
		if (b->map) {
			vkUnmapMemory(g.device, b->mem);
			b->map = NULL;
		}
		vkDestroyBuffer(g.device, b->buffer, NULL);
		vkFreeMemory(g.device, b->mem, NULL);
		b->buffer = VK_NULL_HANDLE;
	}

	VkBufferCreateInfo bi = {0};
	bi.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size               = bytes;
	bi.usage              = usage;
	bi.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
	VK_OK(vkCreateBuffer(g.device, &bi, NULL, &b->buffer));

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(g.device, b->buffer, &req);

	uint32_t              mtype;
	VkMemoryPropertyFlags want =
	    host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	if (!find_memory_type(req.memoryTypeBits, want, &mtype)) {
		/* Fall back to host-visible if no pure device-local type exists. */
		if (host_visible || !find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &mtype)) {
			fprintf(stderr, "Vulkan: no suitable memory type\n");
			return 0;
		}
	}

	VkMemoryAllocateInfo ai = {0};
	ai.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize       = req.size;
	ai.memoryTypeIndex      = mtype;
	VK_OK(vkAllocateMemory(g.device, &ai, NULL, &b->mem));
	VK_OK(vkBindBufferMemory(g.device, b->buffer, b->mem, 0));
	if (host_visible)
		VK_OK(vkMapMemory(g.device, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
	b->capacity = req.size;
	return 1;
}

/* Submit a recorded command buffer and block on the given fence. */
static void submit_wait_buf_fence(VkCommandBuffer cmd, VkFence fence) {
	VkSubmitInfo sub       = {0};
	sub.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	sub.commandBufferCount = 1;
	sub.pCommandBuffers    = &cmd;
	vkResetFences(g.device, 1, &fence);
	vkQueueSubmit(g.queue, 1, &sub, fence);
	vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX);
}

/* Submit a recorded command buffer and block until it completes. */
static void submit_wait_buf(VkCommandBuffer cmd) {
	submit_wait_buf_fence(cmd, g.fence);
}

/* Submit g.cmd (already recorded) and block until it completes. */
static void submit_wait(void) {
	submit_wait_buf(g.cmd);
}

static void buffer_destroy(vk_buffer_t *b) {
	if (b->buffer == VK_NULL_HANDLE)
		return;
	if (b->map) {
		vkUnmapMemory(g.device, b->mem);
		b->map = NULL;
	}
	vkDestroyBuffer(g.device, b->buffer, NULL);
	vkFreeMemory(g.device, b->mem, NULL);
	b->buffer   = VK_NULL_HANDLE;
	b->capacity = 0;
}

/* Cheap content fingerprint over sampled bytes. Used to detect when a CPU
 * pointer has been freed and reused for a different weight (the model streams
 * block weights in mmap mode, so addresses get recycled). */
static uint64_t weight_fingerprint(const void *data, size_t bytes) {
	/* Hash whole 32-bit words, not single bytes: bf16-derived f32 weights have
	 * zero low bytes, so byte sampling at 4-aligned strides would collide. */
	const uint32_t *p       = (const uint32_t *)data;
	size_t          n       = bytes / 4;
	uint64_t        h       = 1469598103934665603ULL; /* FNV-1a offset basis */
	size_t          samples = 8192;
	size_t          step    = n > samples ? n / samples : 1;
	for (size_t i = 0; i < n; i += step) {
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	h ^= bytes;
	h *= 1099511628211ULL;
	return h;
}

/* Usage flags for a shader-bound compute buffer. On the staged (discrete) path
 * it must also be a transfer source/destination so data can move to/from VRAM. */
static VkBufferUsageFlags compute_usage(void) {
	VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	if (g.discrete)
		u |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	return u;
}

/* Ensure a shader-bound compute buffer: device-local on the staged path,
 * host-visible (written directly) on the unified path. */
static int compute_buffer_ensure(vk_buffer_t *b, VkDeviceSize bytes) {
	return buffer_ensure(b, bytes, compute_usage(), g.discrete ? 0 : 1);
}

/* Blocking host->device copy through the weight staging buffer. Used for the
 * one-time upload of cached weights into device-local memory. */
#define STAGE_W_USAGE (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)

static int device_upload(vk_buffer_t *dev, const void *data, size_t bytes) {
	if (!buffer_ensure(&g.stage_w, bytes, STAGE_W_USAGE, 1))
		return 0;
	memcpy(g.stage_w.map, data, bytes);

	VkCommandBufferBeginInfo bgn = {0};
	bgn.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bgn.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g.xfer_cmd, &bgn);
	VkBufferCopy region = {0, 0, bytes};
	vkCmdCopyBuffer(g.xfer_cmd, g.stage_w.buffer, dev->buffer, 1, &region);
	vkEndCommandBuffer(g.xfer_cmd);
	submit_wait_buf_fence(g.xfer_cmd, g.xfer_fence);
	return 1;
}

/* Blocking device->host copy through the weight staging buffer. */
static int device_download(vk_buffer_t *dev, void *out, size_t bytes) {
	if (!buffer_ensure(&g.stage_w, bytes, STAGE_W_USAGE, 1))
		return 0;

	VkCommandBufferBeginInfo bgn = {0};
	bgn.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bgn.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g.xfer_cmd, &bgn);
	/* Order prior compute/transfer writes (possibly in earlier submissions)
	 * before this readback copy. */
	VkMemoryBarrier mb      = {0};
	mb.sType                = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask        = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
	VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	vkCmdPipelineBarrier(g.xfer_cmd, st, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
	VkBufferCopy region = {0, 0, bytes};
	vkCmdCopyBuffer(g.xfer_cmd, dev->buffer, g.stage_w.buffer, 1, &region);
	vkEndCommandBuffer(g.xfer_cmd);
	submit_wait_buf_fence(g.xfer_cmd, g.xfer_fence);
	memcpy(out, g.stage_w.map, bytes);
	return 1;
}

/* Upload host data into a cached weight buffer. On the staged path the buffer
 * is device-local (VRAM-resident across the run); on the unified path it is
 * host-visible and written directly. Rounded up to 4 bytes so the bf16 shader
 * can index it as uints. */
static int weight_upload(vk_buffer_t *buf, const void *data, size_t copy_bytes) {
	size_t alloc_bytes = (copy_bytes + 3) & ~(size_t)3;
	if (g.discrete) {
		if (!buffer_ensure(buf, alloc_bytes, compute_usage(), 0))
			return 0;
		return device_upload(buf, data, copy_bytes);
	}
	if (!buffer_ensure(buf, alloc_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1))
		return 0;
	memcpy(buf->map, data, copy_bytes);
	return 1;
}

/* Look up (or create + upload) a cached weight buffer keyed by CPU pointer.
 * The fingerprint guards against pointer reuse: if the same address now holds
 * different content (freed + realloc'd weight), the entry is re-uploaded.
 * copy_bytes is the real data size; the buffer is rounded up to 4 bytes so the
 * bf16 shader can read it as a uint array. */
static vk_buffer_t *weight_get(const void *key, const void *data, size_t copy_bytes) {
	uint64_t fp = weight_fingerprint(data, copy_bytes);
	for (int i = 0; i < g.weights_count; i++) {
		if (g.weights[i].key != key)
			continue;
		weight_entry_t *e = &g.weights[i];
		if (e->fingerprint == fp && e->bytes == copy_bytes)
			return &e->buf; /* valid hit */
		/* Stale: address reused for a different weight -> re-upload. */
		if (!weight_upload(&e->buf, data, copy_bytes))
			return NULL;
		e->fingerprint = fp;
		e->bytes       = copy_bytes;
		return &e->buf;
	}
	if (g.weights_count == g.weights_cap) {
		int             nc = g.weights_cap ? g.weights_cap * 2 : 64;
		weight_entry_t *nw = realloc(g.weights, (size_t)nc * sizeof(*nw));
		if (!nw)
			return NULL;
		g.weights     = nw;
		g.weights_cap = nc;
	}
	weight_entry_t *e = &g.weights[g.weights_count];
	memset(e, 0, sizeof(*e));
	e->key = key;
	if (!weight_upload(&e->buf, data, copy_bytes))
		return NULL;
	e->fingerprint = fp;
	e->bytes       = copy_bytes;
	g.weights_count++;
	return &e->buf;
}

/* Release all GPU-resident cached weight buffers (VRAM), keeping the device,
 * pipelines and the weight-entry array alive. Used to reclaim the transformer's
 * weights before VAE decode in one-shot generation: the VAE re-uploads its own
 * weights lazily on the next weight_get(). Safe only between batches (never
 * while vae_recording). */
void iris_vulkan_release_weight_cache(void) {
	if (!g.initialized || g.device == VK_NULL_HANDLE)
		return;
	vkDeviceWaitIdle(g.device);
	for (int i = 0; i < g.weights_count; i++)
		buffer_destroy(&g.weights[i].buf);
	g.weights_count = 0;
}

static int create_pipeline(const uint32_t *spv, size_t spv_bytes, VkPipelineLayout layout, VkPipeline *out) {
	VkShaderModuleCreateInfo si = {0};
	si.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	si.codeSize                 = spv_bytes;
	si.pCode                    = spv;
	VkShaderModule module;
	VK_OK(vkCreateShaderModule(g.device, &si, NULL, &module));

	VkComputePipelineCreateInfo pi = {0};
	pi.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pi.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pi.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
	pi.stage.module                = module;
	pi.stage.pName                 = "main";
	pi.layout                      = layout;

	VkResult r = vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pi, NULL, out);
	vkDestroyShaderModule(g.device, module, NULL);
	if (r != VK_SUCCESS) {
		fprintf(stderr, "Vulkan: pipeline creation failed (%d)\n", (int)r);
		return 0;
	}
	return 1;
}

/* Max descriptor sets cycled within one batched VAE decode (one per dispatch). */
#define IRIS_VAE_MAX_SETS 4096

/* Build the resident VAE-decode pipelines + descriptor pool. Best-effort:
 * on any failure g.vae_available stays 0 and the decoder falls back to CPU. */
static void init_vae_resident(void) {
	g.vae_available = 0;

	/* 5 storage-buffer bindings: 4 are enough for VAE decode, the 5th lets the
	 * Qwen attention shader bind Q/K/V/out/mask in one set. Shaders that use
	 * fewer bindings simply leave the trailing ones unread. */
	VkDescriptorSetLayoutBinding binds[5];
	for (int i = 0; i < 5; i++) {
		binds[i]                 = (VkDescriptorSetLayoutBinding){0};
		binds[i].binding         = (uint32_t)i;
		binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		binds[i].descriptorCount = 1;
		binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dli = {0};
	dli.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount                    = 5;
	dli.pBindings                       = binds;
	if (vkCreateDescriptorSetLayout(g.device, &dli, NULL, &g.vae_dset_layout) != VK_SUCCESS)
		return;

	VkPushConstantRange pcr        = {0};
	pcr.stageFlags                 = VK_SHADER_STAGE_COMPUTE_BIT;
	pcr.offset                     = 0;
	pcr.size                       = 64; /* >= largest VAE push-constant struct */
	VkPipelineLayoutCreateInfo pli = {0};
	pli.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount             = 1;
	pli.pSetLayouts                = &g.vae_dset_layout;
	pli.pushConstantRangeCount     = 1;
	pli.pPushConstantRanges        = &pcr;
	if (vkCreatePipelineLayout(g.device, &pli, NULL, &g.vae_pipe_layout) != VK_SUCCESS)
		return;

	VkDescriptorPoolSize ps        = {0};
	ps.type                        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ps.descriptorCount             = IRIS_VAE_MAX_SETS * 5;
	VkDescriptorPoolCreateInfo dpi = {0};
	dpi.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets                    = IRIS_VAE_MAX_SETS;
	dpi.poolSizeCount              = 1;
	dpi.pPoolSizes                 = &ps;
	if (vkCreateDescriptorPool(g.device, &dpi, NULL, &g.vae_dpool) != VK_SUCCESS)
		return;

	VkCommandBufferAllocateInfo cbi = {0};
	cbi.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbi.commandPool                 = g.cmd_pool;
	cbi.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount          = 1;
	if (vkAllocateCommandBuffers(g.device, &cbi, &g.vae_cmd) != VK_SUCCESS)
		return;

	if (!create_pipeline(conv2d_spv, sizeof(conv2d_spv), g.vae_pipe_layout, &g.pipe_conv2d))
		return;
	if (!create_pipeline(groupnorm_spv, sizeof(groupnorm_spv), g.vae_pipe_layout, &g.pipe_groupnorm))
		return;
	if (!create_pipeline(swish_spv, sizeof(swish_spv), g.vae_pipe_layout, &g.pipe_swish))
		return;
	if (!create_pipeline(add_spv, sizeof(add_spv), g.vae_pipe_layout, &g.pipe_add))
		return;
	if (!create_pipeline(upsample_spv, sizeof(upsample_spv), g.vae_pipe_layout, &g.pipe_upsample))
		return;
	if (!create_pipeline(leakyrelu_spv, sizeof(leakyrelu_spv), g.vae_pipe_layout, &g.pipe_leakyrelu))
		return;
	if (!create_pipeline(scaleadd_spv, sizeof(scaleadd_spv), g.vae_pipe_layout, &g.pipe_scaleadd))
		return;

	g.vae_available = 1;

	/* Resident Qwen3 text-encoder pipelines. Best-effort: failure leaves
	 * g.qwen_available 0 and the text encoder falls back to GEMM offload. */
	g.qwen_available = 0;
	if (create_pipeline(gemm_bf16_spv, sizeof(gemm_bf16_spv), g.vae_pipe_layout, &g.pipe_qwen_linear) &&
	    create_pipeline(gemm_q8_spv, sizeof(gemm_q8_spv), g.vae_pipe_layout, &g.pipe_qwen_linear_q8) &&
	    create_pipeline(qwen_rmsnorm_spv, sizeof(qwen_rmsnorm_spv), g.vae_pipe_layout, &g.pipe_qwen_rmsnorm) &&
	    create_pipeline(qwen_headrms_spv, sizeof(qwen_headrms_spv), g.vae_pipe_layout, &g.pipe_qwen_headrms) &&
	    create_pipeline(qwen_rope_spv, sizeof(qwen_rope_spv), g.vae_pipe_layout, &g.pipe_qwen_rope) &&
	    create_pipeline(qwen_attn_spv, sizeof(qwen_attn_spv), g.vae_pipe_layout, &g.pipe_qwen_attn) &&
	    create_pipeline(qwen_silumul_spv, sizeof(qwen_silumul_spv), g.vae_pipe_layout, &g.pipe_qwen_silumul)) {
		g.qwen_available = 1;
	}

	/* Resident transformer op pipelines. Best-effort: failure leaves
	 * g.res_available 0 and the transformer falls back to GEMM offload. */
	g.res_available = 0;
	if (create_pipeline(res_rmsnorm_spv, sizeof(res_rmsnorm_spv), g.vae_pipe_layout, &g.pipe_res_rmsnorm) &&
	    create_pipeline(res_qkrmsnorm_spv, sizeof(res_qkrmsnorm_spv), g.vae_pipe_layout, &g.pipe_res_qkrmsnorm) &&
	    create_pipeline(res_rope_spv, sizeof(res_rope_spv), g.vae_pipe_layout, &g.pipe_res_rope) &&
	    create_pipeline(res_attn_spv, sizeof(res_attn_spv), g.vae_pipe_layout, &g.pipe_res_attn) &&
	    create_pipeline(res_split_spv, sizeof(res_split_spv), g.vae_pipe_layout, &g.pipe_res_split) &&
	    create_pipeline(res_gatedadd_spv, sizeof(res_gatedadd_spv), g.vae_pipe_layout, &g.pipe_res_gatedadd) &&
	    create_pipeline(res_adaln_spv, sizeof(res_adaln_spv), g.vae_pipe_layout, &g.pipe_res_adaln) &&
	    create_pipeline(res_linear_f32_spv, sizeof(res_linear_f32_spv), g.vae_pipe_layout, &g.pipe_res_linear_f32)) {
		g.res_available = 1;
	}

	/* Resident Flux transformer: reuses the resident ops plus one strided copy.
	 * Best-effort: failure leaves g.flux_available 0 and the transformer falls
	 * back to GEMM offload. */
	g.flux_available = 0;
	if (g.res_available && create_pipeline(flux_rowcopy_spv, sizeof(flux_rowcopy_spv), g.vae_pipe_layout, &g.pipe_flux_rowcopy)) {
		g.flux_available = 1;
	}
}

/* ---------------------------------------------------------------------- */

int iris_vulkan_init(void) {
	if (g.initialized)
		return g.available;
	g.initialized = 1;
	g.available   = 0;

	/* Escape hatch: force the CPU path (for debugging / unsupported drivers). */
	if (getenv("IRIS_VK_DISABLE")) {
		if (iris_verbose)
			fprintf(stderr, "Vulkan: disabled via IRIS_VK_DISABLE\n");
		return 0;
	}

	VkApplicationInfo app = {0};
	app.sType             = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName  = "iris";
	app.apiVersion        = VK_API_VERSION_1_0;

	VkInstanceCreateInfo ici = {0};
	ici.sType                = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ici.pApplicationInfo     = &app;
	if (vkCreateInstance(&ici, NULL, &g.instance) != VK_SUCCESS) {
		if (iris_verbose)
			fprintf(stderr, "Vulkan: no instance (loader/driver missing)\n");
		return 0;
	}

	uint32_t n = 0;
	vkEnumeratePhysicalDevices(g.instance, &n, NULL);
	if (n == 0) {
		if (iris_verbose)
			fprintf(stderr, "Vulkan: no physical devices\n");
		return 0;
	}
	VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
	vkEnumeratePhysicalDevices(g.instance, &n, devs);

	/* Pick the best device that exposes a compute queue (prefer discrete). */
	int best_score = -1;
	g.phys         = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t qn = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
		VkQueueFamilyProperties *qf = malloc(qn * sizeof(*qf));
		vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf);
		int cf = -1;
		for (uint32_t q = 0; q < qn; q++) {
			if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				cf = (int)q;
				break;
			}
		}
		free(qf);
		if (cf < 0)
			continue;

		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(devs[i], &props);
		int score = 1;
		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			score = 4;
		else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
			score = 3;
		else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
			score = 2;
		if (score > best_score) {
			best_score     = score;
			g.phys         = devs[i];
			g.queue_family = (uint32_t)cf;
			g.discrete     = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
			snprintf(g.device_name, sizeof(g.device_name), "%s", props.deviceName);
		}
	}
	free(devs);
	if (g.phys == VK_NULL_HANDLE) {
		if (iris_verbose)
			fprintf(stderr, "Vulkan: no compute-capable device\n");
		return 0;
	}

	/* On a discrete GPU, keep compute buffers in device-local VRAM and feed them
	 * through host-visible staging buffers. On unified memory the host-visible
	 * buffers are already device-local, so we write them directly (zero-copy).
	 * IRIS_VK_NOSTAGE forces the direct path (debugging). */
	if (getenv("IRIS_VK_NOSTAGE"))
		g.discrete = 0;
	if (iris_verbose)
		fprintf(stderr, "Vulkan: %s (%s memory path)\n", g.device_name, g.discrete ? "staged VRAM" : "direct host-visible");

	float                   prio = 1.0f;
	VkDeviceQueueCreateInfo qci  = {0};
	qci.sType                    = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qci.queueFamilyIndex         = g.queue_family;
	qci.queueCount               = 1;
	qci.pQueuePriorities         = &prio;

	VkDeviceCreateInfo dci   = {0};
	dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos    = &qci;
	VK_OK(vkCreateDevice(g.phys, &dci, NULL, &g.device));
	vkGetDeviceQueue(g.device, g.queue_family, 0, &g.queue);

	VkCommandPoolCreateInfo cpi = {0};
	cpi.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpi.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpi.queueFamilyIndex        = g.queue_family;
	VK_OK(vkCreateCommandPool(g.device, &cpi, NULL, &g.cmd_pool));

	VkCommandBufferAllocateInfo cbi = {0};
	cbi.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbi.commandPool                 = g.cmd_pool;
	cbi.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount          = 1;
	VK_OK(vkAllocateCommandBuffers(g.device, &cbi, &g.cmd));

	VkFenceCreateInfo fi = {0};
	fi.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VK_OK(vkCreateFence(g.device, &fi, NULL, &g.fence));

	/* Separate pool/buffer/fence for staging copies (see struct comment). */
	VK_OK(vkCreateCommandPool(g.device, &cpi, NULL, &g.xfer_pool));
	VkCommandBufferAllocateInfo xbi = cbi;
	xbi.commandPool                 = g.xfer_pool;
	VK_OK(vkAllocateCommandBuffers(g.device, &xbi, &g.xfer_cmd));
	VK_OK(vkCreateFence(g.device, &fi, NULL, &g.xfer_fence));

	/* Descriptor set layout: 3 storage buffers (A, B, C). */
	VkDescriptorSetLayoutBinding binds[3];
	for (int i = 0; i < 3; i++) {
		binds[i]                 = (VkDescriptorSetLayoutBinding){0};
		binds[i].binding         = (uint32_t)i;
		binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		binds[i].descriptorCount = 1;
		binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dli = {0};
	dli.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount                    = 3;
	dli.pBindings                       = binds;
	VK_OK(vkCreateDescriptorSetLayout(g.device, &dli, NULL, &g.dset_layout));

	VkPushConstantRange pcr = {0};
	pcr.stageFlags          = VK_SHADER_STAGE_COMPUTE_BIT;
	pcr.offset              = 0;
	pcr.size                = sizeof(gemm_pc_t);

	VkPipelineLayoutCreateInfo pli = {0};
	pli.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount             = 1;
	pli.pSetLayouts                = &g.dset_layout;
	pli.pushConstantRangeCount     = 1;
	pli.pPushConstantRanges        = &pcr;
	VK_OK(vkCreatePipelineLayout(g.device, &pli, NULL, &g.pipe_layout));

	VkDescriptorPoolSize ps        = {0};
	ps.type                        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ps.descriptorCount             = 3;
	VkDescriptorPoolCreateInfo dpi = {0};
	dpi.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets                    = 1;
	dpi.poolSizeCount              = 1;
	dpi.pPoolSizes                 = &ps;
	VK_OK(vkCreateDescriptorPool(g.device, &dpi, NULL, &g.dpool));

	VkDescriptorSetAllocateInfo dsa = {0};
	dsa.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsa.descriptorPool              = g.dpool;
	dsa.descriptorSetCount          = 1;
	dsa.pSetLayouts                 = &g.dset_layout;
	VK_OK(vkAllocateDescriptorSets(g.device, &dsa, &g.dset));

	if (!create_pipeline(gemm_f32_spv, sizeof(gemm_f32_spv), g.pipe_layout, &g.pipe_f32))
		return 0;
	if (!create_pipeline(gemm_bf16_spv, sizeof(gemm_bf16_spv), g.pipe_layout, &g.pipe_bf16))
		return 0;
	if (!create_pipeline(gemm_q8_spv, sizeof(gemm_q8_spv), g.pipe_layout, &g.pipe_q8))
		return 0;

	g.available = 1;

	/* Resident VAE-decode resources. Failure here is non-fatal: the GEMM offload
	 * still works and iris_vae.c falls back to the CPU decoder. */
	init_vae_resident();

	return 1;
}

int iris_vulkan_available(void) {
	return g.available;
}

const char *iris_vulkan_device_name(void) {
	return g.device_name[0] ? g.device_name : "Vulkan device";
}

/* ---------------------------------------------------------------------- */

static void bind_descriptors(VkBuffer a, VkBuffer b, VkBuffer c) {
	VkDescriptorBufferInfo bi[3];
	bi[0] = (VkDescriptorBufferInfo){a, 0, VK_WHOLE_SIZE};
	bi[1] = (VkDescriptorBufferInfo){b, 0, VK_WHOLE_SIZE};
	bi[2] = (VkDescriptorBufferInfo){c, 0, VK_WHOLE_SIZE};
	VkWriteDescriptorSet w[3];
	for (int i = 0; i < 3; i++) {
		w[i]                 = (VkWriteDescriptorSet){0};
		w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w[i].dstSet          = g.dset;
		w[i].dstBinding      = (uint32_t)i;
		w[i].descriptorCount = 1;
		w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		w[i].pBufferInfo     = &bi[i];
	}
	vkUpdateDescriptorSets(g.device, 3, w, 0, NULL);
}

/* Core GEMM dispatch shared by all entry points.
 * wtype selects B's storage: 0 = f32, 1 = bf16, 2 = GGML Q8_0 block-quantized.
 * b_cached: cache B by pointer (weights). */
static void vk_gemm(int wtype, int b_cached, int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const void *B,
                    int ldb, float beta, float *C, int ldc) {
	/* Minimal footprints: the last valid element is (rows-1)*ld + cols. Using
	 * exactly this avoids reading/writing past caller buffers when the matrix
	 * is a strided view into a larger one (e.g. per-head attention slices). */
	size_t a_elems = transpose_a ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K;
	size_t b_elems = transpose_b ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N;
	size_t c_elems = (size_t)(M - 1) * ldc + N;

	size_t a_bytes = a_elems * 4;
	size_t c_bytes = c_elems * 4;
	/* Q8_0 packs 32 elements into a 34-byte block; b_elems is a multiple of 32
	 * for weight matrices (contraction dim is block-aligned). */
	size_t b_copy = (wtype == 2) ? (b_elems / 32) * 34 : (wtype == 1) ? (b_elems * 2) : (b_elems * 4);

	int staged = g.discrete;

	/* Shader-bound buffers: device-local VRAM on the staged path, host-visible
	 * on unified memory. On the staged path the host writes land in the staging
	 * buffers and are DMA'd into VRAM inside the same command buffer. */
	if (!compute_buffer_ensure(&g.buf_a, a_bytes))
		return;
	if (!compute_buffer_ensure(&g.buf_c, c_bytes))
		return;
	if (staged) {
		if (!buffer_ensure(&g.stage_a, a_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1))
			return;
		if (!buffer_ensure(&g.stage_c, c_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1))
			return;
	}

	void *a_host = staged ? g.stage_a.map : g.buf_a.map;
	void *c_host = staged ? g.stage_c.map : g.buf_c.map;

	memcpy(a_host, A, a_bytes);
	if (beta != 0.0f) {
		if (ldc == N) {
			memcpy(c_host, C, c_bytes);
		}
		else {
			for (int r = 0; r < M; r++)
				memcpy((float *)c_host + (size_t)r * ldc, C + (size_t)r * ldc, (size_t)N * 4);
		}
	}

	VkBuffer b_handle;
	if (b_cached) {
		vk_buffer_t *wb = weight_get(B, B, b_copy);
		if (!wb)
			return;
		b_handle = wb->buffer;
	}
	else {
		size_t b_alloc = (b_copy + 3) & ~(size_t)3;
		if (!compute_buffer_ensure(&g.buf_b, b_alloc))
			return;
		if (staged) {
			if (!buffer_ensure(&g.stage_b, b_alloc, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1))
				return;
			memcpy(g.stage_b.map, B, b_copy);
		}
		else {
			memcpy(g.buf_b.map, B, b_copy);
		}
		b_handle = g.buf_b.buffer;
	}

	bind_descriptors(g.buf_a.buffer, b_handle, g.buf_c.buffer);

	gemm_pc_t pc;
	pc.M     = (uint32_t)M;
	pc.N     = (uint32_t)N;
	pc.K     = (uint32_t)K;
	pc.lda   = (uint32_t)lda;
	pc.ldb   = (uint32_t)ldb;
	pc.ldc   = (uint32_t)ldc;
	pc.ta    = transpose_a ? 1u : 0u;
	pc.tb    = transpose_b ? 1u : 0u;
	pc.alpha = alpha;
	pc.beta  = beta;

	VkCommandBufferBeginInfo bgn = {0};
	bgn.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bgn.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g.cmd, &bgn);

	if (staged) {
		VkBufferCopy cp = {0, 0, a_bytes};
		vkCmdCopyBuffer(g.cmd, g.stage_a.buffer, g.buf_a.buffer, 1, &cp);
		if (!b_cached) {
			VkBufferCopy cpb = {0, 0, b_copy};
			vkCmdCopyBuffer(g.cmd, g.stage_b.buffer, g.buf_b.buffer, 1, &cpb);
		}
		if (beta != 0.0f) {
			VkBufferCopy cpc = {0, 0, c_bytes};
			vkCmdCopyBuffer(g.cmd, g.stage_c.buffer, g.buf_c.buffer, 1, &cpc);
		}
		VkMemoryBarrier mb = {0};
		mb.sType           = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		mb.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
		mb.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
	}

	vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (wtype == 2) ? g.pipe_q8 : (wtype == 1) ? g.pipe_bf16 : g.pipe_f32);
	vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_layout, 0, 1, &g.dset, 0, NULL);
	vkCmdPushConstants(g.cmd, g.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	uint32_t gx = ((uint32_t)N + 15u) / 16u;
	uint32_t gy = ((uint32_t)M + 15u) / 16u;
	vkCmdDispatch(g.cmd, gx, gy, 1);

	if (staged) {
		VkMemoryBarrier mb = {0};
		mb.sType           = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		mb.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
		mb.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
		VkBufferCopy cp = {0, 0, c_bytes};
		vkCmdCopyBuffer(g.cmd, g.buf_c.buffer, g.stage_c.buffer, 1, &cp);
	}

	vkEndCommandBuffer(g.cmd);
	submit_wait();

	/* Strided writeback: copy only the N valid columns per row so gaps in a
	 * strided output view (ldc > N) are left untouched. */
	const float *c_src = staged ? (const float *)g.stage_c.map : (const float *)g.buf_c.map;
	if (ldc == N) {
		memcpy(C, c_src, c_bytes);
	}
	else {
		for (int r = 0; r < M; r++)
			memcpy(C + (size_t)r * ldc, c_src + (size_t)r * ldc, (size_t)N * 4);
	}
}

void iris_vulkan_sgemm(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                       float *C, int ldc) {
	vk_gemm(0, 0, transpose_a, transpose_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void iris_vulkan_sgemm_cached(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                              float *C, int ldc) {
	vk_gemm(0, 1, transpose_a, transpose_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void iris_vulkan_sgemm_bf16(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const uint16_t *B_bf16, int ldb,
                            float beta, float *C, int ldc) {
	vk_gemm(1, 1, transpose_a, transpose_b, M, N, K, alpha, A, lda, B_bf16, ldb, beta, C, ldc);
}

void iris_vulkan_sgemm_q8(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const void *B_q8, int ldb, float beta,
                          float *C, int ldc) {
	vk_gemm(2, 1, transpose_a, transpose_b, M, N, K, alpha, A, lda, B_q8, ldb, beta, C, ldc);
}

/* ========================================================================
 * Resident VAE-decode tensor API (mirrors the iris_gpu_* Metal surface used
 * by vae_decode_gpu in iris_vae.c). Data stays in GPU buffers between ops;
 * iris_gpu_batch_begin/end fold many dispatches into one submit.
 * ======================================================================== */

/* Allocate a resident tensor buffer (device-local on discrete, host-visible on
 * unified). Sized exactly; transfer-capable for upload/readback/copy. */
static int tensor_buf_alloc(vk_buffer_t *b, size_t bytes) {
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	return buffer_ensure(b, bytes, usage, g.discrete ? 0 : 1);
}

/* Begin recording a VAE op. In a batch this returns the shared command buffer
 * (already recording); standalone, it resets the per-op descriptor pool and
 * opens a fresh one-shot recording. */
static VkCommandBuffer vae_cmd_begin(void) {
	if (g.vae_recording)
		return g.vae_cmd;
	vkResetDescriptorPool(g.device, g.vae_dpool, 0);
	g.dyn_next                   = 0; /* standalone op: dynamic buffers free to reuse */
	VkCommandBufferBeginInfo bgn = {0};
	bgn.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bgn.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g.vae_cmd, &bgn);
	return g.vae_cmd;
}

/* Finish a standalone VAE op (submit + wait). No-op inside a batch. */
static void vae_cmd_finish(void) {
	if (g.vae_recording)
		return;
	vkEndCommandBuffer(g.vae_cmd);
	submit_wait_buf(g.vae_cmd);
}

/* Conservative compute+transfer barrier between batched ops (the previous op
 * may have written any buffer the next one reads). Serializes ops within a
 * batch but keeps a single submit. */
static void vae_barrier(VkCommandBuffer cmd) {
	VkMemoryBarrier mb      = {0};
	mb.sType                = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask        = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	vkCmdPipelineBarrier(cmd, st, st, 0, 1, &mb, 0, NULL, 0, NULL);
}

static VkDescriptorSet vae_alloc_set(void) {
	VkDescriptorSetAllocateInfo ai = {0};
	ai.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool              = g.vae_dpool;
	ai.descriptorSetCount          = 1;
	ai.pSetLayouts                 = &g.vae_dset_layout;
	VkDescriptorSet set            = VK_NULL_HANDLE;
	vkAllocateDescriptorSets(g.device, &ai, &set);
	return set;
}

/* Record a single compute dispatch binding 5 storage buffers + push constants,
 * with a 2D workgroup grid (gx, gy). Shaders that use fewer than 5 bindings
 * leave the trailing ones unread. */
static void resident_dispatch(VkPipeline pipe, const VkBuffer b[5], const void *pc, uint32_t pc_size, uint32_t gx, uint32_t gy) {
	VkCommandBuffer cmd = vae_cmd_begin();
	VkDescriptorSet set = vae_alloc_set();
	if (set == VK_NULL_HANDLE) {
		vae_cmd_finish();
		return;
	}

	VkDescriptorBufferInfo bi[5];
	VkWriteDescriptorSet   w[5];
	for (int i = 0; i < 5; i++) {
		bi[i]                = (VkDescriptorBufferInfo){b[i], 0, VK_WHOLE_SIZE};
		w[i]                 = (VkWriteDescriptorSet){0};
		w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w[i].dstSet          = set;
		w[i].dstBinding      = (uint32_t)i;
		w[i].descriptorCount = 1;
		w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		w[i].pBufferInfo     = &bi[i];
	}
	vkUpdateDescriptorSets(g.device, 5, w, 0, NULL);

	/* Make prior producers (previous batch ops and staged uploads, possibly in
	 * earlier submissions on this queue) available to this dispatch's reads. */
	vae_barrier(cmd);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.vae_pipe_layout, 0, 1, &set, 0, NULL);
	vkCmdPushConstants(cmd, g.vae_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
	vkCmdDispatch(cmd, gx, gy, 1);

	vae_cmd_finish();
}

/* Record a single compute dispatch binding 4 storage buffers + push constants.
 * group_count is the workgroup count along x. */
static void vae_dispatch(VkPipeline pipe, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3, const void *pc, uint32_t pc_size, uint32_t group_count) {
	VkBuffer bufs[5] = {b0, b1, b2, b3, b3};
	resident_dispatch(pipe, bufs, pc, pc_size, group_count, 1);
}

/* Workgroup count for a 1D grid-stride dispatch, capped to the guaranteed
 * 65535 limit (the shaders loop to cover any remainder). */
static uint32_t vae_groups(size_t total) {
	size_t g_ = (total + 255) / 256;
	if (g_ > 65535)
		g_ = 65535;
	if (g_ == 0)
		g_ = 1;
	return (uint32_t)g_;
}

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements) {
	if (!g.vae_available)
		return NULL;
	struct iris_gpu_tensor *t = calloc(1, sizeof(*t));
	if (!t)
		return NULL;
	t->n         = num_elements;
	size_t bytes = num_elements * 4;
	if (!tensor_buf_alloc(&t->buf, bytes)) {
		free(t);
		return NULL;
	}
	if (g.discrete) {
		if (!device_upload(&t->buf, data, bytes)) {
			buffer_destroy(&t->buf);
			free(t);
			return NULL;
		}
	}
	else {
		memcpy(t->buf.map, data, bytes);
	}
	return t;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements) {
	if (!g.vae_available)
		return NULL;
	struct iris_gpu_tensor *t = calloc(1, sizeof(*t));
	if (!t)
		return NULL;
	t->n = num_elements;
	if (!tensor_buf_alloc(&t->buf, num_elements * 4)) {
		free(t);
		return NULL;
	}
	return t;
}

/* Destroy all tensors deferred during a batch. Call only after the batch's
 * command buffer has been submitted and waited on. */
static void vae_flush_pending_free(void) {
	for (int i = 0; i < g.vae_pending_count; i++) {
		iris_gpu_tensor_t t = g.vae_pending_free[i];
		buffer_destroy(&t->buf);
		free(t->shadow);
		free(t);
	}
	g.vae_pending_count = 0;
}

void iris_gpu_tensor_free(iris_gpu_tensor_t t) {
	if (!t)
		return;
	/* Inside a batch the buffer may still be bound to the recording command
	 * buffer; defer destruction until iris_gpu_batch_end() submits + waits. */
	if (g.vae_recording) {
		if (g.vae_pending_count == g.vae_pending_cap) {
			int                nc = g.vae_pending_cap ? g.vae_pending_cap * 2 : 64;
			iris_gpu_tensor_t *np = realloc(g.vae_pending_free, (size_t)nc * sizeof(*np));
			if (np) {
				g.vae_pending_free = np;
				g.vae_pending_cap  = nc;
			}
		}
		if (g.vae_pending_count < g.vae_pending_cap) {
			g.vae_pending_free[g.vae_pending_count++] = t;
			return;
		}
		/* Allocation failed: fall through to immediate destroy. This can
		 * invalidate the command buffer, but only under OOM. */
	}
	buffer_destroy(&t->buf);
	free(t->shadow);
	free(t);
}

void iris_gpu_tensor_read(iris_gpu_tensor_t t, float *out) {
	if (!t)
		return;
	size_t bytes = t->n * 4;
	if (g.discrete)
		device_download(&t->buf, out, bytes);
	else
		memcpy(out, t->buf.map, bytes);
}

void iris_gpu_tensor_write(iris_gpu_tensor_t t, const float *data) {
	if (!t)
		return;
	size_t bytes = t->n * 4;
	if (g.discrete)
		device_upload(&t->buf, data, bytes);
	else
		memcpy(t->buf.map, data, bytes);
}

void iris_gpu_copy_f32(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n) {
	if (!g.vae_available || !dst || !src)
		return;
	VkCommandBuffer cmd = vae_cmd_begin();
	vae_barrier(cmd); /* see vae_dispatch: order prior producers before this copy */
	VkBufferCopy region = {0, 0, n * 4};
	vkCmdCopyBuffer(cmd, src->buf.buffer, dst->buf.buffer, 1, &region);
	vae_cmd_finish();
}

iris_gpu_tensor_t iris_gpu_conv2d_f32(iris_gpu_tensor_t x, const float *weight, const float *bias, int batch, int in_ch, int out_ch, int H, int W, int kH,
                                      int kW, int stride, int padding) {
	if (!g.vae_available || !x)
		return NULL;
	int    outH  = (H + 2 * padding - kH) / stride + 1;
	int    outW  = (W + 2 * padding - kW) / stride + 1;
	size_t out_n = (size_t)batch * out_ch * outH * outW;

	iris_gpu_tensor_t out = iris_gpu_tensor_alloc(out_n);
	if (!out)
		return NULL;

	/* Weights/bias resident in VRAM, cached by pointer across the run. */
	vk_buffer_t *wb = weight_get(weight, weight, (size_t)out_ch * in_ch * kH * kW * 4);
	if (!wb) {
		iris_gpu_tensor_free(out);
		return NULL;
	}
	VkBuffer bias_buf;
	if (bias) {
		vk_buffer_t *bb = weight_get(bias, bias, (size_t)out_ch * 4);
		if (!bb) {
			iris_gpu_tensor_free(out);
			return NULL;
		}
		bias_buf = bb->buffer;
	}
	else {
		bias_buf = x->buf.buffer; /* unused by shader when has_bias == 0 */
	}

	conv_pc_t pc = {(uint32_t)batch,  (uint32_t)in_ch,   (uint32_t)out_ch, (uint32_t)H,    (uint32_t)W,    (uint32_t)kH,           (uint32_t)kW,
	                (uint32_t)stride, (uint32_t)padding, (uint32_t)outH,   (uint32_t)outW, bias ? 1u : 0u, iris_circular ? 1u : 0u};
	vae_dispatch(g.pipe_conv2d, x->buf.buffer, wb->buffer, bias_buf, out->buf.buffer, &pc, sizeof(pc), vae_groups(out_n));
	return out;
}

void iris_gpu_group_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *gamma, const float *beta, int batch, int channels, int spatial,
                             int num_groups, float eps) {
	if (!g.vae_available || !out || !x)
		return;
	vk_buffer_t *gw = weight_get(gamma, gamma, (size_t)channels * 4);
	vk_buffer_t *bw = weight_get(beta, beta, (size_t)channels * 4);
	if (!gw || !bw)
		return;
	gnorm_pc_t pc = {(uint32_t)batch, (uint32_t)channels, (uint32_t)spatial, (uint32_t)num_groups, eps};
	/* One workgroup per (batch, group). */
	vae_dispatch(g.pipe_groupnorm, x->buf.buffer, gw->buffer, bw->buffer, out->buf.buffer, &pc, sizeof(pc), (uint32_t)(batch * num_groups));
}

void iris_gpu_swish_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n) {
	if (!g.vae_available || !out || !x)
		return;
	unary_pc_t pc = {(uint32_t)n};
	vae_dispatch(g.pipe_swish, x->buf.buffer, out->buf.buffer, x->buf.buffer, x->buf.buffer, &pc, sizeof(pc), vae_groups((size_t)n));
}

void iris_gpu_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n) {
	if (!g.vae_available || !out || !a || !b)
		return;
	unary_pc_t pc = {(uint32_t)n};
	vae_dispatch(g.pipe_add, a->buf.buffer, b->buf.buffer, out->buf.buffer, a->buf.buffer, &pc, sizeof(pc), vae_groups((size_t)n));
}

iris_gpu_tensor_t iris_gpu_upsample_nearest_2x_f32(iris_gpu_tensor_t x, int channels, int H, int W) {
	if (!g.vae_available || !x)
		return NULL;
	size_t            out_n = (size_t)channels * (H * 2) * (W * 2);
	iris_gpu_tensor_t out   = iris_gpu_tensor_alloc(out_n);
	if (!out)
		return NULL;
	upsample_pc_t pc = {(uint32_t)channels, (uint32_t)H, (uint32_t)W};
	vae_dispatch(g.pipe_upsample, x->buf.buffer, out->buf.buffer, x->buf.buffer, x->buf.buffer, &pc, sizeof(pc), vae_groups(out_n));
	return out;
}

void iris_gpu_leaky_relu_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n, float slope) {
	if (!g.vae_available || !out || !x)
		return;
	leaky_pc_t pc = {(uint32_t)n, slope};
	vae_dispatch(g.pipe_leakyrelu, x->buf.buffer, out->buf.buffer, x->buf.buffer, x->buf.buffer, &pc, sizeof(pc), vae_groups((size_t)n));
}

void iris_gpu_scale_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, float scale, int n) {
	if (!g.vae_available || !out || !a || !b)
		return;
	scaleadd_pc_t pc = {(uint32_t)n, scale};
	vae_dispatch(g.pipe_scaleadd, a->buf.buffer, b->buf.buffer, out->buf.buffer, a->buf.buffer, &pc, sizeof(pc), vae_groups((size_t)n));
}

/* ========================================================================
 * Resident Qwen3 text-encoder ops. Activations are f32 device tensors;
 * weights are bf16, cached on the GPU by pointer (stable mmap addresses).
 * Each op records one dispatch into the current batch (or runs standalone).
 * ======================================================================== */

int iris_vk_qwen_available(void) {
	return g.qwen_available;
}

void iris_vk_qwen_linear(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *weight_bf16, int seq, int in_dim, int out_dim) {
	if (!g.qwen_available || !out || !x || !weight_bf16)
		return;
	vk_buffer_t *wb = weight_get(weight_bf16, weight_bf16, (size_t)out_dim * in_dim * 2);
	if (!wb)
		return;
	gemm_pc_t pc;
	pc.M          = (uint32_t)seq;
	pc.N          = (uint32_t)out_dim;
	pc.K          = (uint32_t)in_dim;
	pc.lda        = (uint32_t)in_dim;
	pc.ldb        = (uint32_t)in_dim;
	pc.ldc        = (uint32_t)out_dim;
	pc.ta         = 0u;
	pc.tb         = 1u; /* C = x @ W^T (W is [out, in]) */
	pc.alpha      = 1.0f;
	pc.beta       = 0.0f;
	VkBuffer b[5] = {x->buf.buffer, wb->buffer, out->buf.buffer, out->buf.buffer, out->buf.buffer};
	resident_dispatch(g.pipe_qwen_linear, b, &pc, sizeof(pc), ((uint32_t)out_dim + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
}

void iris_vk_qwen_linear_q8(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const void *weight_q8, int seq, int in_dim, int out_dim) {
	if (!g.qwen_available || !out || !x || !weight_q8)
		return;
	/* GGML Q8_0: 34 bytes per 32-element block. */
	size_t       bytes = (size_t)out_dim * in_dim / 32 * 34;
	vk_buffer_t *wb    = weight_get(weight_q8, weight_q8, bytes);
	if (!wb)
		return;
	gemm_pc_t pc;
	pc.M          = (uint32_t)seq;
	pc.N          = (uint32_t)out_dim;
	pc.K          = (uint32_t)in_dim;
	pc.lda        = (uint32_t)in_dim;
	pc.ldb        = (uint32_t)in_dim;
	pc.ldc        = (uint32_t)out_dim;
	pc.ta         = 0u;
	pc.tb         = 1u; /* C = x @ W^T (W is [out, in]) */
	pc.alpha      = 1.0f;
	pc.beta       = 0.0f;
	VkBuffer b[5] = {x->buf.buffer, wb->buffer, out->buf.buffer, out->buf.buffer, out->buf.buffer};
	resident_dispatch(g.pipe_qwen_linear_q8, b, &pc, sizeof(pc), ((uint32_t)out_dim + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
}

void iris_vk_qwen_rms_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *weight_bf16, int seq, int hidden, float eps) {
	if (!g.qwen_available || !out || !x || !weight_bf16)
		return;
	vk_buffer_t *wb = weight_get(weight_bf16, weight_bf16, (size_t)hidden * 2);
	if (!wb)
		return;
	rmsnorm_pc_t pc   = {(uint32_t)seq, (uint32_t)hidden, eps};
	VkBuffer     b[5] = {x->buf.buffer, wb->buffer, out->buf.buffer, out->buf.buffer, out->buf.buffer};
	/* One workgroup per row. */
	resident_dispatch(g.pipe_qwen_rmsnorm, b, &pc, sizeof(pc), (uint32_t)seq, 1);
}

void iris_vk_qwen_head_rms_norm(iris_gpu_tensor_t t, const uint16_t *weight_bf16, int seq, int num_heads, int head_dim, float eps) {
	if (!g.qwen_available || !t || !weight_bf16)
		return;
	vk_buffer_t *wb = weight_get(weight_bf16, weight_bf16, (size_t)head_dim * 2);
	if (!wb)
		return;
	headrms_pc_t pc   = {(uint32_t)seq, (uint32_t)num_heads, (uint32_t)head_dim, eps};
	VkBuffer     b[5] = {t->buf.buffer, wb->buffer, t->buf.buffer, t->buf.buffer, t->buf.buffer};
	resident_dispatch(g.pipe_qwen_headrms, b, &pc, sizeof(pc), vae_groups((size_t)seq * num_heads), 1);
}

void iris_vk_qwen_rope(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_table, const float *sin_table, int seq, int num_q_heads, int num_kv_heads,
                       int head_dim) {
	if (!g.qwen_available || !q || !k || !cos_table || !sin_table)
		return;
	int half = head_dim / 2;
	/* cos/sin tables are stable for the run; only the first seq rows are read. */
	vk_buffer_t *cb = weight_get(cos_table, cos_table, (size_t)seq * half * 4);
	vk_buffer_t *sb = weight_get(sin_table, sin_table, (size_t)seq * half * 4);
	if (!cb || !sb)
		return;
	rope_pc_t pc    = {(uint32_t)seq, (uint32_t)num_q_heads, (uint32_t)num_kv_heads, (uint32_t)head_dim};
	VkBuffer  b[5]  = {q->buf.buffer, k->buf.buffer, cb->buffer, sb->buffer, q->buf.buffer};
	size_t    total = (size_t)seq * (num_q_heads + num_kv_heads) * half;
	resident_dispatch(g.pipe_qwen_rope, b, &pc, sizeof(pc), vae_groups(total), 1);
}

void iris_vk_qwen_attention(iris_gpu_tensor_t out, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t mask, int seq,
                            int num_heads, int num_kv_heads, int head_dim, float scale) {
	if (!g.qwen_available || !out || !q || !k || !v || !mask)
		return;
	attn_pc_t pc   = {(uint32_t)seq, (uint32_t)num_heads, (uint32_t)num_kv_heads, (uint32_t)head_dim, scale};
	VkBuffer  b[5] = {q->buf.buffer, k->buf.buffer, v->buf.buffer, out->buf.buffer, mask->buf.buffer};
	/* One workgroup per (query position, head). */
	resident_dispatch(g.pipe_qwen_attn, b, &pc, sizeof(pc), (uint32_t)seq, (uint32_t)num_heads);
}

void iris_vk_qwen_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n) {
	if (!g.qwen_available || !gate || !up)
		return;
	unary_pc_t pc   = {(uint32_t)n};
	VkBuffer   b[5] = {gate->buf.buffer, up->buf.buffer, gate->buf.buffer, gate->buf.buffer, gate->buf.buffer};
	resident_dispatch(g.pipe_qwen_silumul, b, &pc, sizeof(pc), vae_groups((size_t)n), 1);
}

/* ========================================================================
 * Resident transformer (denoising) ops. Activations are f32 device tensors;
 * immutable weights/tables are cached by pointer (weight_get), while dynamic
 * per-block f32 vectors are uploaded fresh per call (see dyn helper).
 * ======================================================================== */

int iris_vk_res_available(void) {
	return g.res_available;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements) {
	if (!g.vae_available)
		return NULL;
	struct iris_gpu_tensor *t = calloc(1, sizeof(*t));
	if (!t)
		return NULL;
	t->n      = num_elements;
	t->is_f16 = 1;
	if (!tensor_buf_alloc(&t->buf, num_elements * 2)) {
		free(t);
		return NULL;
	}
	return t;
}

float *iris_gpu_tensor_data(iris_gpu_tensor_t t) {
	if (!t)
		return NULL;
	if (!g.discrete)
		return (float *)t->buf.map;
	/* Discrete GPU: device-local memory is not host-mapped. Download into a
	 * lazily-allocated per-tensor shadow and return that. */
	size_t bytes = t->n * (t->is_f16 ? 2 : 4);
	if (!t->shadow) {
		t->shadow = malloc(bytes);
		if (!t->shadow)
			return NULL;
	}
	device_download(&t->buf, t->shadow, bytes);
	return t->shadow;
}

void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t t, int persistent) {
	(void)t;
	(void)persistent; /* no pool under Vulkan; tensors freed explicitly */
}

/* Fetch the next per-call dynamic buffer (>= bytes), recording an upload of
 * `data` into the given command buffer via vkCmdUpdateBuffer. The buffer is
 * distinct from any other dynamic upload live in the current batch, so values
 * recorded earlier are not aliased. bytes must be <= 65536 and a multiple of 4
 * (callers pass small f32 vectors). */
static VkBuffer dyn_upload(VkCommandBuffer cmd, const void *data, size_t bytes) {
	size_t alloc = (bytes + 3) & ~(size_t)3;
	if (g.dyn_next == g.dyn_count) {
		int          nc = g.dyn_count ? g.dyn_count * 2 : 64;
		vk_buffer_t *nb = realloc(g.dyn_bufs, (size_t)nc * sizeof(*nb));
		if (!nb)
			return VK_NULL_HANDLE;
		memset(nb + g.dyn_count, 0, (size_t)(nc - g.dyn_count) * sizeof(*nb));
		g.dyn_bufs  = nb;
		g.dyn_count = nc;
	}
	vk_buffer_t *b = &g.dyn_bufs[g.dyn_next];
	if (!buffer_ensure(b, alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, g.discrete ? 0 : 1))
		return VK_NULL_HANDLE;
	g.dyn_next++;
	vkCmdUpdateBuffer(cmd, b->buffer, 0, alloc, data);
	return b->buffer;
}

void iris_gpu_rms_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *weight, int seq, int hidden, float eps) {
	if (!g.res_available || !out || !x || !weight)
		return;
	VkCommandBuffer cmd  = vae_cmd_begin();
	VkBuffer        wbuf = dyn_upload(cmd, weight, (size_t)hidden * 4);
	if (wbuf == VK_NULL_HANDLE) {
		vae_cmd_finish();
		return;
	}
	VkBuffer     b[5] = {x->buf.buffer, wbuf, out->buf.buffer, out->buf.buffer, out->buf.buffer};
	rmsnorm_pc_t pc   = {(uint32_t)seq, (uint32_t)hidden, eps};
	resident_dispatch(g.pipe_res_rmsnorm, b, &pc, sizeof(pc), (uint32_t)seq, 1);
}

void iris_gpu_qk_rms_norm(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps) {
	if (!g.res_available || !q || !k || !q_weight || !k_weight)
		return;
	/* QK norm weights are immutable block weights -> cache by pointer. */
	vk_buffer_t *qw = weight_get(q_weight, q_weight, (size_t)head_dim * 4);
	vk_buffer_t *kw = weight_get(k_weight, k_weight, (size_t)head_dim * 4);
	if (!qw || !kw)
		return;
	res_qkrms_pc_t pc   = {(uint32_t)seq, (uint32_t)heads, (uint32_t)head_dim, eps};
	VkBuffer       b[5] = {q->buf.buffer, k->buf.buffer, qw->buffer, kw->buffer, q->buf.buffer};
	resident_dispatch(g.pipe_res_qkrmsnorm, b, &pc, sizeof(pc), vae_groups((size_t)seq * heads), 1);
}

void iris_gpu_rope_single_pair_f32(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim) {
	if (!g.res_available || !q || !k || !cos_freq || !sin_freq)
		return;
	/* cos/sin tables are stable for the image geometry -> cache by pointer. */
	vk_buffer_t *cb = weight_get(cos_freq, cos_freq, (size_t)seq * head_dim * 4);
	vk_buffer_t *sb = weight_get(sin_freq, sin_freq, (size_t)seq * head_dim * 4);
	if (!cb || !sb)
		return;
	res_rope_pc_t pc    = {(uint32_t)seq, (uint32_t)heads, (uint32_t)head_dim};
	VkBuffer      b[5]  = {q->buf.buffer, k->buf.buffer, cb->buffer, sb->buffer, q->buf.buffer};
	size_t        total = (size_t)seq * heads * (head_dim / 2);
	resident_dispatch(g.pipe_res_rope, b, &pc, sizeof(pc), vae_groups(total), 1);
}

void iris_gpu_split_qkv_mlp(iris_gpu_tensor_t fused, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t gate,
                            iris_gpu_tensor_t up, int seq, int hidden, int mlp_hidden) {
	if (!g.res_available || !fused)
		return;
	if (hidden > 0 && q && k && v) {
		res_split_pc_t pc   = {(uint32_t)seq, (uint32_t)hidden, (uint32_t)(3 * hidden), 3u};
		VkBuffer       b[5] = {fused->buf.buffer, q->buf.buffer, k->buf.buffer, v->buf.buffer, q->buf.buffer};
		resident_dispatch(g.pipe_res_split, b, &pc, sizeof(pc), ((uint32_t)hidden + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
	}
	if (mlp_hidden > 0 && gate && up) {
		res_split_pc_t pc   = {(uint32_t)seq, (uint32_t)mlp_hidden, (uint32_t)(2 * mlp_hidden), 2u};
		VkBuffer       b[5] = {fused->buf.buffer, gate->buf.buffer, up->buf.buffer, gate->buf.buffer, gate->buf.buffer};
		resident_dispatch(g.pipe_res_split, b, &pc, sizeof(pc), ((uint32_t)mlp_hidden + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
	}
}

void iris_gpu_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n) {
	iris_vk_qwen_silu_mul(gate, up, n); /* same SwiGLU kernel */
}

void iris_gpu_adaln_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *shift, const float *scale, int seq, int hidden, float eps) {
	if (!g.res_available || !out || !x || !shift || !scale)
		return;
	VkCommandBuffer cmd = vae_cmd_begin();
	VkBuffer        sh  = dyn_upload(cmd, shift, (size_t)hidden * 4);
	VkBuffer        sc  = dyn_upload(cmd, scale, (size_t)hidden * 4);
	if (sh == VK_NULL_HANDLE || sc == VK_NULL_HANDLE) {
		vae_cmd_finish();
		return;
	}
	VkBuffer     b[5] = {x->buf.buffer, sh, sc, out->buf.buffer, out->buf.buffer};
	rmsnorm_pc_t pc   = {(uint32_t)seq, (uint32_t)hidden, eps};
	resident_dispatch(g.pipe_res_adaln, b, &pc, sizeof(pc), (uint32_t)seq, 1);
}

void iris_gpu_gated_add(iris_gpu_tensor_t out, const float *gate, iris_gpu_tensor_t proj, int seq, int hidden) {
	if (!g.res_available || !out || !gate || !proj)
		return;
	VkCommandBuffer cmd  = vae_cmd_begin();
	VkBuffer        gbuf = dyn_upload(cmd, gate, (size_t)hidden * 4);
	if (gbuf == VK_NULL_HANDLE) {
		vae_cmd_finish();
		return;
	}
	VkBuffer          b[5] = {out->buf.buffer, gbuf, proj->buf.buffer, out->buf.buffer, out->buf.buffer};
	res_gatedadd_pc_t pc   = {(uint32_t)seq, (uint32_t)hidden};
	resident_dispatch(g.pipe_res_gatedadd, b, &pc, sizeof(pc), ((uint32_t)hidden + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
}

int iris_gpu_linear_bf16_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
	if (!g.res_available || !out || !x || !W_bf16)
		return 0;
	iris_vk_qwen_linear(out, x, W_bf16, seq_len, in_dim, out_dim);
	return 1;
}

int iris_gpu_linear_q8_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const void *W_q8, int seq_len, int in_dim, int out_dim) {
	if (!g.res_available || !out || !x || !W_q8)
		return 0;
	iris_vk_qwen_linear_q8(out, x, W_q8, seq_len, in_dim, out_dim);
	return 1;
}

iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
	if (!g.res_available || !x || !W_bf16)
		return NULL;
	iris_gpu_tensor_t out = iris_gpu_tensor_alloc((size_t)seq_len * out_dim);
	if (!out)
		return NULL;
	iris_vk_qwen_linear(out, x, W_bf16, seq_len, in_dim, out_dim);
	return out;
}

iris_gpu_tensor_t iris_gpu_linear(iris_gpu_tensor_t x, const float *W, const float *b_bias, int seq_len, int in_dim, int out_dim) {
	if (!g.res_available || !x || !W)
		return NULL;
	iris_gpu_tensor_t out = iris_gpu_tensor_alloc((size_t)seq_len * out_dim);
	if (!out)
		return NULL;
	vk_buffer_t *wb = weight_get(W, W, (size_t)out_dim * in_dim * 4);
	if (!wb) {
		iris_gpu_tensor_free(out);
		return NULL;
	}
	VkBuffer bias_buf;
	if (b_bias) {
		vk_buffer_t *bb = weight_get(b_bias, b_bias, (size_t)out_dim * 4);
		if (!bb) {
			iris_gpu_tensor_free(out);
			return NULL;
		}
		bias_buf = bb->buffer;
	}
	else {
		bias_buf = x->buf.buffer; /* unused by shader when has_bias == 0 */
	}
	res_linf32_pc_t pc   = {(uint32_t)seq_len, (uint32_t)in_dim, (uint32_t)out_dim, b_bias ? 1u : 0u};
	VkBuffer        b[5] = {x->buf.buffer, wb->buffer, bias_buf, out->buf.buffer, out->buf.buffer};
	resident_dispatch(g.pipe_res_linear_f32, b, &pc, sizeof(pc), ((uint32_t)out_dim + 15u) / 16u, ((uint32_t)seq_len + 15u) / 16u);
	return out;
}

int iris_gpu_attention_fused(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                             int head_dim, float scale) {
	if (!g.res_available || !out || !Q || !K || !V)
		return 0;
	if (seq_q != seq_k)
		return 0; /* full square self-attention only */
	res_attn_pc_t pc   = {(uint32_t)seq_q, (uint32_t)num_heads, (uint32_t)head_dim, scale};
	VkBuffer      b[5] = {Q->buf.buffer, K->buf.buffer, V->buf.buffer, out->buf.buffer, out->buf.buffer};
	resident_dispatch(g.pipe_res_attn, b, &pc, sizeof(pc), (uint32_t)seq_q, (uint32_t)num_heads);
	return 1;
}

/* bf16 attention / conversion paths are unimplemented under Vulkan: returning 0
 * makes the caller use the f32 attention path above. */
int iris_gpu_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                            int head_dim, float scale) {
	(void)out;
	(void)Q;
	(void)K;
	(void)V;
	(void)seq_q;
	(void)seq_k;
	(void)num_heads;
	(void)head_dim;
	(void)scale;
	return 0;
}
int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads,
                                  int head_dim, float scale) {
	(void)out;
	(void)Q;
	(void)K;
	(void)V;
	(void)seq_q;
	(void)seq_k;
	(void)num_heads;
	(void)head_dim;
	(void)scale;
	return 0;
}
int iris_gpu_convert_f32_to_bf16_into(iris_gpu_tensor_t bf16_out, iris_gpu_tensor_t f32_in) {
	(void)bf16_out;
	(void)f32_in;
	return 0;
}
int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t f32_out, iris_gpu_tensor_t bf16_in) {
	(void)f32_out;
	(void)bf16_in;
	return 0;
}

void iris_gpu_copy_region_f32(iris_gpu_tensor_t dst, size_t dst_offset, iris_gpu_tensor_t src, size_t src_offset, size_t n) {
	if (!g.vae_available || !dst || !src)
		return;
	VkCommandBuffer cmd = vae_cmd_begin();
	vae_barrier(cmd); /* order prior producers before this copy */
	VkBufferCopy region = {src_offset * 4, dst_offset * 4, n * 4};
	vkCmdCopyBuffer(cmd, src->buf.buffer, dst->buf.buffer, 1, &region);
	vae_cmd_finish();
}

/* ========================================================================
 * Resident Flux transformer ops. The Flux blocks reuse the resident op
 * surface (adaln/qk-rmsnorm/rope/attention/linear/gated-add/silu); this adds
 * the strided row-copy used for fused-projection split and attn|mlp concat.
 * ======================================================================== */

int iris_vk_flux_available(void) {
	return g.flux_available;
}

void iris_gpu_row_copy_f32(iris_gpu_tensor_t dst, int dst_stride, int dst_off, iris_gpu_tensor_t src, int src_stride, int src_off, int seq, int w) {
	if (!g.flux_available || !dst || !src)
		return;
	flux_rowcopy_pc_t pc   = {(uint32_t)seq, (uint32_t)w, (uint32_t)src_stride, (uint32_t)src_off, (uint32_t)dst_stride, (uint32_t)dst_off};
	VkBuffer          b[5] = {src->buf.buffer, dst->buf.buffer, dst->buf.buffer, dst->buf.buffer, dst->buf.buffer};
	resident_dispatch(g.pipe_flux_rowcopy, b, &pc, sizeof(pc), ((uint32_t)w + 15u) / 16u, ((uint32_t)seq + 15u) / 16u);
}

void iris_gpu_batch_begin(void) {
	if (!g.vae_available || g.vae_recording)
		return;
	vkResetDescriptorPool(g.device, g.vae_dpool, 0);
	g.dyn_next                   = 0; /* new batch: dynamic buffers free to reuse */
	VkCommandBufferBeginInfo bgn = {0};
	bgn.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bgn.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(g.vae_cmd, &bgn);
	g.vae_recording = 1;
}

void iris_gpu_batch_end(void) {
	if (!g.vae_available || !g.vae_recording)
		return;
	vkEndCommandBuffer(g.vae_cmd);
	g.vae_recording = 0;
	submit_wait_buf(g.vae_cmd);
	/* Batch has completed on the GPU; safe to destroy deferred buffers. */
	vae_flush_pending_free();
}

void iris_vulkan_cleanup(void) {
	if (!g.initialized || g.device == VK_NULL_HANDLE)
		return;
	vkDeviceWaitIdle(g.device);

	g.vae_recording = 0;
	vae_flush_pending_free();
	free(g.vae_pending_free);
	g.vae_pending_free = NULL;
	g.vae_pending_cap  = 0;

	for (int i = 0; i < g.weights_count; i++)
		buffer_destroy(&g.weights[i].buf);
	free(g.weights);
	g.weights       = NULL;
	g.weights_count = g.weights_cap = 0;

	for (int i = 0; i < g.dyn_count; i++)
		buffer_destroy(&g.dyn_bufs[i]);
	free(g.dyn_bufs);
	g.dyn_bufs  = NULL;
	g.dyn_count = g.dyn_next = 0;

	buffer_destroy(&g.buf_a);
	buffer_destroy(&g.buf_b);
	buffer_destroy(&g.buf_c);
	buffer_destroy(&g.stage_a);
	buffer_destroy(&g.stage_b);
	buffer_destroy(&g.stage_c);
	buffer_destroy(&g.stage_w);

	if (g.pipe_f32)
		vkDestroyPipeline(g.device, g.pipe_f32, NULL);
	if (g.pipe_bf16)
		vkDestroyPipeline(g.device, g.pipe_bf16, NULL);
	if (g.pipe_conv2d)
		vkDestroyPipeline(g.device, g.pipe_conv2d, NULL);
	if (g.pipe_groupnorm)
		vkDestroyPipeline(g.device, g.pipe_groupnorm, NULL);
	if (g.pipe_swish)
		vkDestroyPipeline(g.device, g.pipe_swish, NULL);
	if (g.pipe_add)
		vkDestroyPipeline(g.device, g.pipe_add, NULL);
	if (g.pipe_upsample)
		vkDestroyPipeline(g.device, g.pipe_upsample, NULL);
	if (g.pipe_leakyrelu)
		vkDestroyPipeline(g.device, g.pipe_leakyrelu, NULL);
	if (g.pipe_scaleadd)
		vkDestroyPipeline(g.device, g.pipe_scaleadd, NULL);
	if (g.pipe_qwen_linear)
		vkDestroyPipeline(g.device, g.pipe_qwen_linear, NULL);
	if (g.pipe_qwen_rmsnorm)
		vkDestroyPipeline(g.device, g.pipe_qwen_rmsnorm, NULL);
	if (g.pipe_qwen_headrms)
		vkDestroyPipeline(g.device, g.pipe_qwen_headrms, NULL);
	if (g.pipe_qwen_rope)
		vkDestroyPipeline(g.device, g.pipe_qwen_rope, NULL);
	if (g.pipe_qwen_attn)
		vkDestroyPipeline(g.device, g.pipe_qwen_attn, NULL);
	if (g.pipe_qwen_silumul)
		vkDestroyPipeline(g.device, g.pipe_qwen_silumul, NULL);
	if (g.pipe_res_rmsnorm)
		vkDestroyPipeline(g.device, g.pipe_res_rmsnorm, NULL);
	if (g.pipe_res_qkrmsnorm)
		vkDestroyPipeline(g.device, g.pipe_res_qkrmsnorm, NULL);
	if (g.pipe_res_rope)
		vkDestroyPipeline(g.device, g.pipe_res_rope, NULL);
	if (g.pipe_res_attn)
		vkDestroyPipeline(g.device, g.pipe_res_attn, NULL);
	if (g.pipe_res_split)
		vkDestroyPipeline(g.device, g.pipe_res_split, NULL);
	if (g.pipe_res_gatedadd)
		vkDestroyPipeline(g.device, g.pipe_res_gatedadd, NULL);
	if (g.pipe_res_adaln)
		vkDestroyPipeline(g.device, g.pipe_res_adaln, NULL);
	if (g.pipe_res_linear_f32)
		vkDestroyPipeline(g.device, g.pipe_res_linear_f32, NULL);
	if (g.pipe_flux_rowcopy)
		vkDestroyPipeline(g.device, g.pipe_flux_rowcopy, NULL);
	if (g.vae_dpool)
		vkDestroyDescriptorPool(g.device, g.vae_dpool, NULL);
	if (g.vae_pipe_layout)
		vkDestroyPipelineLayout(g.device, g.vae_pipe_layout, NULL);
	if (g.vae_dset_layout)
		vkDestroyDescriptorSetLayout(g.device, g.vae_dset_layout, NULL);
	if (g.dpool)
		vkDestroyDescriptorPool(g.device, g.dpool, NULL);
	if (g.pipe_layout)
		vkDestroyPipelineLayout(g.device, g.pipe_layout, NULL);
	if (g.dset_layout)
		vkDestroyDescriptorSetLayout(g.device, g.dset_layout, NULL);
	if (g.fence)
		vkDestroyFence(g.device, g.fence, NULL);
	if (g.xfer_fence)
		vkDestroyFence(g.device, g.xfer_fence, NULL);
	if (g.cmd_pool)
		vkDestroyCommandPool(g.device, g.cmd_pool, NULL);
	if (g.xfer_pool)
		vkDestroyCommandPool(g.device, g.xfer_pool, NULL);
	vkDestroyDevice(g.device, NULL);
	vkDestroyInstance(g.instance, NULL);
	memset(&g, 0, sizeof(g));
}
