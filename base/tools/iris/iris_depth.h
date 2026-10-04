/*
 * iris_depth.h - Depth Anything 3 (DA3MONO-LARGE) monocular depth estimation
 *
 * Self-contained inference for the Depth Anything 3 "mono" model
 * (DA3MONO-LARGE): a DINOv2 ViT-L/14 backbone feeding a DPT-style head.
 * Loads weights from a single safetensors file (all tensors F32) and
 * produces a single-channel relative-depth map for an RGB image.
 *
 * The heavy transformer matrix multiplies are offloaded to the Vulkan
 * GEMM backend when it is available; otherwise everything runs on the CPU.
 */

#ifndef IRIS_DEPTH_H
#define IRIS_DEPTH_H

#include "iris.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iris_depth iris_depth_t;

/*
 * Load the Depth Anything V2 model from a safetensors file.
 * Returns NULL on error (use iris_get_error() for details).
 */
iris_depth_t *iris_depth_load(const char *path);

/*
 * Enable "tileable" mode: remove the perspective front-to-back brightness
 * ramp the model bakes into the depth, then feather opposite edges together,
 * so that depth estimated from a seamless top-down texture also tiles
 * seamlessly. Off by default.
 */
void iris_depth_set_tileable(iris_depth_t *model, int on);

/*
 * Free the model and its resources.
 */
void iris_depth_free(iris_depth_t *model);

/*
 * Estimate depth for an RGB(A) image. Alpha is ignored. The result is a
 * single-channel (grayscale) image with the same dimensions as the input,
 * where brighter pixels are nearer to the camera (normalized to 0..255).
 * Returns a newly allocated image (free with iris_image_free), NULL on error.
 */
iris_image *iris_depth_estimate(iris_depth_t *model, const iris_image *input);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_DEPTH_H */
