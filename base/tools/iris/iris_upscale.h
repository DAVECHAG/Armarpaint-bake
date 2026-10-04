/*
 * iris_upscale.h - RealESRGAN_x4plus image upscaler (RRDBNet)
 *
 * Self-contained, CPU-only inference for the RealESRGAN_x4plus model
 * (RRDBNet: scale=4, num_feat=64, num_block=23, num_grow_ch=32).
 * Loads weights from a single safetensors file and upscales an RGB image
 * by 4x.
 */

#ifndef IRIS_UPSCALE_H
#define IRIS_UPSCALE_H

#include "iris.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iris_upscale iris_upscale_t;

/*
 * Load the RealESRGAN_x4plus model from a safetensors file.
 * Returns NULL on error (use iris_get_error() for details).
 */
iris_upscale_t *iris_upscale_load(const char *path);

/*
 * Enable "tileable" mode: subtract the smoothest periodic field that makes the
 * upscaled image wrap seamlessly, so a tileable input stays tileable after 4x
 * upscaling. Off by default.
 */
void iris_upscale_set_tileable(iris_upscale_t *model, int on);

/*
 * Free the model and its resources.
 */
void iris_upscale_free(iris_upscale_t *model);

/*
 * Upscale an RGB(A) image by 4x. Alpha is ignored; the result is always
 * a 3-channel RGB image with dimensions (4*width, 4*height).
 * Returns a newly allocated image (free with iris_image_free), NULL on error.
 */
iris_image *iris_upscale_run(iris_upscale_t *model, const iris_image *input);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_UPSCALE_H */
