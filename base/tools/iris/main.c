/*
 * Iris CLI Application
 *
 * Command-line interface for image generation.
 *
 * Usage:
 *   iris -d model/ -p "prompt" -o output.png [options]
 *
 * Options:
 *   -d, --dir PATH        Path to model directory (safetensors)
 *   -p, --prompt TEXT     Text prompt for generation
 *   -o, --output PATH     Output image path
 *   -W, --width N         Output width (default: 256)
 *   -H, --height N        Output height (default: 256)
 *   -s, --steps N         Number of sampling steps (default: 4)
 *   -S, --seed N          Random seed (-1 for random)
 *   -i, --input PATH      Input image for img2img
 *   -q, --quiet           No output, just generate
 *   -v, --verbose         Extra detailed output
 *   -h, --help            Show help
 */

#include "iris.h"
#include "iris_depth.h"
#include "iris_kernels.h"
#include "iris_upscale.h"
#ifdef _WIN32
#define IRIS_NEED_GETOPT
#include "iris_compat.h"
#else
#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

#ifdef USE_VULKAN
#include "iris_vulkan.h"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h> /* sysctlbyname: used by Metal/BLAS banners */
#endif

/* ========================================================================
 * Verbosity Levels
 * ======================================================================== */

typedef enum {
	OUTPUT_QUIET   = 0, /* No output */
	OUTPUT_NORMAL  = 1, /* Progress and essential info */
	OUTPUT_VERBOSE = 2  /* Detailed debugging info */
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

/* ========================================================================
 * CLI Progress Callbacks
 * ======================================================================== */

static int cli_current_step   = 0;
static int cli_legend_printed = 0;

/* Called at the start of each sampling step */
static void cli_step_callback(int step, int total) {
	if (output_level == OUTPUT_QUIET)
		return;

	/* Print legend before first step */
	if (!cli_legend_printed) {
		fprintf(stderr, "Denoising (d=double block, s=single blocks, F=final):\n");
		cli_legend_printed = 1;
	}

	/* Print newline to end previous step's progress (if any) */
	if (cli_current_step > 0) {
		fprintf(stderr, "\n");
	}
	cli_current_step = step;
	fprintf(stderr, "  Step %d/%d ", step, total);
	fflush(stderr);
}

/* Called for each substep within transformer forward */
static void cli_substep_callback(iris_substep_type_t type, int index, int total) {
	if (output_level == OUTPUT_QUIET)
		return;
	(void)total;

	switch (type) {
	case IRIS_SUBSTEP_DOUBLE_BLOCK:
		fputc('d', stderr);
		break;
	case IRIS_SUBSTEP_SINGLE_BLOCK:
		/* Print 's' every 5 single blocks to avoid too much output */
		if ((index + 1) % 5 == 0) {
			fputc('s', stderr);
		}
		break;
	case IRIS_SUBSTEP_FINAL_LAYER:
		fputc('F', stderr);
		break;
	}
	fflush(stderr);
}

/* Track phase timing (wall-clock) */
static struct timeval cli_phase_start_tv;
static const char    *cli_current_phase = NULL;

/* Called at phase boundaries (encoding text, decoding image, etc.) */
static void cli_phase_callback(const char *phase, int done) {
	if (output_level == OUTPUT_QUIET)
		return;

	if (!done) {
		/* If we were showing step progress, end that line first */
		if (cli_current_step > 0) {
			fprintf(stderr, "\n");
			cli_current_step = 0;
		}

		/* Phase starting */
		cli_current_phase = phase;
		gettimeofday(&cli_phase_start_tv, NULL);

		/* Capitalize first letter for display */
		char display[64];
		strncpy(display, phase, sizeof(display) - 1);
		display[sizeof(display) - 1] = '\0';
		if (display[0] >= 'a' && display[0] <= 'z') {
			display[0] -= 32;
		}

		fprintf(stderr, "%s...", display);
		fflush(stderr);
	}
	else {
		/* Phase finished */
		struct timeval now;
		gettimeofday(&now, NULL);
		double elapsed = (now.tv_sec - cli_phase_start_tv.tv_sec) + (now.tv_usec - cli_phase_start_tv.tv_usec) / 1000000.0;
		fprintf(stderr, " done (%.1fs)\n", elapsed);
		cli_current_phase = NULL;
	}
}

/* Set up CLI progress callbacks */
static void cli_setup_progress(void) {
	cli_current_step      = 0;
	cli_legend_printed    = 0;
	cli_current_phase     = NULL;
	iris_step_callback    = cli_step_callback;
	iris_substep_callback = cli_substep_callback;
	iris_phase_callback   = cli_phase_callback;
}

/* Clean up after generation (print final newline) */
static void cli_finish_progress(void) {
	if (cli_current_step > 0) {
		fprintf(stderr, "\n");
		cli_current_step = 0;
	}
	iris_step_callback    = NULL;
	iris_substep_callback = NULL;
	iris_phase_callback   = NULL;
}

/* ========================================================================
 * Timing Helper (wall-clock time)
 * ======================================================================== */

static struct timeval timer_start_tv;

static void timer_begin(void) {
	gettimeofday(&timer_start_tv, NULL);
}

static double timer_end(void) {
	struct timeval now;
	gettimeofday(&now, NULL);
	return (now.tv_sec - timer_start_tv.tv_sec) + (now.tv_usec - timer_start_tv.tv_usec) / 1000000.0;
}

/* ========================================================================
 * Output Helpers
 * ======================================================================== */

/* Print if not quiet */
#define LOG_NORMAL(...)                    \
	do {                                   \
		if (output_level >= OUTPUT_NORMAL) \
			fprintf(stderr, __VA_ARGS__);  \
	} while (0)

/* Print only in verbose mode */
#define LOG_VERBOSE(...)                    \
	do {                                    \
		if (output_level >= OUTPUT_VERBOSE) \
			fprintf(stderr, __VA_ARGS__);   \
	} while (0)

/* ========================================================================
 * Usage and Help
 * ======================================================================== */

/* Default values */
#define DEFAULT_WIDTH    256
#define DEFAULT_HEIGHT   256
#define DEFAULT_STEPS    4
#define MAX_INPUT_IMAGES 16

static void print_usage(const char *prog) {
	fprintf(stderr, "Iris - C Image Generation\n\n");
	fprintf(stderr, "Usage: %s [options]\n\n", prog);
	fprintf(stderr, "Required:\n");
	fprintf(stderr, "  -d, --dir PATH        Path to model directory\n");
	fprintf(stderr, "  -p, --prompt TEXT     Text prompt for generation\n");
	fprintf(stderr, "  -o, --output PATH     Output image path (.png, .ppm)\n\n");
	fprintf(stderr, "Generation options:\n");
	fprintf(stderr, "  -W, --width N         Output width (default: %d)\n", DEFAULT_WIDTH);
	fprintf(stderr, "  -H, --height N        Output height (default: %d)\n", DEFAULT_HEIGHT);
	fprintf(stderr, "  -s, --steps N         Sampling steps (default: auto, 4 distilled / 50 base)\n");
	fprintf(stderr, "  -g, --guidance N      CFG guidance scale (default: auto, 1.0 distilled / 4.0 base)\n");
	fprintf(stderr, "  -S, --seed N          Random seed (-1 for random)\n");
	fprintf(stderr, "      --linear          Use linear timestep schedule\n");
	fprintf(stderr, "      --power           Use power curve timestep schedule (default alpha: 2.0)\n");
	fprintf(stderr, "      --power-alpha N   Set power schedule exponent (default: 2.0)\n");
	fprintf(stderr, "      --sigmoid         Use Flux shifted sigmoid schedule\n\n");
	fprintf(stderr, "Model options:\n");
	fprintf(stderr, "      --base            Force base model mode (undistilled, CFG enabled)\n\n");
	fprintf(stderr, "Reference images (img2img / multi-reference):\n");
	fprintf(stderr, "  -i, --input PATH      Reference image (can specify up to %d)\n", MAX_INPUT_IMAGES);
	fprintf(stderr, "                        Multiple -i flags combine images via in-context conditioning\n\n");
	fprintf(stderr, "      --vary N          Variation amount for img2img (0.0-1.0). Re-noises the input\n");
	fprintf(stderr, "                        and denoises it: low = close to original, high = more change.\n");
	fprintf(stderr, "                        Uses traditional img2img (single -i) instead of in-context.\n\n");
	fprintf(stderr, "Inpainting / object removal (img2img only):\n");
	fprintf(stderr, "      --mask PATH       Black & white mask for img2img: white areas are regenerated,\n");
	fprintf(stderr, "                        black areas are kept from the input image. Requires one -i input.\n");
	fprintf(stderr, "                        No effect in text2img, --upscale or --depth modes.\n\n");
	fprintf(stderr, "Upscaling:\n");
	fprintf(stderr, "      --upscale         4x upscale input (-i) via RealESRGAN_x4plus, write to -o\n");
	fprintf(stderr, "                        (uses <model-dir>/RealESRGAN_x4plus.safetensors; no prompt needed)\n\n");
	fprintf(stderr, "Depth estimation:\n");
	fprintf(stderr, "      --depth           Estimate depth from input (-i) via Depth Anything 3, write to -o\n");
	fprintf(stderr, "                        (uses <model-dir>/da3-mono-large.safetensors; no prompt needed)\n\n");
	fprintf(stderr, "Seamless tiling:\n");
	fprintf(stderr, "      --tile            Make the output seamlessly tileable.\n");
	fprintf(stderr, "                        Generation: circular conv padding.\n");
	fprintf(stderr, "                        With -i (img2img): convert the photo to a seamless tile by\n");
	fprintf(stderr, "                        re-synthesizing it with circular padding, staying close to the\n");
	fprintf(stderr, "                        original. --vary sets the strength (default 0.45): lower\n");
	fprintf(stderr, "                        stays closer to the source, higher closes the seam harder.\n");
	fprintf(stderr, "                        With --upscale: keep a tileable input seamless after 4x.\n");
	fprintf(stderr, "                        With --depth: also flatten the perspective tilt.\n\n");
	fprintf(stderr, "Output options:\n");
	fprintf(stderr, "  -q, --quiet           Silent mode, no output\n");
	fprintf(stderr, "  -v, --verbose         Detailed output\n\n");
	fprintf(stderr, "Other options:\n");
	fprintf(stderr, "  -m, --mmap            Use memory-mapped weights (default, fastest on MPS)\n");
	fprintf(stderr, "      --no-mmap         Disable mmap, load all weights upfront\n");
	fprintf(stderr, "      --vae-tiling      Decode the VAE in overlapping tiles (lower memory, large images)\n");
	fprintf(stderr, "      --no-license-info Suppress non-commercial license warning\n");
	fprintf(stderr, "  -h, --help            Show this help\n\n");
	fprintf(stderr, "Examples:\n");
	fprintf(stderr, "  %s -d model/ -p \"a cat on a rainbow\" -o cat.png\n", prog);
	fprintf(stderr, "  %s -d model/ -p \"oil painting\" -i photo.png -o art.png\n", prog);
	fprintf(stderr, "  %s -d model/ -p \"combine them\" -i car.png -i beach.png -o result.png\n", prog);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
#ifdef USE_METAL
	iris_metal_init();
#endif
#ifdef USE_VULKAN
	iris_vulkan_init();
#endif

	/* Command line options */
	static struct option long_options[] = {{"dir", required_argument, 0, 'd'},
	                                       {"prompt", required_argument, 0, 'p'},
	                                       {"output", required_argument, 0, 'o'},
	                                       {"width", required_argument, 0, 'W'},
	                                       {"height", required_argument, 0, 'H'},
	                                       {"steps", required_argument, 0, 's'},
	                                       {"guidance", required_argument, 0, 'g'},
	                                       {"seed", required_argument, 0, 'S'},
	                                       {"input", required_argument, 0, 'i'},
	                                       {"quiet", no_argument, 0, 'q'},
	                                       {"verbose", no_argument, 0, 'v'},
	                                       {"help", no_argument, 0, 'h'},
	                                       {"version", no_argument, 0, 'V'},
	                                       {"mmap", no_argument, 0, 'm'},
	                                       {"no-mmap", no_argument, 0, 'M'},
	                                       {"base", no_argument, 0, 'B'},
	                                       {"linear", no_argument, 0, 'L'},
	                                       {"power", no_argument, 0, 256},
	                                       {"power-alpha", required_argument, 0, 257},
	                                       {"sigmoid", no_argument, 0, 260},
	                                       {"vae-tiling", no_argument, 0, 261},
	                                       {"upscale", no_argument, 0, 262},
	                                       {"depth", no_argument, 0, 263},
	                                       {"tile", no_argument, 0, 264},
	                                       {"mask", required_argument, 0, 265},
	                                       {"vary", required_argument, 0, 266},
	                                       {"no-license-info", no_argument, 0, 258},
	                                       {0, 0, 0, 0}};

	/* Parse arguments */
	char *model_dir                     = NULL;
	char *prompt                        = NULL;
	char *output_path                   = NULL;
	char *input_paths[MAX_INPUT_IMAGES] = {NULL};
	int   num_inputs                    = 0;
	char *mask_path                     = NULL;
	float variance                      = -1.0f; /* <0 = unset (use in-context img2img) */

	iris_params params = {.width       = DEFAULT_WIDTH,
	                      .height      = DEFAULT_HEIGHT,
	                      .num_steps   = 0, /* 0 = auto from model type */
	                      .seed        = -1,
	                      .guidance    = 0.0f, /* 0 = auto from model type */
	                      .power_alpha = 2.0f,
	                      .circular    = 0};

	int width_set = 0, height_set = 0, steps_set = 0;
	int use_mmap        = 1; /* mmap is default (fastest on MPS) */
	int force_base      = 0;
	int no_license_info = 0;
	int upscale_mode    = 0;
	int depth_mode      = 0;
	int tileable        = 0;

	int opt;
	while ((opt = getopt_long(argc, argv, "d:p:o:W:H:s:g:S:i:t:qvhVmMD", long_options, NULL)) != -1) {
		switch (opt) {
		case 'd':
			model_dir = optarg;
			break;
		case 'p':
			prompt = optarg;
			break;
		case 'o':
			output_path = optarg;
			break;
		case 'W':
			params.width = atoi(optarg);
			width_set    = 1;
			break;
		case 'H':
			params.height = atoi(optarg);
			height_set    = 1;
			break;
		case 's':
			params.num_steps = atoi(optarg);
			steps_set        = 1;
			break;
		case 'g':
			params.guidance = atof(optarg);
			break;
		case 'S':
			params.seed = atoll(optarg);
			break;
		case 'i':
			if (num_inputs < MAX_INPUT_IMAGES) {
				input_paths[num_inputs++] = optarg;
			}
			else {
				fprintf(stderr, "Warning: Maximum %d input images supported\n", MAX_INPUT_IMAGES);
			}
			break;
		case 'q':
			output_level = OUTPUT_QUIET;
			break;
		case 'v':
			output_level = OUTPUT_VERBOSE;
			iris_verbose = 1;
			break;
		case 'h':
			print_usage(argv[0]);
			return 0;
		case 'V':
			fprintf(stderr, "Iris v1.0.0\n");
			return 0;
		case 'm':
			use_mmap = 1;
			break;
		case 'M':
			use_mmap = 0;
			break;
		case 'B':
			force_base = 1;
			break;
		case 'L':
			params.schedule = IRIS_SCHEDULE_LINEAR;
			break;
		case 256:
			params.schedule = IRIS_SCHEDULE_POWER;
			break;
		case 257:
			params.power_alpha = atof(optarg);
			params.schedule    = IRIS_SCHEDULE_POWER;
			break;
		case 260:
			params.schedule = IRIS_SCHEDULE_SIGMOID;
			break;
		case 261:
			iris_vae_tiling = 1;
			break;
		case 262:
			upscale_mode = 1;
			break;
		case 263:
			depth_mode = 1;
			break;
		case 264:
			tileable        = 1;
			params.circular = 1;
			break;
		case 265:
			mask_path = optarg;
			break;
		case 266:
			variance = atof(optarg);
			break;
		case 258:
			no_license_info = 1;
			break;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	/* Backend banner (suppressed by --quiet) */
	if (output_level != OUTPUT_QUIET) {
#ifdef USE_METAL
		if (iris_metal_available()) {
			long   ncpu           = sysconf(_SC_NPROCESSORS_ONLN);
			char   cpu_brand[128] = "Apple Silicon";
			size_t len            = sizeof(cpu_brand);
			sysctlbyname("machdep.cpu.brand_string", cpu_brand, &len, NULL, 0);
			fprintf(stderr, "MPS: Metal GPU | %s | %ld cores\n", cpu_brand, ncpu);
		}
#elif defined(USE_VULKAN)
		if (iris_vulkan_available()) {
			long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
			fprintf(stderr, "Vulkan: %s | %ld CPU cores\n", iris_vulkan_device_name(), ncpu);
		}
		else {
			fprintf(stderr, "Vulkan: unavailable, using CPU fallback\n");
		}
#else
		fprintf(stderr, "Generic: Pure C backend (no acceleration)\n");
#endif
	}

	/* Validate required arguments */
	if (!model_dir) {
		fprintf(stderr, "Error: Model directory (-d) is required\n\n");
		print_usage(argv[0]);
		return 1;
	}

	if ((mask_path || variance >= 0.0f) && (upscale_mode || depth_mode))
		LOG_NORMAL("Note: --mask / --vary have no effect in --upscale / --depth modes\n");

	/* ============== Upscale mode (RealESRGAN_x4plus, no diffusion) ============== */
	if (upscale_mode) {
		if (num_inputs < 1) {
			fprintf(stderr, "Error: --upscale requires an input image (-i)\n");
			return 1;
		}
		if (!output_path) {
			fprintf(stderr, "Error: Output path (-o) is required\n");
			return 1;
		}

		char model_path[1024];
		snprintf(model_path, sizeof(model_path), "%s/RealESRGAN_x4plus.safetensors", model_dir);

		LOG_NORMAL("Loading RealESRGAN_x4plus...");
		if (output_level >= OUTPUT_NORMAL)
			fflush(stderr);
		timer_begin();
		iris_upscale_t *up = iris_upscale_load(model_path);
		if (!up) {
			fprintf(stderr, "\nError: Failed to load upscale model: %s\n", model_path);
			return 1;
		}
		iris_upscale_set_tileable(up, tileable);
		LOG_NORMAL(" done (%.1fs)\n", timer_end());

		iris_image *src = iris_image_load(input_paths[0]);
		if (!src) {
			fprintf(stderr, "Error: Failed to load input image: %s\n", input_paths[0]);
			iris_upscale_free(up);
			return 1;
		}
		LOG_NORMAL("Upscaling %dx%d -> %dx%d (4x)...", src->width, src->height, src->width * 4, src->height * 4);
		if (output_level >= OUTPUT_NORMAL)
			fflush(stderr);
		timer_begin();
		iris_image *result = iris_upscale_run(up, src);
		iris_image_free(src);
		iris_upscale_free(up);
		if (!result) {
			fprintf(stderr, "\nError: Upscale failed\n");
			return 1;
		}
		LOG_NORMAL(" done (%.1fs)\n", timer_end());

		if (iris_image_save(result, output_path) != 0) {
			fprintf(stderr, "Error: Failed to save image: %s\n", output_path);
			iris_image_free(result);
			return 1;
		}
		LOG_NORMAL("Saved %s %dx%d\n", output_path, result->width, result->height);
		iris_image_free(result);

#ifdef USE_METAL
		iris_metal_cleanup();
#endif
#ifdef USE_VULKAN
		iris_vulkan_cleanup();
#endif
		return 0;
	}

	/* ============== Depth mode (Depth Anything V2, no diffusion) ============== */
	if (depth_mode) {
		if (num_inputs < 1) {
			fprintf(stderr, "Error: --depth requires an input image (-i)\n");
			return 1;
		}
		if (!output_path) {
			fprintf(stderr, "Error: Output path (-o) is required\n");
			return 1;
		}

		char model_path[1024];
		snprintf(model_path, sizeof(model_path), "%s/da3-mono-large.safetensors", model_dir);

		LOG_NORMAL("Loading Depth Anything 3 (mono-large)...");
		if (output_level >= OUTPUT_NORMAL)
			fflush(stderr);
		timer_begin();
		iris_depth_t *dm = iris_depth_load(model_path);
		if (!dm) {
			fprintf(stderr, "\nError: Failed to load depth model: %s\n", model_path);
			return 1;
		}
		LOG_NORMAL(" done (%.1fs)\n", timer_end());
		iris_depth_set_tileable(dm, tileable);

		iris_image *src = iris_image_load(input_paths[0]);
		if (!src) {
			fprintf(stderr, "Error: Failed to load input image: %s\n", input_paths[0]);
			iris_depth_free(dm);
			return 1;
		}
		LOG_NORMAL("Estimating depth %dx%d...", src->width, src->height);
		if (output_level >= OUTPUT_NORMAL)
			fflush(stderr);
		timer_begin();
		iris_image *result = iris_depth_estimate(dm, src);
		iris_image_free(src);
		iris_depth_free(dm);
		if (!result) {
			fprintf(stderr, "\nError: Depth estimation failed\n");
			return 1;
		}
		LOG_NORMAL(" done (%.1fs)\n", timer_end());

		if (iris_image_save(result, output_path) != 0) {
			fprintf(stderr, "Error: Failed to save image: %s\n", output_path);
			iris_image_free(result);
			return 1;
		}
		LOG_NORMAL("Saved %s %dx%d\n", output_path, result->width, result->height);
		iris_image_free(result);

#ifdef USE_METAL
		iris_metal_cleanup();
#endif
#ifdef USE_VULKAN
		iris_vulkan_cleanup();
#endif
		return 0;
	}

	if (!prompt) {
		fprintf(stderr, "Error: Prompt (-p) is required\n\n");
		print_usage(argv[0]);
		return 1;
	}
	if (!output_path) {
		fprintf(stderr, "Error: Output path (-o) is required\n\n");
		print_usage(argv[0]);
		return 1;
	}

	/* Validate parameters */
	if (params.width < 64 || params.width > 4096) {
		fprintf(stderr, "Error: Width must be between 64 and 4096\n");
		return 1;
	}
	if (params.height < 64 || params.height > 4096) {
		fprintf(stderr, "Error: Height must be between 64 and 4096\n");
		return 1;
	}
	if (steps_set && (params.num_steps < 1 || params.num_steps > IRIS_MAX_STEPS)) {
		fprintf(stderr, "Error: Steps must be between 1 and %d\n", IRIS_MAX_STEPS);
		return 1;
	}

	/* Set seed */
	int64_t actual_seed;
	if (params.seed >= 0) {
		actual_seed = params.seed;
	}
	else {
		actual_seed = (int64_t)time(NULL);
	}
	iris_set_seed(actual_seed);
	LOG_NORMAL("Seed: %lld\n", (long long)actual_seed);

	/* Verbose header */
	LOG_VERBOSE("Iris Image Generator\n");
	LOG_VERBOSE("================================\n");
	LOG_VERBOSE("Model: %s\n", model_dir);
	if (prompt)
		LOG_VERBOSE("Prompt: %s\n", prompt);
	LOG_VERBOSE("Output: %s\n", output_path);
	LOG_VERBOSE("Size: %dx%d\n", params.width, params.height);
	LOG_VERBOSE("Steps: %d\n", params.num_steps);
	if (num_inputs > 0) {
		LOG_VERBOSE("Input images: %d\n", num_inputs);
		for (int i = 0; i < num_inputs; i++) {
			LOG_VERBOSE("  [%d] %s\n", i + 1, input_paths[i]);
		}
	}
	LOG_VERBOSE("\n");

	/* Load model (VAE only at startup, other components loaded on-demand) */
	LOG_NORMAL("Loading VAE...");
	if (output_level >= OUTPUT_NORMAL)
		fflush(stderr);
	timer_begin();

	iris_ctx *ctx = iris_load_dir(model_dir);
	if (!ctx) {
		fprintf(stderr, "\nError: Failed to load model: %s\n", iris_get_error());
		return 1;
	}

	/* Enable mmap mode if requested (reduces memory, slower inference) */
	if (use_mmap) {
		iris_set_mmap(ctx, 1);
		LOG_VERBOSE("  Using mmap mode for text encoder (lower memory)\n");
	}

	/* Override model type if --base was specified */
	if (force_base) {
		iris_set_base_mode(ctx);
	}

	/* Resolve auto-parameters now that we know the model type */
	if (!steps_set || params.num_steps <= 0) {
		params.num_steps = iris_is_distilled(ctx) ? 4 : 50;
	}
	if (params.guidance <= 0) {
		params.guidance = iris_is_distilled(ctx) ? 1.0f : 4.0f;
	}

	double load_time = timer_end();
	LOG_NORMAL(" done (%.1fs)\n", load_time);
	LOG_NORMAL("Model: %s\n", iris_model_info(ctx));

	/* Non-commercial license warning for 9B model */
	if (iris_is_non_commercial(ctx) && !no_license_info && output_level != OUTPUT_QUIET) {
		fprintf(stderr, "\nNOTE: This model is released under a NON COMMERCIAL LICENSE.\n"
		                "The output can only be used under the terms of the\n"
		                "FLUX non-commercial license:\n"
		                "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md\n"
		                "(use --no-license-info to suppress this message)\n\n");
	}

	/* Set up progress callbacks (for normal and verbose modes) */
	if (output_level >= OUTPUT_NORMAL) {
		cli_setup_progress();
	}

	/* Generate image */
	iris_image    *output = NULL;
	struct timeval total_start_tv;
	gettimeofday(&total_start_tv, NULL);

	if (num_inputs > 0) {
		/* ============== Image-to-image mode (single or multi-reference) ============== */
		LOG_NORMAL("Loading %d input image%s...", num_inputs, num_inputs > 1 ? "s" : "");
		if (output_level >= OUTPUT_NORMAL)
			fflush(stderr);
		timer_begin();

		iris_image *inputs[MAX_INPUT_IMAGES];
		for (int i = 0; i < num_inputs; i++) {
			inputs[i] = iris_image_load(input_paths[i]);
			if (!inputs[i]) {
				fprintf(stderr, "\nError: Failed to load input image: %s\n", input_paths[i]);
				for (int j = 0; j < i; j++)
					iris_image_free(inputs[j]);
				iris_free(ctx);
				return 1;
			}
		}

		LOG_NORMAL(" done (%.1fs)\n", timer_end());
		for (int i = 0; i < num_inputs; i++) {
			LOG_VERBOSE("  Input[%d]: %dx%d, %d channels\n", i + 1, inputs[i]->width, inputs[i]->height, inputs[i]->channels);
		}

		/* Use first input image dimensions if not explicitly set */
		if (!width_set)
			params.width = inputs[0]->width;
		if (!height_set)
			params.height = inputs[0]->height;

		if (mask_path) {
			/* Masked inpainting / object removal (single input). */
			if (num_inputs > 1)
				LOG_NORMAL("Note: --mask uses only the first input image\n");
			iris_image *mask_img = iris_image_load(mask_path);
			if (!mask_img) {
				fprintf(stderr, "\nError: Failed to load mask image: %s\n", mask_path);
				for (int i = 0; i < num_inputs; i++)
					iris_image_free(inputs[i]);
				iris_free(ctx);
				return 1;
			}
			if (variance >= 0.0f)
				LOG_NORMAL("Note: --vary is ignored when --mask is used\n");
			output = iris_inpaint(ctx, prompt, inputs[0], mask_img, &params);
			iris_image_free(mask_img);
		}
		else if (tileable) {
			/* Re-synthesize the whole image seamlessly (circular padding).
			 * --vary, if given, sets the strength; otherwise use the default. */
			if (num_inputs > 1)
				LOG_NORMAL("Note: --tile uses only the first input image\n");
			output = iris_make_tileable(ctx, prompt, inputs[0], variance, &params);
		}
		else if (variance >= 0.0f) {
			/* Traditional noise-and-denoise variation (single input). */
			if (num_inputs > 1)
				LOG_NORMAL("Note: --vary uses only the first input image\n");
			output = iris_img2img_strength(ctx, prompt, inputs[0], variance, &params);
		}
		else {
			/* Generate with multi-reference */
			output = iris_multiref(ctx, prompt, (const iris_image **)inputs, num_inputs, &params);
		}

		for (int i = 0; i < num_inputs; i++) {
			iris_image_free(inputs[i]);
		}
	}
	else {
		/* ============== Text-to-image mode ============== */
		if (mask_path || variance >= 0.0f)
			LOG_NORMAL("Note: --mask / --vary have no effect in text-to-image mode (no input image)\n");
		/* Note: iris_generate handles text encoding internally.
		 * We can't easily time it separately without modifying the library.
		 * The progress callbacks will show denoising progress. */
		output = iris_generate(ctx, prompt, &params);
	}

	/* Finish progress display */
	cli_finish_progress();

	if (!output) {
		fprintf(stderr, "Error: Generation failed: %s\n", iris_get_error());
		iris_free(ctx);
		return 1;
	}

	struct timeval total_end_tv;
	gettimeofday(&total_end_tv, NULL);
	double total_time = (total_end_tv.tv_sec - total_start_tv.tv_sec) + (total_end_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
	LOG_VERBOSE("Generated in %.1fs total\n", total_time);
	LOG_VERBOSE("  Output: %dx%d, %d channels\n", output->width, output->height, output->channels);

	/* Save output */
	LOG_NORMAL("Saving...");
	if (output_level >= OUTPUT_NORMAL)
		fflush(stderr);
	timer_begin();

	if (iris_image_save_with_seed(output, output_path, actual_seed) != 0) {
		fprintf(stderr, "\nError: Failed to save image: %s\n", output_path);
		iris_image_free(output);
		iris_free(ctx);
		return 1;
	}

	LOG_NORMAL(" %s %dx%d (%.1fs)\n", output_path, output->width, output->height, timer_end());

	/* Print total time (always, unless quiet) */
	struct timeval final_tv;
	gettimeofday(&final_tv, NULL);
	double total_time_final = (final_tv.tv_sec - total_start_tv.tv_sec) + (final_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
	LOG_NORMAL("Total generation time: %.1f seconds\n", load_time + total_time_final);

	/* Cleanup */
	iris_image_free(output);
	iris_free(ctx);

#ifdef USE_METAL
	iris_metal_cleanup();
#endif
#ifdef USE_VULKAN
	iris_vulkan_cleanup();
#endif

	return 0;
}
