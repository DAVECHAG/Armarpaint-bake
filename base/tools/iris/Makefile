# Iris - C Image Generation Engine
# Supported models: FLUX.2 Klein (4B/9B)
# Makefile

CC = gcc
# Generated headers (SPIR-V / Metal shaders) live in $(BUILD_DIR), so add it
# to the include path for every build.
BUILD_DIR = build
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -I$(BUILD_DIR)
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = iris.c iris_kernels.c iris_tokenizer.c iris_vae.c iris_transformer_flux.c iris_sample.c iris_image.c jpeg.c iris_safetensors.c iris_gguf.c iris_qwen3.c iris_qwen3_tokenizer.c iris_upscale.c iris_depth.c
OBJS = $(addprefix $(BUILD_DIR)/,$(SRCS:.c=.o))
MAIN = main.c
TARGET = $(BUILD_DIR)/iris
LIB = $(BUILD_DIR)/libiris.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address -I$(BUILD_DIR)

.PHONY: all clean debug lib install info help cpu metal vulkan
.NOTPARALLEL: metal vulkan

# Shader compiler for the Vulkan backend (emits SPIR-V as a C array)
GLSLC = glslc

# Default: show available targets
all: help

help:
	@echo "Iris - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make cpu      - Pure C, no dependencies (slow)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make metal    - Apple Silicon with Metal GPU (fastest)"
endif
else
	@echo "  make vulkan   - Cross-platform GPU via Vulkan compute"
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make metal && ./build/iris -d flux-klein-4b -p \"a cat\" -o cat.png"

# Create the build directory on demand (order-only prerequisite).
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# =============================================================================
# Backend: cpu (pure C, no BLAS)
# =============================================================================
cpu: CFLAGS = $(CFLAGS_BASE) -DCPU_BUILD
cpu: clean $(TARGET)
	@echo ""
	@echo "Built with CPU backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

# =============================================================================
# Backend: metal (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
METAL_CFLAGS = $(CFLAGS_BASE) -DUSE_METAL
METAL_OBJCFLAGS = $(METAL_CFLAGS) -fobjc-arc
METAL_LDFLAGS = $(LDFLAGS) -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation
METAL_OBJS = $(addprefix $(BUILD_DIR)/,$(SRCS:.c=.metal.o)) $(BUILD_DIR)/iris_metal.o $(BUILD_DIR)/main.metal.o

metal: clean metal-build
	@echo ""
	@echo "Built with METAL backend (Metal GPU acceleration)"

metal-build: $(METAL_OBJS)
	$(CC) $(METAL_CFLAGS) -o $(TARGET) $^ $(METAL_LDFLAGS)

$(BUILD_DIR)/%.metal.o: %.c iris.h iris_kernels.h | $(BUILD_DIR)
	$(CC) $(METAL_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
$(BUILD_DIR)/iris_shaders_source.h: iris_shaders.metal | $(BUILD_DIR)
	xxd -i $< > $@

$(BUILD_DIR)/iris_metal.o: iris_metal.m iris_metal.h $(BUILD_DIR)/iris_shaders_source.h | $(BUILD_DIR)
	$(CC) $(METAL_OBJCFLAGS) -c -o $@ $<

else
metal:
	@echo "Error: Metal backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
metal:
	@echo "Error: Metal backend requires macOS"
	@exit 1
endif

# =============================================================================
# Backend: vulkan (cross-platform GPU compute, GEMM offload)
# =============================================================================
VULKAN_CFLAGS = $(CFLAGS_BASE) -DUSE_VULKAN
VULKAN_LDFLAGS = $(LDFLAGS) -lvulkan

VULKAN_SPV = $(addprefix $(BUILD_DIR)/, \
             iris_vulkan_gemm_spv.h iris_vulkan_gemm_bf16_spv.h \
             iris_vulkan_gemm_q8_spv.h \
             iris_vulkan_conv2d_spv.h iris_vulkan_groupnorm_spv.h \
             iris_vulkan_swish_spv.h iris_vulkan_add_spv.h iris_vulkan_upsample_spv.h \
             iris_vulkan_leakyrelu_spv.h iris_vulkan_scaleadd_spv.h \
             iris_vulkan_qwen_rmsnorm_spv.h iris_vulkan_qwen_headrms_spv.h \
             iris_vulkan_qwen_rope_spv.h iris_vulkan_qwen_attn_spv.h \
             iris_vulkan_qwen_silumul_spv.h \
             iris_vulkan_res_rmsnorm_spv.h iris_vulkan_res_qkrmsnorm_spv.h \
             iris_vulkan_res_rope_spv.h iris_vulkan_res_attn_spv.h \
             iris_vulkan_res_split_spv.h iris_vulkan_res_gatedadd_spv.h \
             iris_vulkan_res_adaln_spv.h iris_vulkan_res_linear_f32_spv.h \
             iris_vulkan_flux_rowcopy_spv.h)

vulkan: CFLAGS = $(VULKAN_CFLAGS)
vulkan: clean $(VULKAN_SPV) vulkan-build
	@echo ""
	@echo "Built with VULKAN backend (GPU GEMM offload + resident VAE decode)"

vulkan-build: $(OBJS) $(BUILD_DIR)/iris_vulkan.o $(BUILD_DIR)/main.o
	$(CC) $(VULKAN_CFLAGS) -o $(TARGET) $^ $(VULKAN_LDFLAGS)

# Compile GLSL compute shaders to SPIR-V embedded as a C uint32 array.
# glslc -mfmt=c emits a brace-enclosed initializer list, included verbatim
# by iris_vulkan.c (no Vulkan SDK linkage needed beyond the loader).
$(BUILD_DIR)/iris_vulkan_gemm_spv.h: iris_vulkan_gemm.comp | $(BUILD_DIR)
	$(GLSLC) -fshader-stage=compute -O $< -mfmt=c -o $@

$(BUILD_DIR)/iris_vulkan_gemm_bf16_spv.h: iris_vulkan_gemm_bf16.comp | $(BUILD_DIR)
	$(GLSLC) -fshader-stage=compute -O $< -mfmt=c -o $@

# Resident VAE-decode compute shaders.
$(BUILD_DIR)/iris_vulkan_%_spv.h: iris_vulkan_%.comp | $(BUILD_DIR)
	$(GLSLC) -fshader-stage=compute -O $< -mfmt=c -o $@

$(BUILD_DIR)/iris_vulkan.o: iris_vulkan.c iris_vulkan.h $(VULKAN_SPV) | $(BUILD_DIR)
	$(CC) $(VULKAN_CFLAGS) -c -o $@ $<

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) $(BUILD_DIR)/main.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS) | $(BUILD_DIR)
	ar rcs $@ $^

$(BUILD_DIR)/%.o: %.c iris.h iris_kernels.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Utilities
# =============================================================================
install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 iris.h /usr/local/include/
	install -m 644 iris_kernels.h /usr/local/include/

clean:
	rm -rf $(BUILD_DIR)

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo "Build dir: $(BUILD_DIR)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  cpu     - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  metal   - Metal GPU (recommended)"
endif
else
	@echo "  vulkan  - Cross-platform GPU via Vulkan compute"
endif

# =============================================================================
# Dependencies
# =============================================================================
$(BUILD_DIR)/iris.o: iris.c iris.h iris_kernels.h iris_safetensors.h iris_qwen3.h
$(BUILD_DIR)/iris_kernels.o: iris_kernels.c iris_kernels.h
$(BUILD_DIR)/iris_tokenizer.o: iris_tokenizer.c iris.h
$(BUILD_DIR)/iris_vae.o: iris_vae.c iris.h iris_kernels.h
$(BUILD_DIR)/iris_transformer_flux.o: iris_transformer_flux.c iris.h iris_kernels.h iris_safetensors.h iris_gguf.h
$(BUILD_DIR)/iris_gguf.o: iris_gguf.c iris_gguf.h iris_safetensors.h
$(BUILD_DIR)/iris_sample.o: iris_sample.c iris.h iris_kernels.h
$(BUILD_DIR)/iris_image.o: iris_image.c iris.h
$(BUILD_DIR)/iris_safetensors.o: iris_safetensors.c iris_safetensors.h
$(BUILD_DIR)/iris_upscale.o: iris_upscale.c iris_upscale.h iris_safetensors.h iris.h
$(BUILD_DIR)/iris_depth.o: iris_depth.c iris_depth.h iris_safetensors.h iris.h
$(BUILD_DIR)/iris_qwen3.o: iris_qwen3.c iris_qwen3.h iris_safetensors.h
$(BUILD_DIR)/iris_qwen3_tokenizer.o: iris_qwen3_tokenizer.c iris_qwen3.h
$(BUILD_DIR)/main.o: main.c iris.h iris_kernels.h
