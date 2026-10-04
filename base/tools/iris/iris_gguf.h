/*
 * iris_gguf.h - GGUF file format reader
 *
 * Reads GGUF v2/v3 container files into the same safetensors_file_t structure
 * used by the safetensors reader, so the rest of the codebase (find / data /
 * get_f32 / get_bf16) works unchanged. Only the container differs: BF16/F16/F32
 * tensor payloads are byte-identical to safetensors, so no dequantization is
 * needed for a BF16 GGUF.
 *
 * GGUF layout:
 *   - magic "GGUF" (4 bytes) + uint32 version
 *   - uint64 tensor_count, uint64 metadata_kv_count
 *   - metadata key-value pairs (typed)
 *   - tensor directory: name, n_dims, dims[], ggml_type, offset
 *   - padding to general.alignment
 *   - tensor data (offsets are relative to this section)
 */

#ifndef IRIS_GGUF_H
#define IRIS_GGUF_H

#include "iris_safetensors.h"

/* Open a GGUF file (memory-mapped). Returns a safetensors_file_t populated with
 * the GGUF's native tensor names, or NULL on failure. Free with
 * safetensors_close() (same struct, same cleanup). */
safetensors_file_t *gguf_open(const char *path);

#endif /* IRIS_GGUF_H */
