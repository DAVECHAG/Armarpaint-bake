/*
 * iris_gguf.c - GGUF file format reader implementation
 *
 * Parses the GGUF header/metadata/tensor-directory and fills a
 * safetensors_file_t so downstream code is format-agnostic. The mmap'd payload
 * is shared; for BF16/F16/F32 tensors the bytes match safetensors exactly.
 */

#include "iris_gguf.h"
#ifdef _WIN32
#include "iris_compat.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* GGML tensor types we support. BF16 checkpoints use only F32 + BF16; Q8_0
 * checkpoints additionally use the Q8_0 block-quantized type for the large
 * projection/embedding weights (norms stay F32). */
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_BF16 30

/* GGML Q8_0: 32 quantized elements per block, stored as an fp16 scale (2 bytes)
 * followed by 32 signed int8 quants -> 34 bytes per block. */
#define GGML_Q8_0_BLOCK 32
#define GGML_Q8_0_BYTES 34

/* GGUF metadata value types */
enum {
	GGUF_T_UINT8   = 0,
	GGUF_T_INT8    = 1,
	GGUF_T_UINT16  = 2,
	GGUF_T_INT16   = 3,
	GGUF_T_UINT32  = 4,
	GGUF_T_INT32   = 5,
	GGUF_T_FLOAT32 = 6,
	GGUF_T_BOOL    = 7,
	GGUF_T_STRING  = 8,
	GGUF_T_ARRAY   = 9,
	GGUF_T_UINT64  = 10,
	GGUF_T_INT64   = 11,
	GGUF_T_FLOAT64 = 12
};

/* Cursor over the mmap'd file, with end-of-buffer guarding. */
typedef struct {
	const uint8_t *p;
	const uint8_t *end;
	int            error;
} gguf_cursor_t;

static uint64_t cur_read(gguf_cursor_t *c, void *dst, size_t n) {
	if (c->error || (size_t)(c->end - c->p) < n) {
		c->error = 1;
		return 0;
	}
	if (dst)
		memcpy(dst, c->p, n);
	c->p += n;
	return 1;
}

static uint32_t cur_u32(gguf_cursor_t *c) {
	uint32_t v = 0;
	cur_read(c, &v, 4);
	return v;
}
static uint64_t cur_u64(gguf_cursor_t *c) {
	uint64_t v = 0;
	cur_read(c, &v, 8);
	return v;
}

/* GGUF string: uint64 length + raw bytes (not NUL-terminated). */
static uint64_t cur_str(gguf_cursor_t *c, char *out, size_t max) {
	uint64_t len = cur_u64(c);
	if (c->error || (size_t)(c->end - c->p) < len) {
		c->error = 1;
		return 0;
	}
	if (out) {
		size_t n = len < max - 1 ? (size_t)len : max - 1;
		memcpy(out, c->p, n);
		out[n] = '\0';
	}
	c->p += len;
	return len;
}

static size_t gguf_scalar_size(uint32_t t) {
	switch (t) {
	case GGUF_T_UINT8:
	case GGUF_T_INT8:
	case GGUF_T_BOOL:
		return 1;
	case GGUF_T_UINT16:
	case GGUF_T_INT16:
		return 2;
	case GGUF_T_UINT32:
	case GGUF_T_INT32:
	case GGUF_T_FLOAT32:
		return 4;
	case GGUF_T_UINT64:
	case GGUF_T_INT64:
	case GGUF_T_FLOAT64:
		return 8;
	default:
		return 0;
	}
}

/* Read one metadata value, capturing it only if it is the alignment key. */
static void gguf_read_value(gguf_cursor_t *c, uint32_t type, int want_align, uint32_t *align_out) {
	if (type == GGUF_T_STRING) {
		cur_str(c, NULL, 0);
	}
	else if (type == GGUF_T_ARRAY) {
		uint32_t et = cur_u32(c);
		uint64_t n  = cur_u64(c);
		for (uint64_t i = 0; i < n && !c->error; i++) {
			gguf_read_value(c, et, 0, NULL);
		}
	}
	else {
		size_t sz = gguf_scalar_size(type);
		if (sz == 0) {
			c->error = 1;
			return;
		}
		uint64_t v = 0;
		cur_read(c, &v, sz);
		if (want_align && type == GGUF_T_UINT32 && align_out)
			*align_out = (uint32_t)v;
	}
}

static safetensor_dtype_t gguf_map_dtype(uint32_t ggml_type, size_t *elem_size) {
	switch (ggml_type) {
	case GGML_TYPE_F32:
		*elem_size = 4;
		return DTYPE_F32;
	case GGML_TYPE_F16:
		*elem_size = 2;
		return DTYPE_F16;
	case GGML_TYPE_BF16:
		*elem_size = 2;
		return DTYPE_BF16;
	case GGML_TYPE_Q8_0:
		*elem_size = 0;
		return DTYPE_Q8_0; /* block type, see below */
	default:
		*elem_size = 0;
		return DTYPE_UNKNOWN;
	}
}

safetensors_file_t *gguf_open(const char *path) {
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		perror("gguf_open: open failed");
		return NULL;
	}

	struct stat st;
	if (fstat(fd, &st) < 0) {
		perror("gguf_open: fstat failed");
		close(fd);
		return NULL;
	}
	size_t file_size = (size_t)st.st_size;
	if (file_size < 24) {
		fprintf(stderr, "gguf_open: file too small\n");
		close(fd);
		return NULL;
	}

	void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (data == MAP_FAILED) {
		perror("gguf_open: mmap failed");
		return NULL;
	}

	gguf_cursor_t c = {(const uint8_t *)data, (const uint8_t *)data + file_size, 0};

	char magic[4];
	cur_read(&c, magic, 4);
	if (memcmp(magic, "GGUF", 4) != 0) {
		fprintf(stderr, "gguf_open: not a GGUF file\n");
		munmap(data, file_size);
		return NULL;
	}
	uint32_t version = cur_u32(&c);
	if (version != 2 && version != 3) {
		fprintf(stderr, "gguf_open: unsupported GGUF version %u\n", version);
		munmap(data, file_size);
		return NULL;
	}
	uint64_t n_tensors = cur_u64(&c);
	uint64_t n_kv      = cur_u64(&c);

	if (n_tensors > SAFETENSORS_MAX_TENSORS) {
		fprintf(stderr, "gguf_open: %llu tensors exceeds limit %d\n", (unsigned long long)n_tensors, SAFETENSORS_MAX_TENSORS);
		munmap(data, file_size);
		return NULL;
	}

	/* Metadata: we only care about general.alignment (default 32). */
	uint32_t alignment = 32;
	for (uint64_t i = 0; i < n_kv && !c.error; i++) {
		char key[128];
		cur_str(&c, key, sizeof(key));
		uint32_t type       = cur_u32(&c);
		int      want_align = (strcmp(key, "general.alignment") == 0);
		gguf_read_value(&c, type, want_align, &alignment);
	}
	if (alignment == 0)
		alignment = 32;

	safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
	if (!sf) {
		munmap(data, file_size);
		return NULL;
	}
	sf->path        = strdup(path);
	sf->data        = data;
	sf->file_size   = file_size;
	sf->header_json = NULL; /* not used for GGUF; safetensors_close free(NULL) is safe */

	/* Tensor directory. */
	int nt = 0;
	for (uint64_t i = 0; i < n_tensors && !c.error; i++) {
		safetensor_t *t = &sf->tensors[nt];
		cur_str(&c, t->name, sizeof(t->name));
		uint32_t ndim = cur_u32(&c);
		if (ndim > 8) {
			c.error = 1;
			break;
		}

		/* GGML stores dims fastest-first; reverse to PyTorch (slowest-first). */
		int64_t ne[8];
		for (uint32_t d = 0; d < ndim; d++)
			ne[d] = (int64_t)cur_u64(&c);
		t->ndim = (int)ndim;
		for (uint32_t d = 0; d < ndim; d++)
			t->shape[d] = ne[ndim - 1 - d];

		uint32_t ggml_type = cur_u32(&c);
		uint64_t offset    = cur_u64(&c);

		size_t elem_size = 0;
		t->dtype         = gguf_map_dtype(ggml_type, &elem_size);
		if (t->dtype == DTYPE_UNKNOWN) {
			fprintf(stderr,
			        "gguf_open: %s: unsupported ggml type %u "
			        "(only F32/F16/BF16/Q8_0 supported)\n",
			        t->name, ggml_type);
			c.error = 1;
			break;
		}

		int64_t numel = 1;
		for (uint32_t d = 0; d < ndim; d++)
			numel *= ne[d];
		t->data_offset = (size_t)offset; /* relative to data section */
		if (t->dtype == DTYPE_Q8_0) {
			/* Block-quantized: 34 bytes per 32 elements. ne[0] (the fastest GGML
			 * dim) is a multiple of 32, so blocks tile the tensor exactly. */
			t->data_size = (size_t)(numel / GGML_Q8_0_BLOCK) * GGML_Q8_0_BYTES;
		}
		else {
			t->data_size = (size_t)numel * elem_size;
		}
		nt++;
	}

	if (c.error) {
		fprintf(stderr, "gguf_open: %s: malformed header\n", path);
		safetensors_close(sf);
		return NULL;
	}

	/* Data section begins at the next alignment boundary after the directory. */
	size_t header_end = (size_t)(c.p - (const uint8_t *)data);
	size_t data_start = (header_end + alignment - 1) / alignment * alignment;
	if (data_start < 8)
		data_start = 8;

	/* Reuse safetensors_data()'s arithmetic: it returns
	 * data + 8 + header_size + data_offset. Set header_size so that
	 * 8 + header_size == data_start, leaving data_offset as the GGUF-relative
	 * tensor offset. */
	sf->header_size = data_start - 8;
	sf->num_tensors = nt;

	/* Validate every tensor lies within the file. */
	for (int i = 0; i < nt; i++) {
		safetensor_t *t = &sf->tensors[i];
		if (data_start + t->data_offset + t->data_size > file_size) {
			fprintf(stderr,
			        "gguf_open: %s: tensor '%s' extends past end of file "
			        "(truncated download?)\n",
			        path, t->name);
			safetensors_close(sf);
			return NULL;
		}
	}

	return sf;
}
