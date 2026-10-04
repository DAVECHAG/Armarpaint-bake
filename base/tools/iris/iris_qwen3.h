/*
 * Qwen3 Text Encoder for Iris
 *
 * Implements the Qwen3 text encoder that produces embeddings for image
 * generation. Supports multiple model sizes (architecture read from config.json).
 */

#ifndef IRIS_QWEN3_H
#define IRIS_QWEN3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Architecture Constants
 * ======================================================================== */

/* Fixed constants (same across model sizes) */
#define QWEN3_VOCAB_SIZE   151936
#define QWEN3_MAX_SEQ_LEN  512
#define QWEN3_RMS_NORM_EPS 1e-6f
#define QWEN3_ROPE_THETA   1000000.0f

/* Output layers to extract (0-indexed)
 * Python uses hidden_states[9,18,27] which are outputs AFTER layers 8,17,26
 * since hidden_states[0] is embedding and hidden_states[i] is output after layer i-1 */
#define QWEN3_OUTPUT_LAYER_1 8
#define QWEN3_OUTPUT_LAYER_2 17
#define QWEN3_OUTPUT_LAYER_3 26

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct qwen3_model     qwen3_model_t;
typedef struct qwen3_tokenizer qwen3_tokenizer_t;

/* ========================================================================
 * Tokenizer API
 * ======================================================================== */

/*
 * Load tokenizer from HuggingFace tokenizer.json file.
 */
qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path);

/*
 * Free tokenizer resources.
 */
void qwen3_tokenizer_free(qwen3_tokenizer_t *tok);

/*
 * Tokenize text with chat template.
 * Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt, int *num_tokens, int max_len);

/*
 * Pad tokens to max_len with PAD token.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask);

/*
 * Get token string by ID.
 */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id);

/*
 * Get token ID by string.
 */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token);

/* ========================================================================
 * Model API
 * ======================================================================== */

/*
 * Load Qwen3 model weights from HuggingFace model directory.
 * Directory should contain model-00001-of-00002.safetensors, etc.
 */
qwen3_model_t *qwen3_model_load(const char *model_dir);

/*
 * Free model resources.
 */
void qwen3_model_free(qwen3_model_t *model);

/*
 * Run forward pass to generate text embeddings.
 *
 * input_ids: Token IDs [seq_len]
 * attention_mask: Attention mask [seq_len] (1 for real tokens, 0 for padding)
 * seq_len: Length of input sequences
 *
 * Returns: Embedding array [seq_len, 7680] (caller must free)
 * Extracts hidden states from layers 9, 18, 27 and concatenates them.
 */
float *qwen3_forward(qwen3_model_t *model, const int *input_ids, const int *attention_mask, int seq_len);

/* ========================================================================
 * Combined Text Encoder API
 * ======================================================================== */

typedef struct qwen3_encoder {
	qwen3_tokenizer_t *tokenizer;
	qwen3_model_t     *model;
} qwen3_encoder_t;

/*
 * Load complete text encoder (tokenizer + model).
 * model_dir should contain both tokenizer/ and text_encoder/ subdirectories.
 * use_mmap: if true, use memory-mapped bf16 weights (saves ~8GB, slower inference)
 */
qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap);

/*
 * Free encoder resources.
 */
void qwen3_encoder_free(qwen3_encoder_t *enc);

/*
 * Encode text prompt to embeddings.
 * Returns: Embedding array [512, text_dim] (caller must free)
 * text_dim is 7680 (3 * hidden_size for the 4B model).
 */
float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt);

/*
 * Encode text prompt to embeddings, returning actual token count.
 * Same as qwen3_encode_text but also returns the number of real
 * (non-padding) tokens in *out_num_tokens.
 * Returns: Embedding array [512, text_dim] (caller must free)
 */
float *qwen3_encode_text_ex(qwen3_encoder_t *enc, const char *prompt, int *out_num_tokens);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_QWEN3_H */
