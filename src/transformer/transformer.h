#ifndef TRANSFORMER_H
# define TRANSFORMER_H

# include "../core/tensor.h"
# include "../nn/embedding.h"
# include "../nn/positional_encoding.h"
# include "../nn/layer_norm.h"
# include "transformer_block.h"

# include <vector>

class Transformer
{
	private:
		size_t	_embedding_dim;

		Embedding			_embedding;
		PositionalEncoding	_positional;
		LayerNorm			_final_norm;
		std::vector<TransformerBlock>	_blocks;

		static std::vector<TransformerBlock>	create_blocks(
			size_t num_layers,
			size_t embedding_dim,
			size_t num_heads,
			size_t hidden_dim);

	public:
		Transformer(
			size_t vocab_size,
			size_t max_length,
			size_t embedding_dim,
			size_t num_layers,
			size_t num_heads,
			size_t hidden_dim);

		Tensor	forward(const Tensor& tokens);

		Embedding&			embedding();
		PositionalEncoding&	positional();
		LayerNorm&			final_norm();
		std::vector<TransformerBlock>&	blocks();
};

#endif
