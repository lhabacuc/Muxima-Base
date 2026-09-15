#ifndef TRANSFORMER_BLOCK_H
# define TRANSFORMER_BLOCK_H

# include "../core/tensor.h"
# include "../nn/feed_forward.h"
# include "../nn/layer_norm.h"
# include "multi_head_attention.h"

class TransformerBlock
{
	private:
		size_t				_embedding_dim;
		MultiHeadAttention	_attention;
		FeedForward			_feed_forward;
		LayerNorm			_norm1;
		LayerNorm			_norm2;

	public:
		TransformerBlock(
			size_t embedding_dim,
			size_t num_heads,
			size_t hidden_dim);

		Tensor	forward(const Tensor& input);

		MultiHeadAttention&	attention();
		FeedForward&	feed_forward();
		LayerNorm&			norm1();
		LayerNorm&			norm2();
};

#endif