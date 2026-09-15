#include "transformer_block.h"

#include "../core/math.h"

#include <stdexcept>

TransformerBlock::TransformerBlock(
	size_t embedding_dim,
	size_t num_heads,
	size_t hidden_dim)
	: _embedding_dim(embedding_dim),
	_attention(embedding_dim, num_heads),
	_feed_forward(embedding_dim, hidden_dim),
	_norm1(embedding_dim, 1e-5f),
	_norm2(embedding_dim, 1e-5f)
{
}

Tensor	TransformerBlock::forward(const Tensor& input)
{
	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"TransformerBlock expects 2D tensor"));

	if (input.shape()[1] != _embedding_dim)
		throw (std::invalid_argument(
			"Invalid embedding dimension"));

	Tensor	normalized = _norm1.forward(input);
	Tensor	attention_output = _attention.forward(normalized);
	Tensor	residual = add(input, attention_output);

	normalized = _norm2.forward(residual);
	Tensor	ffn_output = _feed_forward.forward(normalized);

	return (add(residual, ffn_output));
}

MultiHeadAttention&	TransformerBlock::attention()
{
	return (_attention);
}

FeedForward&	TransformerBlock::feed_forward()
{
	return (_feed_forward);
}

LayerNorm&	TransformerBlock::norm1()
{
	return (_norm1);
}

LayerNorm&	TransformerBlock::norm2()
{
	return (_norm2);
}