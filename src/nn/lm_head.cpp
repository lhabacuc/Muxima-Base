#include "lm_head.h"

#include <stdexcept>

LMHead::LMHead(
	size_t embedding_dim,
	size_t vocab_size)
	: _embedding_dim(embedding_dim),
	_vocab_size(vocab_size),
	_projection(embedding_dim, vocab_size)
{
}

Tensor	LMHead::forward(const Tensor& input)
{
	(void)_vocab_size;
	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"LMHead expects 2D tensor"));

	if (input.shape()[1] != _embedding_dim)
		throw (std::invalid_argument(
			"Invalid embedding dimension"));

	return (_projection.forward(input));
}

Linear&	LMHead::projection()
{
	return (_projection);
}