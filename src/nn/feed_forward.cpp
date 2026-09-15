#include "feed_forward.h"

#include "../core/math.h"

#include <stdexcept>

FeedForward::FeedForward(
	size_t embedding_dim,
	size_t hidden_dim)
	: _embedding_dim(embedding_dim),
	_hidden_dim(hidden_dim),
	_up(embedding_dim, hidden_dim),
	_down(hidden_dim, embedding_dim)
{
}

Tensor	FeedForward::forward(const Tensor& input)
{
	Tensor	hidden;
	Tensor	output;

	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"FeedForward expects 2D tensor"));

	if (input.shape()[1] != _embedding_dim)
		throw (std::invalid_argument(
			"Invalid embedding dimension"));

	hidden = _up.forward(input);
	hidden = gelu(hidden);
	output = _down.forward(hidden);

	return (output);
}

Linear&	FeedForward::up()
{
	return (_up);
}

Linear&	FeedForward::down()
{
	return (_down);
}