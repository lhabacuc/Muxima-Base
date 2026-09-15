#include "self_attention.h"

#include "../core/math.h"

#include <cmath>
#include <stdexcept>

SelfAttention::SelfAttention(size_t embedding_dim)
	: _embedding_dim(embedding_dim),
	_head_dim(embedding_dim),
	_query(embedding_dim, embedding_dim),
	_key(embedding_dim, embedding_dim),
	_value(embedding_dim, embedding_dim),
	_output(embedding_dim, embedding_dim)
{
}

Tensor	SelfAttention::forward(const Tensor& input)
{
	Tensor	q(input.shape());
	Tensor	k(input.shape());
	Tensor	v(input.shape());
	Tensor	scores(input.shape());
	Tensor	weights(input.shape());
	Tensor	context(input.shape());

	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"SelfAttention expects 2D tensor"));

	if (input.shape()[1] != _embedding_dim)
		throw (std::invalid_argument(
			"Invalid embedding dimension"));

	q = _query.forward(input);
	k = _key.forward(input);
	v = _value.forward(input);

	scores = matmul(q, transpose(k));
	scores = scalar_multiply(
		scores,
		1.0f / std::sqrt(static_cast<float>(_head_dim)));

	weights = softmax(scores);
	context = matmul(weights, v);

	return (_output.forward(context));
}

Linear&	SelfAttention::query()
{
	return (_query);
}

Linear&	SelfAttention::key()
{
	return (_key);
}

Linear&	SelfAttention::value()
{
	return (_value);
}

Linear&	SelfAttention::output()
{
	return (_output);
}
