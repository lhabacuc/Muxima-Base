#include "multi_head_attention.h"

#include "../core/math.h"

#include <cmath>
#include <stdexcept>
#include <vector>

static size_t	calculate_head_dim(
	size_t embedding_dim,
	size_t num_heads)
{
	if (num_heads == 0)
		throw (std::invalid_argument(
			"Number of heads cannot be zero"));

	if (embedding_dim % num_heads != 0)
		throw (std::invalid_argument(
			"Embedding dimension must be divisible by heads"));

	return (embedding_dim / num_heads);
}

static Tensor	extract_head(
	const Tensor& tensor,
	size_t head,
	size_t head_dim)
{
	std::vector<size_t> shape;
	size_t	sequence_length;
	size_t	i;
	size_t	j;

	sequence_length = tensor.shape()[0];

	shape.push_back(sequence_length);
	shape.push_back(head_dim);

	{
		Tensor	result(shape);

		i = 0;
		while (i < sequence_length)
		{
			j = 0;
			while (j < head_dim)
			{
				result.data()[i * head_dim + j]
					= tensor.data()[
						i * tensor.shape()[1]
						+ head * head_dim + j];
				j++;
			}
			i++;
		}
		return (result);
	}
}

MultiHeadAttention::MultiHeadAttention(
	size_t embedding_dim,
	size_t num_heads)
	: _embedding_dim(embedding_dim),
	_num_heads(num_heads),
	_head_dim(calculate_head_dim(
		embedding_dim, num_heads)),
	_query(embedding_dim, embedding_dim),
	_key(embedding_dim, embedding_dim),
	_value(embedding_dim, embedding_dim),
	_output(embedding_dim, embedding_dim)
{
}

Tensor	MultiHeadAttention::attention(
	const Tensor& input,
	size_t head)
{
	Tensor	q = extract_head(
		_query.forward(input),
		head,
		_head_dim);
	Tensor	k = extract_head(
		_key.forward(input),
		head,
		_head_dim);
	Tensor	v = extract_head(
		_value.forward(input),
		head,
		_head_dim);
	Tensor	scores = matmul(q, transpose(k));

	scores = scalar_multiply(
		scores,
		1.0f / std::sqrt(
			static_cast<float>(_head_dim)));

	Tensor	weights = softmax(scores);

	return (matmul(weights, v));
}

Tensor	MultiHeadAttention::concatenate_heads(
	const std::vector<Tensor>& heads)
{
	size_t	sequence_length;
	size_t	head_dim;
	size_t	num_heads;
	size_t	i;
	size_t	j;
	size_t	k;
	std::vector<size_t> shape;

	num_heads = heads.size();
	sequence_length = heads[0].shape()[0];
	head_dim = heads[0].shape()[1];

	shape.push_back(sequence_length);
	shape.push_back(head_dim * num_heads);

	{
		Tensor	result(shape);

		i = 0;
		while (i < sequence_length)
		{
			j = 0;
			while (j < num_heads)
			{
				k = 0;
				while (k < head_dim)
				{
					result.data()[
						i * head_dim * num_heads
						+ j * head_dim + k]
						= heads[j].data()[
							i * head_dim + k];
					k++;
				}
				j++;
			}
			i++;
		}
		return (result);
	}
}

Tensor	MultiHeadAttention::forward(
	const Tensor& input)
{
	std::vector<Tensor> heads;
	size_t	head;

	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"MultiHeadAttention expects 2D tensor"));

	if (input.shape()[1] != _embedding_dim)
		throw (std::invalid_argument(
			"Invalid embedding dimension"));

	head = 0;
	while (head < _num_heads)
	{
		heads.push_back(attention(input, head));
		head++;
	}

	Tensor	projected = concatenate_heads(heads);
	Tensor	result = _output.forward(projected);

	return (result);
}

Linear&	MultiHeadAttention::query()
{
	return (_query);
}

Linear&	MultiHeadAttention::key()
{
	return (_key);
}

Linear&	MultiHeadAttention::value()
{
	return (_value);
}

Linear&	MultiHeadAttention::output()
{
	return (_output);
}
