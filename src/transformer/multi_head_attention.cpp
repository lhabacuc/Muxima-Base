#include "multi_head_attention.h"

#include <cmath>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(size_t embedding_dim,
	size_t num_heads)
	: _query(embedding_dim, embedding_dim),
	_key(embedding_dim, embedding_dim),
	_value(embedding_dim, embedding_dim),
	_output(embedding_dim, embedding_dim),
	_num_heads(num_heads),
	_head_dim(0)
{
	if (embedding_dim == 0)
		throw std::invalid_argument("embedding_dim must be greater than zero");
	if (num_heads == 0)
		throw std::invalid_argument("num_heads must be greater than zero");
	if (embedding_dim % num_heads != 0)
		throw std::invalid_argument(
			"embedding_dim must be divisible by num_heads");

	_head_dim = embedding_dim / num_heads;
}

Tensor	MultiHeadAttention::create_head(const Tensor& tensor,
	size_t head) const
{
	const std::vector<size_t>& shape = tensor.shape();
	std::vector<size_t> result_shape;

	if (shape.size() != 2)
		throw std::invalid_argument("attention input must be 2D");

	result_shape.push_back(shape[0]);
	result_shape.push_back(_head_dim);

	Tensor result(result_shape);

	size_t start;
	size_t row;
	size_t column;

	start = head * _head_dim;
	row = 0;
	while (row < shape[0])
	{
		column = 0;
		while (column < _head_dim)
		{
			result.at(std::vector<size_t>{
				row, column
			}) = tensor.at(std::vector<size_t>{
				row, start + column
			});
			column++;
		}
		row++;
	}
	return result;
}

Tensor	MultiHeadAttention::create_causal_mask(
	size_t sequence_length) const
{
	std::vector<size_t> shape;

	shape.push_back(sequence_length);
	shape.push_back(sequence_length);

	Tensor mask(shape);

	size_t row;
	size_t column;

	row = 0;
	while (row < sequence_length)
	{
		column = 0;
		while (column < sequence_length)
		{
			if (column <= row)
				mask.at(std::vector<size_t>{row, column}) = 0.0f;
			else
				mask.at(std::vector<size_t>{row, column}) = -1e9f;
			column++;
		}
		row++;
	}
	return mask;
}

Tensor	MultiHeadAttention::apply_mask(const Tensor& scores,
	const Tensor& mask) const
{
	return add(scores, mask);
}

Tensor	MultiHeadAttention::forward(const Tensor& input)
{
	const std::vector<size_t>& shape = input.shape();

	if (shape.size() != 2)
		throw std::invalid_argument("attention input must be 2D");

	Tensor query = _query.forward(input);
	Tensor key = _key.forward(input);
	Tensor value = _value.forward(input);
	Tensor mask = create_causal_mask(shape[0]);

	std::vector<size_t> output_shape;
	output_shape.push_back(shape[0]);
	output_shape.push_back(_num_heads * _head_dim);

	Tensor combined(output_shape);

	size_t head;
	size_t row;
	size_t column;

	head = 0;
	while (head < _num_heads)
	{
		Tensor q_head = create_head(query, head);
		Tensor k_head = create_head(key, head);
		Tensor v_head = create_head(value, head);

		Tensor k_transposed = transpose(k_head);
		Tensor scores = matmul(q_head, k_transposed);

		scores = scalar_multiply(
			scores,
			1.0f / std::sqrt(static_cast<float>(_head_dim)));

		scores = add(scores, mask);

		Tensor attention = softmax(scores);
		Tensor context = matmul(attention, v_head);

		row = 0;
		while (row < shape[0])
		{
			column = 0;
			while (column < _head_dim)
			{
				size_t output_column;

				output_column = head * _head_dim + column;

				combined.at(std::vector<size_t>{
					row, output_column
				}) = context.at(std::vector<size_t>{
					row, column
				});
				column++;
			}
			row++;
		}
		head++;
	}

	return _output.forward(combined);
}

Linear&	MultiHeadAttention::query()
{
	return _query;
}

Linear&	MultiHeadAttention::key()
{
	return _key;
}

Linear&	MultiHeadAttention::value()
{
	return _value;
}

Linear&	MultiHeadAttention::output()
{
	return _output;
}

size_t	MultiHeadAttention::num_heads() const
{
	return _num_heads;
}

size_t	MultiHeadAttention::head_dim() const
{
	return _head_dim;
}
