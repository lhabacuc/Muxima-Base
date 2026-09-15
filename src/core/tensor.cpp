#include "tensor.h"

Tensor::Tensor(const std::vector<size_t>& shape)
	: _shape(shape)
{
	size_t elements;

	elements = 1;
	for (size_t i = 0; i < _shape.size(); i++)
		elements *= _shape[i];
	_data.resize(elements);
}

size_t Tensor::index(size_t row, size_t column) const
{
	return (row * _shape[1] + column);
}

float& Tensor::operator()(size_t row, size_t column)
{
	return (_data[index(row, column)]);
}

const float& Tensor::operator()(size_t row, size_t column) const
{
	return (_data[index(row, column)]);
}

const std::vector<float>& Tensor::data() const
{
	return (_data);
}

const std::vector<size_t>& Tensor::shape() const
{
	return (_shape);
}

size_t Tensor::size() const
{
	return (_data.size());
}
