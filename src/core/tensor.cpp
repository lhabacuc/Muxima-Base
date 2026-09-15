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

const std::vector<float>& Tensor::data() const
{
	return _data;
}

const std::vector<size_t>& Tensor::shape() const
{
	return _shape;
}

size_t Tensor::size() const
{
	return _data.size();
}
