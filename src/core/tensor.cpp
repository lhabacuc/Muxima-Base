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

size_t Tensor::index(const std::vector<size_t>& position) const
{
	size_t index;
	size_t stride;
	size_t i;

	index = 0;
	stride = 1;
	i = _shape.size();

	while (i > 0)
	{
		i--;
		index += position[i] * stride;
		stride *= _shape[i];
	}
	return (index);
}

float& Tensor::at(const std::vector<size_t>& position)
{
	return (_data[index(position)]);
}

const float& Tensor::at(const std::vector<size_t>& position) const
{
	return (_data[index(position)]);
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
