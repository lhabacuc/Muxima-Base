#include "tensor.h"

#include <stdexcept>

Tensor::Tensor(const std::vector<size_t>& shape)
	: _shape(shape)
{
	size_t elements;

	if (_shape.empty())
		throw (std::invalid_argument("Tensor shape cannot be empty"));

	elements = 1;
	for (size_t i = 0; i < _shape.size(); i++)
	{
		if (_shape[i] == 0)
			throw (std::invalid_argument("Tensor dimension cannot be zero"));
		elements *= _shape[i];
	}
	_data.resize(elements);
}

size_t Tensor::index(const std::vector<size_t>& position) const
{
	size_t index;
	size_t stride;
	size_t i;

	if (position.size() != _shape.size())
		throw (std::out_of_range("Invalid number of indices"));

	index = 0;
	stride = 1;
	i = _shape.size();

	while (i > 0)
	{
		i--;
		if (position[i] >= _shape[i])
			throw (std::out_of_range("Tensor index out of range"));
		index += position[i] * stride;
		stride *= _shape[i];
	}
	return (index);
}

void Tensor::reshape(const std::vector<size_t>& new_shape)
{
	size_t elements;

	if (new_shape.empty())
		throw (std::invalid_argument("Tensor shape cannot be empty"));

	elements = 1;
	for (size_t i = 0; i < new_shape.size(); i++)
	{
		if (new_shape[i] == 0)
			throw (std::invalid_argument("Tensor dimension cannot be zero"));
		elements *= new_shape[i];
	}

	if (elements != _data.size())
		throw (std::invalid_argument("Invalid reshape size"));

	_shape = new_shape;
}

std::vector<float>& Tensor::data()
{
	return (_data);
}

const std::vector<float>& Tensor::data() const
{
	return (_data);
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
