#include "math.h"

#include <stdexcept>

Tensor	add(const Tensor& a, const Tensor& b)
{
	Tensor	result(a.shape());
	size_t	i;

	if (a.shape() != b.shape())
		throw (std::invalid_argument("Tensor shapes must match"));

	i = 0;
	while (i < a.size())
	{
		result.data()[i] = a.data()[i] + b.data()[i];
		i++;
	}
	return (result);
}

Tensor	subtract(const Tensor& a, const Tensor& b)
{
	Tensor	result(a.shape());
	size_t	i;

	if (a.shape() != b.shape())
		throw (std::invalid_argument("Tensor shapes must match"));

	i = 0;
	while (i < a.size())
	{
		result.data()[i] = a.data()[i] - b.data()[i];
		i++;
	}
	return (result);
}

Tensor	multiply(const Tensor& a, const Tensor& b)
{
	Tensor	result(a.shape());
	size_t	i;

	if (a.shape() != b.shape())
		throw (std::invalid_argument("Tensor shapes must match"));

	i = 0;
	while (i < a.size())
	{
		result.data()[i] = a.data()[i] * b.data()[i];
		i++;
	}
	return (result);
}

Tensor	scalar_multiply(const Tensor& tensor, float scalar)
{
	Tensor	result(tensor.shape());
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		result.data()[i] = tensor.data()[i] * scalar;
		i++;
	}
	return (result);
}

static Tensor	create_matmul_result(const Tensor& a, const Tensor& b)
{
	std::vector<size_t> shape;

	if (a.shape().size() != 2 || b.shape().size() != 2)
		throw (std::invalid_argument("matmul requires 2D tensors"));
	if (a.shape()[1] != b.shape()[0])
		throw (std::invalid_argument("Invalid matmul shapes"));

	shape.push_back(a.shape()[0]);
	shape.push_back(b.shape()[1]);
	return (Tensor(shape));
}

static void	fill_matmul(Tensor& result, const Tensor& a, const Tensor& b)
{
	size_t	rows;
	size_t	inner;
	size_t	columns;
	size_t	i;
	size_t	j;
	size_t	k;
	float	sum;

	rows = a.shape()[0];
	inner = a.shape()[1];
	columns = b.shape()[1];

	i = 0;
	while (i < rows)
	{
		j = 0;
		while (j < columns)
		{
			sum = 0.0f;
			k = 0;
			while (k < inner)
			{
				sum += a.data()[i * inner + k]
					* b.data()[k * columns + j];
				k++;
			}
			result.data()[i * columns + j] = sum;
			j++;
		}
		i++;
	}
}

Tensor	matmul(const Tensor& a, const Tensor& b)
{
	Tensor	result(create_matmul_result(a, b));

	fill_matmul(result, a, b);
	return (result);
}

static Tensor	create_transpose_result(const Tensor& tensor)
{
	std::vector<size_t> shape;

	if (tensor.shape().size() != 2)
		throw (std::invalid_argument("transpose requires 2D tensor"));

	shape.push_back(tensor.shape()[1]);
	shape.push_back(tensor.shape()[0]);
	return (Tensor(shape));
}

Tensor	transpose(const Tensor& tensor)
{
	Tensor	result(create_transpose_result(tensor));
	size_t	rows;
	size_t	columns;
	size_t	i;
	size_t	j;

	rows = tensor.shape()[0];
	columns = tensor.shape()[1];

	i = 0;
	while (i < rows)
	{
		j = 0;
		while (j < columns)
		{
			result.data()[j * rows + i]
				= tensor.data()[i * columns + j];
			j++;
		}
		i++;
	}
	return (result);
}
