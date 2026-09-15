#include "math.h"

#include <matrix-hpp/matrix.hpp>
#include <cmath>
#include <stdexcept>

static bool	can_broadcast_bias(const Tensor& a, const Tensor& b)
{
	if (a.shape().size() != 2 || b.shape().size() != 2)
		return (false);
	if (b.shape()[0] != 1)
		return (false);
	if (a.shape()[1] != b.shape()[1])
		return (false);
	return (true);
}

Tensor	relu(const Tensor& tensor)
{
	Tensor	result(tensor.shape());
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		if (tensor.data()[i] > 0.0f)
			result.data()[i] = tensor.data()[i];
		else
			result.data()[i] = 0.0f;
		i++;
	}
	return (result);
}

Tensor	gelu(const Tensor& tensor)
{
	Tensor	result(tensor.shape());
	float	x;
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		x = tensor.data()[i];
		result.data()[i] = 0.5f * x
			* (1.0f + ::erf(x / std::sqrt(2.0f)));
		i++;
	}
	return (result);
}

Tensor	add(const Tensor& a, const Tensor& b)
{
	Tensor	result(a.shape());
	size_t	i;
	size_t	j;
	size_t	columns;

	if (a.shape() == b.shape())
	{
		i = 0;
		while (i < a.size())
		{
			result.data()[i] = a.data()[i] + b.data()[i];
			i++;
		}
		return (result);
	}

	if (!can_broadcast_bias(a, b))
		throw (std::invalid_argument("Incompatible tensor shapes"));

	columns = a.shape()[1];
	i = 0;
	while (i < a.shape()[0])
	{
		j = 0;
		while (j < columns)
		{
			result.data()[i * columns + j]
				= a.data()[i * columns + j] + b.data()[j];
			j++;
		}
		i++;
	}
	return (result);
}

static Tensor	create_softmax_result(const Tensor& tensor)
{
	return (Tensor(tensor.shape()));
}

Tensor	softmax(const Tensor& tensor)
{
	Tensor	result(create_softmax_result(tensor));
	size_t	rows;
	size_t	columns;
	size_t	i;
	size_t	j;
	float	max_value;
	float	total;

	if (tensor.shape().size() != 2)
		throw (std::invalid_argument(
			"softmax expects 2D tensor"));

	rows = tensor.shape()[0];
	columns = tensor.shape()[1];

	i = 0;
	while (i < rows)
	{
		max_value = tensor.data()[i * columns];
		j = 1;
		while (j < columns)
		{
			if (tensor.data()[i * columns + j] > max_value)
				max_value = tensor.data()[i * columns + j];
			j++;
		}

		total = 0.0f;
		j = 0;
		while (j < columns)
		{
			result.data()[i * columns + j]
				= std::exp(
					tensor.data()[i * columns + j] - max_value);
			total += result.data()[i * columns + j];
			j++;
		}

		j = 0;
		while (j < columns)
		{
			result.data()[i * columns + j] /= total;
			j++;
		}
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

Tensor	matmul(const Tensor& a, const Tensor& b)
{
	// Using matrix-hpp (https://github.com/gc1905/matrix-hpp) for optimized matmul
	if (a.shape().size() != 2 || b.shape().size() != 2)
		throw (std::invalid_argument("matmul requires 2D tensors"));
	if (a.shape()[1] != b.shape()[0])
		throw (std::invalid_argument("Invalid matmul shapes"));

	size_t rows = a.shape()[0];
	size_t inner = a.shape()[1];
	size_t cols = b.shape()[1];

	Mtx::Matrix<float> matA(rows, inner);
	Mtx::Matrix<float> matB(inner, cols);

	size_t i = 0;
	while (i < rows)
	{
		size_t j = 0;
		while (j < inner)
		{
			matA(i, j) = a.data()[i * inner + j];
			j++;
		}
		i++;
	}
	i = 0;
	while (i < inner)
	{
		size_t j = 0;
		while (j < cols)
		{
			matB(i, j) = b.data()[i * cols + j];
			j++;
		}
		i++;
	}

	Mtx::Matrix<float> matC = matA * matB;

	Tensor result(create_matmul_result(a, b));
	i = 0;
	while (i < rows)
	{
		size_t j = 0;
		while (j < cols)
		{
			result.data()[i * cols + j] = matC(i, j);
			j++;
		}
		i++;
	}
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
	// Using matrix-hpp transpose (https://github.com/gc1905/matrix-hpp)
	if (tensor.shape().size() != 2)
		throw (std::invalid_argument("transpose requires 2D tensor"));

	size_t rows = tensor.shape()[0];
	size_t cols = tensor.shape()[1];

	Mtx::Matrix<float> mat(rows, cols);
	size_t i = 0;
	while (i < rows)
	{
		size_t j = 0;
		while (j < cols)
		{
			mat(i, j) = tensor.data()[i * cols + j];
			j++;
		}
		i++;
	}

	Mtx::Matrix<float> matT = Mtx::transpose(mat);

	Tensor result(create_transpose_result(tensor));
	i = 0;
	while (i < cols)
	{
		size_t j = 0;
		while (j < rows)
		{
			result.data()[i * rows + j] = matT(i, j);
			j++;
		}
		i++;
	}
	return (result);
}

float	sum(const Tensor& tensor)
{
	float	result;
	size_t	i;

	result = 0.0f;
	i = 0;
	while (i < tensor.size())
	{
		result += tensor.data()[i];
		i++;
	}
	return (result);
}

float	mean(const Tensor& tensor)
{
	if (tensor.size() == 0)
		throw (std::invalid_argument("Cannot calculate mean of empty tensor"));
	return (sum(tensor) / static_cast<float>(tensor.size()));
}

Tensor	exp(const Tensor& tensor)
{
	Tensor	result(tensor.shape());
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		result.data()[i] = std::exp(tensor.data()[i]);
		i++;
	}
	return (result);
}

Tensor	log(const Tensor& tensor)
{
	Tensor	result(tensor.shape());
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		result.data()[i] = std::log(tensor.data()[i]);
		i++;
	}
	return (result);
}

Tensor	sqrt(const Tensor& tensor)
{
	Tensor	result(tensor.shape());
	size_t	i;

	i = 0;
	while (i < tensor.size())
	{
		result.data()[i] = std::sqrt(tensor.data()[i]);
		i++;
	}
	return (result);
}

