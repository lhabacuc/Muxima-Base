#include "initialization.h"

#include <cmath>
#include <cstdlib>
#include <vector>

static Tensor	create_tensor(size_t input_size, size_t output_size)
{
	std::vector<size_t> shape;

	shape.push_back(input_size);
	shape.push_back(output_size);
	return (Tensor(shape));
}

Tensor	xavier_uniform(size_t input_size, size_t output_size)
{
	Tensor	result(create_tensor(input_size, output_size));
	float	limit;
	float	random_value;
	size_t	i;

	limit = std::sqrt(
		6.0f / static_cast<float>(input_size + output_size));

	i = 0;
	while (i < result.size())
	{
		random_value = static_cast<float>(std::rand())
			/ static_cast<float>(RAND_MAX);
		random_value = random_value * 2.0f - 1.0f;
		result.data()[i] = random_value * limit;
		i++;
	}
	return (result);
}
