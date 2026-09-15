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
