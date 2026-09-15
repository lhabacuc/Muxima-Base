#include "cross_entropy.h"

#include "../core/math.h"

#include <cmath>
#include <stdexcept>

float	CrossEntropy::forward(
	const Tensor& logits,
	const Tensor& targets)
{
	size_t	rows;
	size_t	columns;
	size_t	i;
	size_t	j;
	size_t	target;
	float	max_value;
	float	sum_exp;
	float	log_sum;
	float	loss;

	if (logits.shape().size() != 2)
		throw (std::invalid_argument(
			"CrossEntropy expects 2D logits"));

	if (targets.shape().size() != 1)
		throw (std::invalid_argument(
			"CrossEntropy expects 1D targets"));

	rows = logits.shape()[0];
	columns = logits.shape()[1];

	if (targets.shape()[0] != rows)
		throw (std::invalid_argument(
			"Invalid target size"));

	loss = 0.0f;
	i = 0;

	while (i < rows)
	{
		max_value = logits.data()[i * columns];
		j = 1;

		while (j < columns)
		{
			if (logits.data()[i * columns + j]
				> max_value)
				max_value = logits.data()[
					i * columns + j];
			j++;
		}

		sum_exp = 0.0f;
		j = 0;

		while (j < columns)
		{
			sum_exp += std::exp(
				logits.data()[i * columns + j]
				- max_value);
			j++;
		}

		log_sum = max_value + std::log(sum_exp);

		target = static_cast<size_t>(
			targets.data()[i]);

		if (target >= columns)
			throw (std::out_of_range(
				"Target index out of range"));

		loss -= logits.data()[
			i * columns + target] - log_sum;

		i++;
	}

	return (loss / static_cast<float>(rows));
}

Tensor	CrossEntropy::backward(
	const Tensor& logits,
	const Tensor& targets)
{
	std::vector<size_t>	shape;
	size_t				rows;
	size_t				columns;
	size_t				target;
	size_t				i;
	size_t				j;
	float				max_value;
	float				sum_exp;
	float				probability;

	if (logits.shape().size() != 2)
		throw (std::invalid_argument(
			"CrossEntropy expects 2D logits"));

	if (targets.shape().size() != 1)
		throw (std::invalid_argument(
			"CrossEntropy expects 1D targets"));

	rows = logits.shape()[0];
	columns = logits.shape()[1];

	if (targets.shape()[0] != rows)
		throw (std::invalid_argument(
			"Invalid target size"));

	shape.push_back(rows);
	shape.push_back(columns);

	Tensor	gradient(shape);

	i = 0;
	while (i < rows)
	{
		max_value = logits.data()[i * columns];
		j = 1;

		while (j < columns)
		{
			if (logits.data()[i * columns + j]
				> max_value)
				max_value = logits.data()[
					i * columns + j];
			j++;
		}

		sum_exp = 0.0f;
		j = 0;

		while (j < columns)
		{
			sum_exp += std::exp(
				logits.data()[i * columns + j]
				- max_value);
			j++;
		}

		target = static_cast<size_t>(
			targets.data()[i]);

		if (target >= columns)
			throw (std::out_of_range(
				"Target index out of range"));

		j = 0;
		while (j < columns)
		{
			probability = std::exp(
				logits.data()[i * columns + j]
				- max_value);

			probability /= sum_exp;

			gradient.data()[
				i * columns + j] =
				probability;

			j++;
		}

		gradient.data()[
			i * columns + target] -= 1.0f;

		j = 0;
		while (j < columns)
		{
			gradient.data()[
				i * columns + j] /=
				static_cast<float>(rows);
			j++;
		}

		i++;
	}

	return (gradient);
}
