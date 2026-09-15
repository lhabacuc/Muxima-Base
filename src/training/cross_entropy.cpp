#include "cross_entropy.h"

#include <cmath>
#include <stdexcept>

static float	row_loss(
	const Tensor& logits,
	const Tensor& targets,
	size_t row,
	size_t vocab_size)
{
	float	max_value;
	float	sum_exp;
	float	log_sum;
	float	target;
	size_t	i;

	max_value = logits.data()[row * vocab_size];
	i = 1;
	while (i < vocab_size)
	{
		if (logits.data()[row * vocab_size + i] > max_value)
			max_value = logits.data()[row * vocab_size + i];
		i++;
	}

	sum_exp = 0.0f;
	i = 0;
	while (i < vocab_size)
	{
		sum_exp += std::exp(
			logits.data()[row * vocab_size + i] - max_value);
		i++;
	}

	log_sum = max_value + std::log(sum_exp);
	target = targets.data()[row];

	if (target < 0.0f
		|| target >= static_cast<float>(vocab_size))
		throw (std::out_of_range(
			"Target token is outside vocabulary"));

	return (-(logits.data()[row * vocab_size
		+ static_cast<size_t>(target)] - log_sum));
}

float	CrossEntropy::forward(
	const Tensor& logits,
	const Tensor& targets)
{
	size_t	sequence_length;
	size_t	vocab_size;
	size_t	i;
	float	loss;

	if (logits.shape().size() != 2)
		throw (std::invalid_argument(
			"Logits must be a 2D tensor"));

	if (targets.shape().size() != 1)
		throw (std::invalid_argument(
			"Targets must be a 1D tensor"));

	sequence_length = logits.shape()[0];
	vocab_size = logits.shape()[1];

	if (targets.shape()[0] != sequence_length)
		throw (std::invalid_argument(
			"Logits and targets lengths do not match"));

	loss = 0.0f;
	i = 0;
	while (i < sequence_length)
	{
		loss += row_loss(
			logits,
			targets,
			i,
			vocab_size);
		i++;
	}

	return (loss / static_cast<float>(sequence_length));
}