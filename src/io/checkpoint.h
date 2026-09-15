#ifndef CHECKPOINT_H
# define CHECKPOINT_H

# include "../core/tensor.h"

# include <string>
# include <vector>

struct CheckpointParameter
{
	std::string			name;
	std::vector<size_t>	shape;
	std::vector<float>	data;
};

struct Checkpoint
{
	std::string					model_name;
	size_t						step;
	size_t						epoch;
	float						loss;
	std::vector<CheckpointParameter>	parameters;
};

class CheckpointIO
{
	public:
		static void	save(
			const std::string& path,
			const Checkpoint& checkpoint);

		static Checkpoint	load(
			const std::string& path);

		static CheckpointParameter	from_tensor(
			const std::string& name,
			const Tensor& tensor);

		static Tensor	to_tensor(
			const CheckpointParameter& parameter);
};

#endif
