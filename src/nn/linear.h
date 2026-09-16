#ifndef LINEAR_H
# define LINEAR_H

# include "../core/autograd_graph.h"
# include "../core/variable.h"

class Linear
{
	private:
		Variable	_weights;
		Variable	_bias;

		static Tensor	create_weights(
			size_t input_size,
			size_t output_size);

		static Tensor	create_bias(
			size_t output_size);

	public:
		Linear(
			size_t input_size,
			size_t output_size);

			Tensor	forward(const Tensor& input);

		Variable*	forward(Variable& input, AutogradGraph& graph);

		Variable&	weights();
		Variable&	bias();
};

#endif
