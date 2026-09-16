#ifndef TRAINER_H
# define TRAINER_H

# include "../transformer/transformer.h"
# include "../training/dataset.h"
# include "../training/cross_entropy.h"
# include "../training/optimizer.h"
# include "../training/parameter_list.h"

class Trainer
{
	private:
		Transformer		&_model;
		Dataset			&_dataset;
		Optimizer		&_optimizer;
		ParameterList	&_parameters;

		float	train_sample(const TrainingSample& sample);

	public:
		Trainer(Transformer& model, Dataset& dataset,
			Optimizer& optimizer, ParameterList& parameters);

		void	train(size_t epochs);
};

#endif
