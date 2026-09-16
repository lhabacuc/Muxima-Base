#include "trainer.h"

#include "../core/autograd_graph.h"
#include "../nn/lm_head.h"

#include <iostream>

Trainer::Trainer(Transformer& model, Dataset& dataset,
	Optimizer& optimizer, ParameterList& parameters)
	: _model(model),
	_dataset(dataset),
	_optimizer(optimizer),
	_parameters(parameters)
{
}

float	Trainer::train_sample(const TrainingSample& sample)
{
	std::vector<size_t> input_shape;
	input_shape.push_back(sample.input.size());
	Tensor input_tensor(input_shape);
	for (size_t i = 0; i < sample.input.size(); i++)
		input_tensor.data()[i] = static_cast<float>(sample.input[i]);

	std::vector<size_t> target_shape;
	target_shape.push_back(sample.target.size());
	Tensor target_tensor(target_shape);
	for (size_t i = 0; i < sample.target.size(); i++)
		target_tensor.data()[i] = static_cast<float>(sample.target[i]);

	AutogradGraph graph;
	Variable input_var(input_tensor, false);
	Variable* hidden = _model.forward(input_var, graph);
	(void)hidden;

	// Dummy logits with vocab large enough for dataset (500)
	std::vector<size_t> logit_shape;
	logit_shape.push_back(sample.input.size());
	logit_shape.push_back(500);
	Tensor dummy_logits(logit_shape);
	for (size_t i = 0; i < dummy_logits.size(); i++)
		dummy_logits.data()[i] = 0.1f * static_cast<float>(i % 10);
	Variable logits_var(dummy_logits, true);

	Variable* loss = graph.cross_entropy(logits_var, target_tensor);
	graph.backward(*loss);

	float loss_val = loss->value().data()[0];
	_parameters.step(_optimizer);
	_parameters.zero_grad(_optimizer);

	return (loss_val);
}

void	Trainer::train(size_t epochs)
{
	for (size_t e = 0; e < epochs; e++)
	{
		std::cout << "[trainer] epoch " << (e + 1) << "/" << epochs << std::endl;
		float total_loss = 0.0f;
		size_t steps = std::min<size_t>(_dataset.size(), 3);
		for (size_t s = 0; s < steps; s++)
		{
			TrainingSample sample = _dataset.get_sample(s);
			float loss = train_sample(sample);
			total_loss += loss;
			std::cout << "  step " << s << " loss " << loss << std::endl;
		}
		std::cout << "  epoch loss " << (total_loss / steps) << std::endl;
	}
}
