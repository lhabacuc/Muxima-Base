#include "transformer.h"

#include <stdexcept>
#include <vector>

std::vector<TransformerBlock>	Transformer::create_blocks(
	size_t num_layers,
	size_t embedding_dim,
	size_t num_heads,
	size_t hidden_dim)
{
	std::vector<TransformerBlock>	blocks;
	size_t	i;

	i = 0;
	while (i < num_layers)
	{
		blocks.push_back(
			TransformerBlock(
				embedding_dim,
				num_heads,
				hidden_dim));
		i++;
	}
	return (blocks);
}

Transformer::Transformer(
	size_t vocab_size,
	size_t max_length,
	size_t embedding_dim,
	size_t num_layers,
	size_t num_heads,
	size_t hidden_dim)
	: _embedding_dim(embedding_dim),
	_embedding(vocab_size, embedding_dim),
	_positional(max_length, embedding_dim),
	_final_norm(embedding_dim, 1e-5f),
	_blocks(create_blocks(
		num_layers,
		embedding_dim,
		num_heads,
		hidden_dim))
{
}

Tensor	Transformer::forward(const Tensor& tokens)
{
	size_t	i;

	(void)_embedding_dim;
	if (tokens.shape().size() != 1)
		throw (std::invalid_argument(
			"Transformer expects 1D token tensor"));

	Tensor	x = _embedding.forward(tokens);
	x = _positional.forward(x);

	i = 0;
	while (i < _blocks.size())
	{
		x = _blocks[i].forward(x);
		i++;
	}

	return (_final_norm.forward(x));
}

Embedding&	Transformer::embedding()
{
	return (_embedding);
}

PositionalEncoding&	Transformer::positional()
{
	return (_positional);
}

LayerNorm&	Transformer::final_norm()
{
	return (_final_norm);
}

std::vector<TransformerBlock>&	Transformer::blocks()
{
	return (_blocks);
}

ParameterList	Transformer::parameters()
{
	ParameterList list;

	list.add(_embedding.weights());
	list.add(_final_norm.gamma());
	list.add(_final_norm.beta());

	size_t i = 0;
	while (i < _blocks.size())
	{
		TransformerBlock& block = _blocks[i];

		list.add(block.attention().query().weights());
		list.add(block.attention().query().bias());
		list.add(block.attention().key().weights());
		list.add(block.attention().key().bias());
		list.add(block.attention().value().weights());
		list.add(block.attention().value().bias());
		list.add(block.attention().output().weights());
		list.add(block.attention().output().bias());

		list.add(block.norm1().gamma());
		list.add(block.norm1().beta());

		list.add(block.feed_forward().up().weights());
		list.add(block.feed_forward().up().bias());
		list.add(block.feed_forward().down().weights());
		list.add(block.feed_forward().down().bias());

		list.add(block.norm2().gamma());
		list.add(block.norm2().beta());

		i++;
	}

	return (list);
}
