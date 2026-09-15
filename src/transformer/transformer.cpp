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
