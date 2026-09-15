#ifndef AUTOGRAD_GRAPH_H
# define AUTOGRAD_GRAPH_H

# include "variable.h"

# include <vector>

class AutogradGraph
{
	private:
		enum OperationType
		{
			OP_NONE,
			OP_ADD,
			OP_SUBTRACT,
			OP_MULTIPLY,
			OP_MATMUL,
			OP_RELU,
			OP_GELU,
			OP_CROSS_ENTROPY
		};

		struct Node
		{
			OperationType	type;
			Variable		*output;
			Variable		*left;
			Variable		*right;
			Tensor			*targets;

			Node(
				OperationType type,
				Variable *output,
				Variable *left,
				Variable *right,
				Tensor *targets);
		};

		std::vector<Node*>	_nodes;

		Node*	find_node(
			Variable& variable);

		void	backward_node(
			Node *node);

		void	build_topology(
			Variable& variable,
			std::vector<Node*>& topology,
			std::vector<Variable*>& visited);

		void	build_node_topology(
			Node *node,
			std::vector<Node*>& topology,
			std::vector<Variable*>& visited);

	public:
		AutogradGraph();
		~AutogradGraph();

		Variable*	add(
			Variable& left,
			Variable& right);

		Variable*	subtract(
			Variable& left,
			Variable& right);

		Variable*	multiply(
			Variable& left,
			Variable& right);

		Variable*	matmul(
			Variable& left,
			Variable& right);

		Variable*	relu(
			Variable& input);

		Variable*	gelu(
			Variable& input);

		Variable*	cross_entropy(
			Variable& logits,
			const Tensor& targets);

		void	backward(
			Variable& output);
};

#endif
