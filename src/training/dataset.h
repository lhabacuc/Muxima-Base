#ifndef DATASET_H
# define DATASET_H

# include "../tokenizer/tokenizer.h"

# include <cstddef>
# include <string>
# include <vector>

struct TrainingSample
{
	std::vector<size_t>	input;
	std::vector<size_t>	target;
};

class Dataset
{
	private:
		std::vector<size_t>	_tokens;
		size_t				_sequence_length;
		Tokenizer			_tokenizer;

		void	load_tokens(
			const std::string& path);

		void	add_text(
			const std::string& text);

	public:
		Dataset(
			const std::string& corpus_path,
			const std::string& tokenizer_path,
			size_t sequence_length);

		TrainingSample	get_sample(
			size_t index) const;

		size_t	size() const;
		size_t	sequence_length() const;

		const std::vector<size_t>&	tokens() const;
};

#endif
