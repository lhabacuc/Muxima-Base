#ifndef BPE_TRAINER_H
# define BPE_TRAINER_H

# include <map>
# include <string>
# include <vector>

class BPETrainer
{
	private:
		std::map<std::string, size_t>	_vocab;
		std::vector<std::pair<std::string, std::string> > _merges;

		size_t	_vocab_size;
		size_t	_next_id;

		void	initialize_vocab(
			const std::vector<std::string>& words);

		std::map<std::string, size_t>	count_pairs(
			const std::vector<std::vector<std::string> >& words) const;

		void	merge_pair(
			std::vector<std::string>& word,
			const std::string& left,
			const std::string& right);

		std::string	merge_key(
			const std::string& left,
			const std::string& right) const;

	public:
		BPETrainer(size_t vocab_size);

		void	train(
			const std::string& input_path,
			const std::string& output_path);
};

#endif
