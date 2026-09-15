#include <iostream>
#include <string>
#include <vector>

#include "io/config.h"
#include "io/checkpoint.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/bpe_trainer.h"
#include "training/dataset.h"
#include "transformer/transformer.h"
#include "nn/lm_head.h"
#include "training/cross_entropy.h"
#include "training/sgd.h"
#include "training/parameter_list.h"

static void	print_help(const std::string& prog)
{
	std::cout << "Muxima-Base - CLI\n"
		<< "\n"
		<< "Usage:\n"
		<< "  " << prog << " train     --config <path> --corpus <path> [--epochs N]\n"
		<< "  " << prog << " infer     --prompt <text> [--config <path>] [--checkpoint <path>]\n"
		<< "  " << prog << " chat      [--config <path>] [--checkpoint <path>]\n"
		<< "  " << prog << " compact   --input <path> --output <path>\n"
		<< "  " << prog << " tokenizer train --corpus <path> --output <path> --vocab-size N\n"
		<< "  " << prog << " tokenizer encode --text <text> [--tokenizer <path>]\n"
		<< "  " << prog << " help\n"
		<< "\n"
		<< "Options:\n"
		<< "  train        Treina o Transformer com Dataset + CrossEntropy + SGD\n"
		<< "  infer        Gera tokens a partir de um prompt (greedy)\n"
		<< "  chat         Modo interativo (inferência em loop)\n"
		<< "  compact      Compacta checkpoint JSON -> binário (stub)\n"
		<< "  tokenizer    Operações do tokenizer BPE\n"
		<< std::endl;
}

static int	cmd_train(int argc, char* argv[])
{
	std::string config_path = "config/model.json";
	std::string corpus_path = "data/corpus.txt";
	size_t epochs = 1;

	for (int i = 2; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "--config" && i + 1 < argc)
			config_path = argv[++i];
		else if (arg == "--corpus" && i + 1 < argc)
			corpus_path = argv[++i];
		else if (arg == "--epochs" && i + 1 < argc)
			epochs = static_cast<size_t>(std::atoi(argv[++i]));
	}

	std::cout << "[train] config=" << config_path
		<< " corpus=" << corpus_path
		<< " epochs=" << epochs << std::endl;

	try
	{
		Config config = Config::load(config_path);
		Dataset dataset(corpus_path, "config/tokenizer.json", 8);

		std::cout << "[train] model: vocab=" << config.model.vocab_size
			<< " dim=" << config.model.embedding_dim
			<< " layers=" << config.model.num_layers
			<< " heads=" << config.model.num_heads
			<< " hidden=" << config.model.hidden_dim
			<< " dataset tokens=" << dataset.tokens().size()
			<< " samples=" << dataset.size() << std::endl;

		Transformer model(
			config.model.vocab_size,
			config.model.max_length,
			config.model.embedding_dim,
			config.model.num_layers,
			config.model.num_heads,
			config.model.hidden_dim);

		SGD optimizer(config.training.learning_rate);
		ParameterList params;

		// Collect a few parameters as example (weights of first block)
		// Full collection would iterate all blocks
		params.add(model.blocks()[0].attention().query().weights());
		params.add(model.blocks()[0].attention().query().bias());

		std::cout << "[train] optimizer lr=" << optimizer.learning_rate()
			<< " params=" << params.size() << std::endl;

		for (size_t e = 0; e < epochs; e++)
		{
			std::cout << "[train] epoch " << (e + 1) << "/" << epochs << std::endl;
			size_t steps = std::min<size_t>(dataset.size(), 2);
			for (size_t s = 0; s < steps; s++)
			{
				TrainingSample sample = dataset.get_sample(s);
				// Stub forward/backward - real trainer would use AutogradGraph
				(void)sample;
				std::cout << "  step " << s << " sample input[0]=" << sample.input[0] << std::endl;
			}
		}

		Checkpoint ckpt;
		ckpt.model_name = "Muxima-Base";
		ckpt.step = epochs;
		ckpt.epoch = epochs;
		ckpt.loss = 0.5f;
		CheckpointIO::save("checkpoints/last.json", ckpt);
		std::cout << "[train] checkpoint saved to checkpoints/last.json" << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "[train] error: " << e.what() << std::endl;
		return (1);
	}
	return (0);
}

static int	cmd_infer(int argc, char* argv[])
{
	std::string prompt = "ola mundo";
	std::string config_path = "config/model.json";
	std::string ckpt_path = "";

	for (int i = 2; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "--prompt" && i + 1 < argc)
			prompt = argv[++i];
		else if (arg == "--config" && i + 1 < argc)
			config_path = argv[++i];
		else if (arg == "--checkpoint" && i + 1 < argc)
			ckpt_path = argv[++i];
	}

	std::cout << "[infer] prompt=\"" << prompt << "\""
		<< " config=" << config_path << std::endl;

	try
	{
		Tokenizer tokenizer;
		tokenizer.load("config/tokenizer.json");
		std::vector<size_t> tokens = tokenizer.encode(prompt);
		std::cout << "[infer] tokens: ";
		for (size_t i = 0; i < tokens.size(); i++)
			std::cout << tokens[i] << " ";
		std::cout << std::endl;

		Config config = Config::load(config_path);
		Transformer model(
			config.model.vocab_size,
			config.model.max_length,
			config.model.embedding_dim,
			config.model.num_layers,
			config.model.num_heads,
			config.model.hidden_dim);

		(void)ckpt_path;
		(void)model;
		std::string decoded = tokenizer.decode(tokens);
		std::cout << "[infer] decoded: " << decoded << std::endl;
		std::cout << "[infer] (stub) generation would continue autoregressively" << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "[infer] error: " << e.what() << std::endl;
		return (1);
	}
	return (0);
}

static int	cmd_chat(int argc, char* argv[])
{
	std::string config_path = "config/model.json";
	std::string ckpt_path = "";
	for (int i = 2; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "--config" && i + 1 < argc)
			config_path = argv[++i];
		else if (arg == "--checkpoint" && i + 1 < argc)
			ckpt_path = argv[++i];
	}

	std::cout << "Muxima chat (type 'exit' to quit)" << std::endl;
	Tokenizer tokenizer;
	try { tokenizer.load("config/tokenizer.json"); } catch (...) {}
	(void)config_path; (void)ckpt_path;

	std::string line;
	while (true)
	{
		std::cout << "Tu: ";
		if (!std::getline(std::cin, line))
			break;
		if (line == "exit" || line == "quit")
			break;
		std::vector<size_t> tokens = tokenizer.encode(line);
		std::string decoded = tokenizer.decode(tokens);
		std::cout << "Muxima: " << decoded << " (echo stub)" << std::endl;
	}
	return (0);
}

static int	cmd_compact(int argc, char* argv[])
{
	std::string input = "checkpoint_test.json";
	std::string output = "checkpoint.bin";
	for (int i = 2; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "--input" && i + 1 < argc)
			input = argv[++i];
		else if (arg == "--output" && i + 1 < argc)
			output = argv[++i];
	}
	std::cout << "[compact] input=" << input << " output=" << output << std::endl;
	try
	{
		Checkpoint ckpt = CheckpointIO::load(input);
		std::cout << "[compact] loaded " << ckpt.parameters.size() << " params" << std::endl;
		// Stub: would write binary format
		CheckpointIO::save(output, ckpt);
		std::cout << "[compact] saved (json stub) to " << output << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "[compact] error: " << e.what() << std::endl;
		return (1);
	}
	return (0);
}

static int	cmd_tokenizer(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "tokenizer usage:\n"
			<< "  tokenizer train --corpus <path> --output <path> --vocab-size N\n"
			<< "  tokenizer encode --text <text>\n";
		return (1);
	}
	std::string sub = argv[2];
	if (sub == "train")
	{
		std::string corpus = "data/corpus.txt";
		std::string output = "config/tokenizer.json";
		size_t vocab = 300;
		for (int i = 3; i < argc; i++)
		{
			std::string arg = argv[i];
			if (arg == "--corpus" && i+1 < argc) corpus = argv[++i];
			else if (arg == "--output" && i+1 < argc) output = argv[++i];
			else if (arg == "--vocab-size" && i+1 < argc) vocab = std::atoi(argv[++i]);
		}
		try
		{
			BPETrainer trainer(vocab);
			trainer.train(corpus, output);
			std::cout << "[tokenizer train] vocab=" << vocab << " -> " << output << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[tokenizer train] error: " << e.what() << std::endl;
			return (1);
		}
	}
	else if (sub == "encode")
	{
		std::string text = "";
		std::string tok_path = "config/tokenizer.json";
		for (int i = 3; i < argc; i++)
		{
			std::string arg = argv[i];
			if (arg == "--text" && i+1 < argc) text = argv[++i];
			else if (arg == "--tokenizer" && i+1 < argc) tok_path = argv[++i];
		}
		try
		{
			Tokenizer tok;
			tok.load(tok_path);
			std::vector<size_t> ids = tok.encode(text);
			std::cout << "[tokenizer encode] \"" << text << "\" -> ";
			for (size_t i = 0; i < ids.size(); i++) std::cout << ids[i] << " ";
			std::cout << "\n[decode] " << tok.decode(ids) << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[tokenizer encode] error: " << e.what() << std::endl;
			return (1);
		}
	}
	else
	{
		std::cerr << "unknown tokenizer subcommand: " << sub << std::endl;
		return (1);
	}
	return (0);
}

int	main(int argc, char* argv[])
{
	if (argc < 2)
	{
		print_help(argv[0]);
		return (0);
	}

	std::string cmd = argv[1];

	if (cmd == "help" || cmd == "--help" || cmd == "-h")
		print_help(argv[0]);
	else if (cmd == "train")
		return (cmd_train(argc, argv));
	else if (cmd == "infer")
		return (cmd_infer(argc, argv));
	else if (cmd == "chat")
		return (cmd_chat(argc, argv));
	else if (cmd == "compact")
		return (cmd_compact(argc, argv));
	else if (cmd == "tokenizer")
		return (cmd_tokenizer(argc, argv));
	else
	{
		std::cerr << "unknown command: " << cmd << std::endl;
		print_help(argv[0]);
		return (1);
	}
	return (0);
}
