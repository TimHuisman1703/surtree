﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

//Code for the paper: 
// "MurTree: optimal classification trees via dynamic programming and search", JMLR, 2022.
//  Note that some of the comments directly refer to the paper:
// 
//Authors: Emir Demirović, Anna Lukina, Emmanuel Hebrard, Jeffrey Chan, James Bailey, Christopher Leckie, Kotagiri Ramamohanarao, Peter J. Stuckey
//For any issues related to the code, please feel free to contact Dr Emir Demirović, e.demirovic@tudelft.nl

#include "code/MurTree/Engine/solver.h"
#include "code/MurTree/Engine/dataset_cache.h"
#include "code/MurTree/Engine/hyper_parameter_tuner.h"
#include "code/MurTree/Utilities/parameter_handler.h"
#include "code/MurTree/Utilities/stopwatch.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <algorithm>

MurTree::ParameterHandler DefineParameters()
{
	MurTree::ParameterHandler parameters;

	parameters.DefineNewCategory("Main Parameters");
	parameters.DefineNewCategory("Algorithmic Parameters");
	parameters.DefineNewCategory("Tuning Parameters");

	parameters.DefineStringParameter
	(
		"file",
		"Location to the dataset.",
		"", //default value
		"Main Parameters"
	);

	parameters.DefineFloatParameter
	(
		"time",
		"Maximum runtime given in seconds.",
		600, //default value
		"Main Parameters",
		0, //min value
		INT32_MAX //max value
	);

	parameters.DefineIntegerParameter
	(
		"max-depth",
		"Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.",
		3, //default value
		"Main Parameters",
		0, //min value
		20 //max value
	);

	parameters.DefineIntegerParameter
	(
		"max-num-nodes",
		"Maximum number of *decision/feature nodes* allowed. Note that a tree with k feature nodes has k+1 leaf nodes.",
		7, //default value
		"Main Parameters",
		0,
		INT32_MAX
	);

	parameters.DefineFloatParameter
	(
		"sparse-coefficient",
		"Assigns the penalty for using decision/feature nodes. Large sparse coefficients will result in smaller trees.",
		0.0,
		"Main Parameters",
		0.0,
		1.0
	);

	parameters.DefineBooleanParameter
	(
		"verbose",
		"Determines if the solver should print logging information to the standard output.",
		true,
		"Main Parameters"
	);

	parameters.DefineBooleanParameter
	(
		"all-trees",
		"Instructs the algorithm to compute trees using all allowed combinations of max-depth and max-num-nodes. Used to stress-test the algorithm.",
		false,
		"Main Parameters"
	);

	parameters.DefineStringParameter
	(
		"result-file",
		"The results of the algorithm are printed in the provided file, used for simple benchmarking. The output file contains the runtime, misclassification score, and number of cache entries. Leave blank to avoid printing.",
		"", //default value
		"Main Parameters"
	);

	//Internal algorithmic parameters-----------

	parameters.DefineBooleanParameter
	(
		"incremental-frequency",
		"Activate incremental frequency computation, which takes into account previously computed trees when recomputing the frequency. In our experiments proved to be effective on all datasets.",
		true,
		"Algorithmic Parameters"
	);

	parameters.DefineStringParameter
	(
		"node-selection",
		"Node selection strategy used to decide on whether the algorithm should examine the left or right child node first.",
		"dynamic", //default value
		"Algorithmic Parameters",
		{ "dynamic", "post-order" }
	);

	parameters.DefineStringParameter
	(
		"feature-ordering",
		"Feature ordering strategy used to determine the order in which features will be inspected in each node.",
		"in-order", //default value
		"Algorithmic Parameters",
		{ "in-order", "random" }
	);

	parameters.DefineIntegerParameter
	(
		"random-seed",
		"Random seed used only if the feature-ordering is set to random. A seed of -1 assings the seed based on the current time.",
		3,
		"Algorithmic Parameters",
		-1,
		INT32_MAX
	);

	parameters.DefineStringParameter
	(
		"cache-type",
		"Cache type used to store computed subtrees. \"Dataset\" is more powerful than \"branch\" but may required more computational time. Need to be determined experimentally. \"Closure\" is experimental and typically slower than other options.",
		"branch", //default value
		"Algorithmic Parameters",
		{ "branch", "dataset", "closure" }
	);

	parameters.DefineIntegerParameter
	(
		"duplicate-factor",
		"Duplicates the instances the given amount of times. Used for stress-testing the algorithm, not a practical parameter.",
		1,
		"Algorithmic Parameters",
		1,
		INT32_MAX
	);

	parameters.DefineFloatParameter
	(
		"upper-bound",
		"Initial upper bound.",
		DBL_MAX, //default value
		"Algorithmic Parameters",
		0,
		DBL_MAX
	);

	//Tuning parameters

	parameters.DefineBooleanParameter
	(
		"hyper-parameter-tuning",
		"Activate hyper-parameter tuning using max-depth and max-num-nodes as the maximum values allowed. The splits need to be provided in the appropriate folder...see the code. todo",
		false,
		"Tuning Parameters"
	);

	parameters.DefineStringParameter
	(
		"splits-location-prefix",
		"Prefix to where the splits may be found. Used in combination with hyper-parameter-tuning. The splits need to be provided in the appropriate folder...see the code. todo",
		"", //default value
		"Tuning Parameters"
	);

	parameters.DefineStringParameter
	(
		"hyper-parameter-stats-file",
		"Location of the output file that contains information about the hyper-parameter procedure.",
		"", //default value
		"Tuning Parameters"
	);

	return parameters;
}

void CheckParameters(MurTree::ParameterHandler& parameters)
{
	if (parameters.GetStringParameter("file") == "")
	{
		std::cout << "Error: No file given!\n"; exit(1);
	}

	if (parameters.GetIntegerParameter("max-depth") > parameters.GetIntegerParameter("max-num-nodes"))
	{
		std::cout << "Error: The depth parameter is greater than the number of nodes!\n";
		exit(1);
	}

	if (parameters.GetIntegerParameter("max-num-nodes") > (static_cast<long long>(uint32_t(1)) << parameters.GetIntegerParameter("max-depth")) - 1)
	{
		std::cout << "Error: The number of nodes exceeds the limit imposed by the depth!\n";
		exit(1);
	}

	if (parameters.GetBooleanParameter("hyper-parameter-tuning") && parameters.GetStringParameter("splits-location-prefix") == "")
	{
		std::cout << "Error: hyper tuning specified but no splits given\n";
		exit(1);
	}

	if (parameters.GetBooleanParameter("hyper-parameter-tuning") && parameters.GetStringParameter("hyper-parameter-stats-file") == "")
	{
		std::cout << "Error: hyper tuning specified but no output file location given\n";
		exit(1);
	}
}

void PrintTree(MurTree::DecisionNode* node, std::ofstream& out)
{
	if (node->IsLabelNode()) {
		out << "[";
		out << node->theta_;
		out << "]";
	}
	else {
		out << "[";
		out << node->feature_;
		out << ",";
		PrintTree(node->left_child_, out);
		out << ",";
		PrintTree(node->right_child_, out);
		out << "]";
	}
}

int main(int argc, char* argv[])
{
	MurTree::ParameterHandler parameters = DefineParameters();

	bool manual_activation = false;
	bool single_parameter_set_tuning = false;
	std::string splits_location_prefix = "";
	std::string hyper_parameter_stats_file = "";	

	//standard expected case when run from the console
	//see DefineParameters for info about the parameters
	if (argc > 1)
	{
		parameters.ParseCommandLineArguments(argc, argv);
	}
	//this else statement was mostly used for debugging, now disabled
	else
	{
		std::cout << "Manual activation, no parameters supplied\n";

		manual_activation = true;

		std::string file_location;

		file_location = "datasetsSA\\LeukSurv.txt";

		// Tag for Ctrl+F-ing: SETTINGS

		parameters.SetStringParameter("file", file_location);
		parameters.SetStringParameter("result-file", "result.txt");
		parameters.SetBooleanParameter("hyper-parameter-tuning", false);
		//splits_location_prefix = "datasetsTesting\\MurTreeSplits\\anneal";
		//hyper_parameter_stats_file = "datasetsTesting\\MurTreeHyperStats\\anneal_new.txt";
		single_parameter_set_tuning = false;
		parameters.SetIntegerParameter("duplicate-factor", 1);
		parameters.SetStringParameter("cache-type", "dataset");// "closure";// "dataset";
		parameters.SetBooleanParameter("all-trees", false);
		parameters.SetIntegerParameter("max-depth", 2);
		parameters.SetIntegerParameter("max-num-nodes", 3);
		parameters.SetFloatParameter("upper-bound", DBL_MAX);//INT32_MAX
		parameters.SetFloatParameter("sparse-coefficient", 0.0);
		parameters.SetStringParameter("node-selection", "dynamic");
		parameters.SetStringParameter("feature-ordering", "in-order");

		parameters.SetBooleanParameter("verbose", true);
	}

	CheckParameters(parameters);

	if (parameters.GetBooleanParameter("verbose")) { parameters.PrintParameterValues(); }

	if (parameters.GetIntegerParameter("random-seed") == -1) { srand((int)time(0)); }
	else { srand((int)parameters.GetIntegerParameter("random-seed")); }

	MurTree::Stopwatch stopwatch;
	stopwatch.Initialise(0);

	if (parameters.GetBooleanParameter("hyper-parameter-tuning"))
	{
		std::cout << "NEW HYPER PARAMETER TUNING\n";
		MurTree::HyperParameterTuningInput hyper_p;
		hyper_p.benchmark_location = parameters.GetStringParameter("file");
		hyper_p.partition_files_location_prefix = splits_location_prefix;
		hyper_p.max_depth = (int)parameters.GetIntegerParameter("max-depth");
		hyper_p.max_num_nodes = (int)parameters.GetIntegerParameter("max-num-nodes");
		hyper_p.single_parameter_set_tuning = single_parameter_set_tuning;

		clock_t c_s = clock();
		MurTree::HyperParameterTuningOutput result = MurTree::HyperParameterTuner::Solve(hyper_p, parameters);
		std::cout << "BEST PARAMS: \n\tDepth = " << result.result.decision_tree_->Depth() << "\n\tNum nodes: " << result.result.decision_tree_->NumNodes() << "\n";
		std::cout << "Train/test: " << result.train_accuracy << ", " << result.test_accuracy << "\n";
		std::cout << "CLOCK: " << double(clock() - c_s) / CLOCKS_PER_SEC << "\n";

		std::ofstream f(hyper_parameter_stats_file.c_str());
		f << result.result.decision_tree_->Depth() << "\n";
		f << result.result.decision_tree_->NumNodes() << "\n";
		f << result.train_accuracy << "\n";
		f << result.test_accuracy << "\n";
		f << double(clock() - c_s) / CLOCKS_PER_SEC << "\n";
		f << "[blank]\n";
		f << single_parameter_set_tuning;
		return 0;
	}
	else
	{
		if (parameters.GetBooleanParameter("verbose")) { std::cout << "Optimal tree computation started!\n"; }
		MurTree::Solver mur_tree_solver(parameters);
		if (parameters.GetBooleanParameter("verbose")) {
			std::cout << "Instances = " << mur_tree_solver.binary_data_->NumInstances() << ", ";
			std::cout << "Features = " << mur_tree_solver.binary_data_->NumFeatures() << "\n";
		}
		clock_t clock_before_solve = clock();
		MurTree::SolverResult mur_tree_solver_result = mur_tree_solver.Solve(parameters);
		clock_t clock_after_solve = clock();
		
		if (parameters.GetBooleanParameter("verbose"))
		{
			if (mur_tree_solver_result.decision_tree_)
			{
				std::cout << "ERROR: " << mur_tree_solver_result.error << "\n";
				std::cout << "DEPTH: " << mur_tree_solver_result.decision_tree_->Depth() << "\n";
				std::cout << "NUM NODES: " << mur_tree_solver_result.decision_tree_->NumNodes() << "\n";
				std::cout << "TIME: " << stopwatch.TimeElapsedInSeconds() << " seconds\n";
				std::cout << "CLOCKS FOR SOLVE: " << double(clock_after_solve - clock_before_solve) / CLOCKS_PER_SEC << "\n";
			}
			else
			{
				std::cout << "Timeout, no tree stats\n";
			}
		}

		if (parameters.GetStringParameter("result-file") != "")
		{
			std::ofstream out(parameters.GetStringParameter("result-file").c_str());
			//double clock_time_total = (double(clock() - clock_before_solve) / CLOCKS_PER_SEC);
			//out << clock_time_total << "\n";
			out << "Time: " << double(clock_after_solve - clock_before_solve) / CLOCKS_PER_SEC << "\n";
			
			if (mur_tree_solver_result.decision_tree_) //print stats if finished within the allocated time
			{
				out << "Error: " << mur_tree_solver_result.error << "\n";
				out << "Tree:\n";
				PrintTree(mur_tree_solver_result.decision_tree_, out);
				out << "\n";
			}
		}
	}
	return 0;
}
