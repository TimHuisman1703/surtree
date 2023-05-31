//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "file_reader.h"
#include "runtime_assert.h"

#include <iostream>
#include <sstream>
#include <string>
#include <set>

namespace MurTree
{
std::vector<FeatureVectorBinary> FileReader::ReadDataDL(std::string filename, int duplicate_instances_factor)
{
	std::vector<FeatureVectorBinary> feature_vectors;
	
	std::ifstream file(filename.c_str());

	if (!file) { std::cout << "Error: File " << filename << " does not exist!\n"; runtime_assert(file); }

	std::string line;
	int id = 0;
	uint32_t num_features = INT32_MAX;
	while (std::getline(file, line))
	{
		int spaces = 0;
		for (int i = 0; i < line.length(); i++) {
			if (line[i] == ' ') {
				spaces++;
			}
		}
		runtime_assert(num_features == INT32_MAX || num_features == spaces - 1);
		if (num_features == INT32_MAX) { num_features = spaces - 1; }

		std::istringstream iss(line);
		//the first value in the line is the label, followed by 0-1 features
		double time;
		int event;
		iss >> time;
		iss >> event;
		std::vector<bool> v;
		for (uint32_t i = 0; i < num_features; i++)
		{
			uint32_t temp;
			iss >> temp;
			runtime_assert(temp == 0 || temp == 1);
			v.push_back(temp);
		}

		for (int i = 0; i < duplicate_instances_factor; i++)
		{
			feature_vectors.push_back(FeatureVectorBinary(v, time, event, id));
			id++;			
		}
	}
	return feature_vectors;
}

std::vector<SplitData> FileReader::ReadSplits(std::string data_file, std::string splits_location_prefix)
{//splits_location_prefix' + '_train[i].txt'
	std::vector<FeatureVectorBinary> raw_data = ReadDataDL(data_file);
	std::vector<SplitData> splits;
	for (int k = 0; k < 5; k++)
	{
		std::string file_test = splits_location_prefix + "_test" + std::to_string(k) + ".txt";
		std::string file_train = splits_location_prefix + "_train" + std::to_string(k) + ".txt";
		SplitData split;
		split.testing_data = ReadSplits_ReadPartitionFile(raw_data, file_test);
		split.training_data = ReadSplits_ReadPartitionFile(raw_data, file_train);
		splits.push_back(split);
	}
	return splits;
}

std::vector<FeatureVectorBinary> FileReader::ReadSplits_ReadPartitionFile(std::vector<FeatureVectorBinary>& instances, std::string file)
{
	std::ifstream input(file.c_str());
	std::vector<FeatureVectorBinary> partition(instances.size());

	int num_instances = (int)instances.size();
	std::vector<int> labels(num_instances, -1);
	std::vector<int> indicies(num_instances, -1);

	for (int j = 0; j < instances.size(); j++)
	{
		auto& fv = instances[j];
		indicies[fv.GetID()] = j;
	}

	runtime_assert(input);
	int index;
	while (input >> index)
	{
		int i = indicies[index];
		partition.push_back(instances[i]);
	}
	return partition;
}
}