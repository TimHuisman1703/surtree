#pragma once

#include <vector>
#include <unordered_map>
#include <assert.h>

#include "../Data Structures/binary_data.h"

namespace MurTree
{

class NelsonAalenEstimator
{
public:
	NelsonAalenEstimator(BinaryDataInternal& data);

	void ApplyCumulativeHazard(BinaryDataInternal& data, bool spot_on = false);
	void ApplyCumulativeHazard(FeatureVectorBinary* fv, bool spot_on = false);

	double CumulativeHazard(double time, bool spot_on = false);

//private:
	std::vector<double> keys;
	std::vector<double> values;
	std::unordered_map<double, double> map;
};
}