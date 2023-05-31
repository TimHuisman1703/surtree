#include "nelson_aalen_estimator.h"

#include <algorithm>

namespace MurTree
{
NelsonAalenEstimator::NelsonAalenEstimator(BinaryDataInternal& data) {
	std::unordered_map<double, std::pair<int, int>> events;

	int at_risk = 0;
	for (const FeatureVectorBinary* fv : data.GetInstances())
	{
		double time = fv->GetTime();
		int event = fv->GetEvent();

		if (events.find(time) == events.end()) {
			keys.push_back(time);
			events[time] = { 0, 0 };
		}

		if (event == 0) {
			events[time].first++;
		}
		else {
			events[time].second++;
		}

		at_risk++;
	}

	std::sort(keys.begin(), keys.end());

	double sum = 0;
	for (double time : keys) {
		int censored = events[time].first;
		int died = events[time].second;

		sum += (float)died / at_risk;
		at_risk -= died + censored;

		values.push_back(sum);
		map[time] = sum;
	}

	ApplyCumulativeHazard(data, true);
}

void NelsonAalenEstimator::ApplyCumulativeHazard(BinaryDataInternal& data, bool spot_on) {
	for (FeatureVectorBinary* fv : data.GetInstances())
	{
		ApplyCumulativeHazard(fv);
	}
}

void NelsonAalenEstimator::ApplyCumulativeHazard(FeatureVectorBinary* fv, bool spot_on) {
	double hazard = CumulativeHazard(fv->GetTime(), false);
	fv->SetHazard(hazard);
}

double NelsonAalenEstimator::CumulativeHazard(double time, bool spot_on) {
	if (spot_on)
		return map[time];

	int left = 0;
	int right = (int)keys.size() - 1;
	while (left != right) {
		int mid = (left + right + 1) / 2;

		if (keys[mid] > time) {
			right = mid - 1;
		}
		else {
			left = mid;
		}
	}
	return values[left];
}

}//end namespace MurTree