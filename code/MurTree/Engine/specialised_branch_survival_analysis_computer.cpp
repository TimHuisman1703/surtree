//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "binary_data_difference_computer.h"
#include "specialised_branch_survival_analysis_computer.h"
#include "../Utilities/runtime_assert.h"

namespace MurTree
{
SpecialisedBranchSurvivalAnalysisComputer::SpecialisedBranchSurvivalAnalysisComputer(int num_features, int num_total_data_points, bool using_incremental_updates) :
	hazard_sum_(num_features),
	event_sum_(num_features),
	negative_log_hazard_sum_(num_features),
	instances_amount_(num_features),
	total_hazard_sum_(0),
	total_event_sum_(0),
	total_negative_log_hazard_sum_(0),
	total_instances_amount_(0),
	using_incremental_updates_(using_incremental_updates),
	data_old_(num_features), 
	data_to_remove_(num_features), 
	data_to_add_(num_features)
{
}

bool SpecialisedBranchSurvivalAnalysisComputer::Initialise(BinaryDataInternal& data_new)
{
	if (using_incremental_updates_)
	{
		//note that the function updates data_to_add and data_to_remove
		BinaryDataDifferenceComputer::ComputeDifference(data_old_, data_new, data_to_add_, data_to_remove_);
		
		if (data_to_add_.Size() + data_to_remove_.Size() == 0) { return false; }
		
		data_old_ = data_new;
	}

	if (using_incremental_updates_ && data_to_add_.Size() + data_to_remove_.Size() < data_new.Size())
	{
		//remove incrementally
		UpdateCounts(data_to_remove_, -1);
		UpdateCounts(data_to_add_, +1);
	}
	else
	{
		//compute from scratch
		hazard_sum_.ResetToZeros();
		event_sum_.ResetToZeros();
		negative_log_hazard_sum_.ResetToZeros();
		instances_amount_.ResetToZeros();
		total_hazard_sum_ = 0.0;
		total_event_sum_ = 0;
		total_negative_log_hazard_sum_ = 0.0;
		total_instances_amount_ = 0;
		UpdateCounts(data_new, +1);
	}

	return true;
}

int SpecialisedBranchSurvivalAnalysisComputer::ProbeDifference(BinaryDataInternal& data)
{
	return BinaryDataDifferenceComputer::ComputeDifferenceMetrics(data_old_, data).total_difference;
}

double SpecialisedBranchSurvivalAnalysisComputer::PenaltyBranchOneOne(int f1, int f2)
{
	double hazard_sum = hazard_sum_(f1, f2);
	int event_sum = event_sum_(f1, f2);
	double negative_log_hazard_sum = negative_log_hazard_sum_(f1, f2);
	int instances_amount = instances_amount_(f1, f2);
	return CalculateError(hazard_sum, event_sum, negative_log_hazard_sum, instances_amount);
}

double SpecialisedBranchSurvivalAnalysisComputer::PenaltyBranchOneZero(int f1, int f2)
{
	double hazard_sum = hazard_sum_(f1, f1) - hazard_sum_(f1, f2);
	int event_sum = event_sum_(f1, f1) - event_sum_(f1, f2);
	double negative_log_hazard_sum = negative_log_hazard_sum_(f1, f1) - negative_log_hazard_sum_(f1, f2);
	int instances_amount = total_instances_amount_ - instances_amount_(f1, f2);
	return CalculateError(hazard_sum, event_sum, negative_log_hazard_sum, instances_amount);
}

double SpecialisedBranchSurvivalAnalysisComputer::PenaltyBranchZeroOne(int f1, int f2)
{
	double hazard_sum = hazard_sum_(f2, f2) - hazard_sum_(f1, f2);
	int event_sum = event_sum_(f2, f2) - event_sum_(f1, f2);
	double negative_log_hazard_sum = negative_log_hazard_sum_(f2, f2) - negative_log_hazard_sum_(f1, f2);
	int instances_amount = instances_amount_(f2, f2) - instances_amount_(f1, f2);
	return CalculateError(hazard_sum, event_sum, negative_log_hazard_sum, instances_amount);
}

double SpecialisedBranchSurvivalAnalysisComputer::PenaltyBranchZeroZero(int f1, int f2)
{
	double hazard_sum = total_hazard_sum_ - hazard_sum_(f1, f1) - hazard_sum_(f2, f2) + hazard_sum_(f1, f2);
	int event_sum = total_event_sum_ - event_sum_(f1, f1) - event_sum_(f2, f2) + event_sum_(f1, f2);
	double negative_log_hazard_sum = total_negative_log_hazard_sum_ - negative_log_hazard_sum_(f1, f1) - negative_log_hazard_sum_(f2, f2) + negative_log_hazard_sum_(f1, f2);
	int instances_amount = total_instances_amount_ - instances_amount_(f1, f1) - instances_amount_(f2, f2) + instances_amount_(f1, f2);
	return CalculateError(hazard_sum, event_sum, negative_log_hazard_sum, instances_amount);
}

bool SpecialisedBranchSurvivalAnalysisComputer::HasEventBranchOneOne(int f1, int f2)
{
	return event_sum_(f1, f2) > 0;
}

bool SpecialisedBranchSurvivalAnalysisComputer::HasEventBranchOneZero(int f1, int f2)
{
	return event_sum_(f1, f1) - event_sum_(f1, f2) > 0;
}

bool SpecialisedBranchSurvivalAnalysisComputer::HasEventBranchZeroOne(int f1, int f2)
{
	return event_sum_(f2, f2) - event_sum_(f1, f2) > 0;
}

bool SpecialisedBranchSurvivalAnalysisComputer::HasEventBranchZeroZero(int f1, int f2)
{
	return total_event_sum_ - event_sum_(f1, f1) - event_sum_(f2, f2) + event_sum_(f1, f2) > 0;
}

double SpecialisedBranchSurvivalAnalysisComputer::CalculateError(double hazard_sum, int event_sum, double negative_log_hazard_sum, int instances_amount)
{
	// There has to be at least one person in a leaf who died
	if (event_sum == 0)
		return 0;

	double theta = event_sum / hazard_sum;
	double error = negative_log_hazard_sum
		- event_sum * (log(theta) + 1)
		+ hazard_sum * theta;

	return std::max(error, 0.0);
}

void SpecialisedBranchSurvivalAnalysisComputer::UpdateCounts(BinaryDataInternal& data, int value)
{
	for (FeatureVectorBinary *instance : data.GetInstances())
	{
		double hazard = instance->GetHazard();
		double new_hazard_sum = hazard * value;
		int new_event_sum = instance->GetEvent() * value;
		double new_negative_log_hazard_sum = 0;
		if (instance->GetEvent()) {
			new_negative_log_hazard_sum = -log(hazard) * value;
		}
		int new_instances_amount = value;

		total_hazard_sum_ += new_hazard_sum;
		total_event_sum_ += new_event_sum;
		total_negative_log_hazard_sum_ += new_negative_log_hazard_sum;
		total_instances_amount_ += new_instances_amount;

		int num_present_features = instance->NumPresentFeatures();
		for (int i = 0; i < num_present_features; i++)
		{
			int f1 = instance->GetJthPresentFeature(i);
			for (int j = i; j < num_present_features; j++)
			{
				int f2 = instance->GetJthPresentFeature(j);

				hazard_sum_(f1, f2) += new_hazard_sum;
				event_sum_(f1, f2) += new_event_sum;
				negative_log_hazard_sum_(f1, f2) += new_negative_log_hazard_sum;
				instances_amount_(f1, f2) += new_instances_amount;
			}
		}
	}
}

}
