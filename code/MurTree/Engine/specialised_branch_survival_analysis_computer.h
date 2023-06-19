//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

#include "../Data Structures/binary_data.h"
#include "../Data Structures/symmetric_matrix_accumulator.h"
#include "../Data Structures/symmetric_matrix_counter.h"

namespace MurTree
{
class SpecialisedBranchSurvivalAnalysisComputer
{
public:
	SpecialisedBranchSurvivalAnalysisComputer(int num_features, int num_feature_vectors, bool using_incremental_updates); //assumes the feature vectors ID are in the range of [0, ..., num_feature_vectors)

	bool Initialise(BinaryDataInternal& data); //returns true if any changed have been made; otherwise false. This is useful since if no changes were necessary we may use the results we previously computed.
	int ProbeDifference(BinaryDataInternal& data);

	double PenaltyBranchOneOne(int f1, int f2);
	double PenaltyBranchOneZero(int f1, int f2);
	double PenaltyBranchZeroOne(int f1, int f2);
	double PenaltyBranchZeroZero(int f1, int f2);

	bool HasEventBranchOneOne(int f1, int f2);
	bool HasEventBranchOneZero(int f1, int f2);
	bool HasEventBranchZeroOne(int f1, int f2);
	bool HasEventBranchZeroZero(int f1, int f2);

	double CalculateError(double hazard_sum, int event_sum, double negative_log_hazard_sum);

	void UpdateCounts(BinaryDataInternal& data, int value);

	SymmetricMatrixAccumulator hazard_sum_;
	SymmetricMatrixCounter event_sum_;
	SymmetricMatrixAccumulator negative_log_hazard_sum_;
	double total_hazard_sum_;
	int total_event_sum_;
	double total_negative_log_hazard_sum_;
	BinaryDataInternal data_old_, data_to_remove_, data_to_add_;
	bool using_incremental_updates_;
};
}