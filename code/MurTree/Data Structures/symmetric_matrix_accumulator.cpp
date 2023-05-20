//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#include "symmetric_matrix_accumulator.h"

#include <memory>
#include <assert.h>
#include <iostream>

namespace MurTree
{
SymmetricMatrixAccumulator::SymmetricMatrixAccumulator(int num_features) :
	accs_(0),
	num_features_(num_features)
{
	accs_ = new double[NumElements()];
	ResetToZeros();
}

SymmetricMatrixAccumulator::~SymmetricMatrixAccumulator()
{
	delete[] accs_;
	accs_ = 0;
}

double SymmetricMatrixAccumulator::operator()(int feature1, int feature2) const
{
	assert(feature1 <= feature2);
	int index = IndexSymmetricMatrix(feature1, feature2);
	return accs_[index];
}

double& SymmetricMatrixAccumulator::operator()(int feature1, int feature2)
{
	assert(feature1 <= feature2);
	int index = IndexSymmetricMatrix(feature1, feature2);
	return accs_[index];
}

void SymmetricMatrixAccumulator::ResetToZeros()
{
	for (int i = 0; i < NumElements(); i++) { accs_[i] = 0; }
	//memset(data2d_, 0, sizeof(int)*NumElements());
}

int SymmetricMatrixAccumulator::NumElements() const
{
	//the matrix is symmetric, and effectively each entry stores num_labels entries, corresponding the counts for each label
	return (num_features_ * (num_features_ + 1)) / 2;
}

int SymmetricMatrixAccumulator::IndexSymmetricMatrix(int index_row, int index_column) const
{
	assert(index_row <= index_column);
	return num_features_ * index_row + index_column - index_row * (index_row + 1) / 2;
}
}