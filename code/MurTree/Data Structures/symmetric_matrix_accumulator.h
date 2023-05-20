//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

namespace MurTree
{
	//it is implemented as a symmetric matirix
	class SymmetricMatrixAccumulator
	{
	public:
		SymmetricMatrixAccumulator(int num_features);
		~SymmetricMatrixAccumulator();

		double operator()(int feature1, int feature2) const;
		double& operator()(int feature1, int feature2);

		void ResetToZeros();

	private:
		int NumElements() const;
		int IndexSymmetricMatrix(int index_row, int index_column) const;

		double* accs_;
		int num_features_;
	};
}