﻿//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirović

#pragma once

namespace MurTree
{
	//it is implemented as a symmetric matirix
	class SymmetricMatrixCounter
	{
	public:
		SymmetricMatrixCounter(int num_features);
		~SymmetricMatrixCounter();

		int operator()(int feature1, int feature2) const;
		int& operator()(int feature1, int feature2);

		void ResetToZeros();

	private:
		int NumElements() const;
		int IndexSymmetricMatrix(int index_row, int index_column) const;

		int* counts_;
		int num_features_;
	};
}