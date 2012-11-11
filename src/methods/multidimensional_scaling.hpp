#ifndef tapkee_mds_h_
#define tapkee_mds_h_

#include "../defines.hpp"
#include "../utils/time.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                             PairwiseCallback callback)
{
	timed_context context("Multidimensional scaling distance matrix computation");

	DenseMatrix distance_matrix(end-begin,end-begin);

	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			double d = callback(*i_iter,*j_iter);
			d *= d;
			distance_matrix(i_iter-begin,j_iter-begin) = d;
			distance_matrix(j_iter-begin,i_iter-begin) = d;
		}
	}
	return distance_matrix;
};

void mds_process_matrix(const DenseSymmetricMatrix& const_distance_matrix)
{
	timed_context context("Multidimensional distance matrix processing");

	DenseSymmetricMatrix& distance_matrix = const_cast<DenseSymmetricMatrix&>(const_distance_matrix); 
	double grand_mean = 0.0;
	DenseVector col_means;
	{
		timed_context c("Computing means");
		col_means = distance_matrix.colwise().mean();
		grand_mean = distance_matrix.mean();
	}
	{
		timed_context c("Mutating matrix");
		distance_matrix.array() += grand_mean;
		distance_matrix.colwise() -= col_means;
		distance_matrix.rowwise() -= col_means.transpose();
		distance_matrix.array() *= -0.5;
	}
};
#endif
