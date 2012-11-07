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

void mds_process_matrix(DenseSymmetricMatrix& distance_matrix)
{
	timed_context context("Multidimensional distance matrix processing");

	unsigned int N = distance_matrix.cols();
	DenseVector col_means(N);
	DenseVector row_means(N);
	for (unsigned int i=0; i<N; ++i)
	{
		col_means[i] = distance_matrix.col(i).mean();
		row_means[i] = distance_matrix.row(i).mean();
	}
	double grand_mean = distance_matrix.mean();
	distance_matrix.array() += grand_mean;
	distance_matrix.rowwise() -= col_means.transpose();
	distance_matrix.colwise() -= row_means;
	distance_matrix.array() *= -0.5;
};
#endif
