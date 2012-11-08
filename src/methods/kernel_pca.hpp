#ifndef TAPKEE_KERNEL_PCA_H_
#define TAPKEE_KERNEL_PCA_H_

#include "../defines.hpp"
#include "../utils/time.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_centered_kernel_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                                    PairwiseCallback callback)
{
	timed_context context("Constructing kPCA centered kernel matrix");

	DenseMatrix kernel_matrix(end-begin,end-begin);

	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			double k = callback(*i_iter,*j_iter);
			kernel_matrix(i_iter-begin,j_iter-begin) = k;
			kernel_matrix(j_iter-begin,i_iter-begin) = k;
		}
	}

	double grand_mean = 0.0;
	DenseVector col_means;
	{
		timed_context c("Computing means");
		col_means = kernel_matrix.colwise().mean();
		grand_mean = kernel_matrix.mean();
	}
	{
		timed_context c("Mutating matrix");
		kernel_matrix.array() += grand_mean;
		kernel_matrix.colwise() -= col_means;
		kernel_matrix.rowwise() -= col_means.transpose();
	}

	return kernel_matrix.selfadjointView<Eigen::Upper>();
};

#endif
