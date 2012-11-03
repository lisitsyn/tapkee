#ifndef edrt_mds_h_
#define edrt_mds_h_

#include "../defines.hpp"
#include "../utils/time.hpp"

#include <iostream>
using std::cout;
using std::endl;

template <class RandomAccessIterator, class PairwiseCallback>
DenseMatrix mds_distance_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
                                const PairwiseCallback& callback)
{
	timed_context context("Multidimensional scaling distance matrix computation");

	cout << "N = " << end-begin << endl;
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
	DenseVector col_means(end-begin);
	DenseVector row_means(end-begin);
	for (int i=0; i<(end-begin); ++i)
	{
		col_means[i] = distance_matrix.col(i).mean();
		row_means[i] = distance_matrix.row(i).mean();
	}
	double grand_mean = distance_matrix.mean();
	distance_matrix.array() += grand_mean;
	distance_matrix.rowwise() -= col_means.transpose();
	distance_matrix.colwise() -= row_means;
	
	distance_matrix.array() *= -0.5;

	return distance_matrix;
};
#endif
