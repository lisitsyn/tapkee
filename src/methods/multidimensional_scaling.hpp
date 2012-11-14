#ifndef tapkee_mds_h_
#define tapkee_mds_h_

#include "../defines.hpp"
#include "../utils/time.hpp"

#include <algorithm>

using std::random_shuffle;
using std::fill;

template <class RandomAccessIterator>
std::vector<RandomAccessIterator> select_landmarks_random(RandomAccessIterator begin, RandomAccessIterator end, DefaultScalarType ratio)
{
	std::vector<RandomAccessIterator> landmarks;
	landmarks.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
		landmarks.push_back(iter);
	random_shuffle(landmarks.begin(),landmarks.end());
	landmarks.erase(landmarks.begin() + static_cast<unsigned int>(landmarks.size()*ratio),landmarks.end());
	return landmarks;
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(std::vector<RandomAccessIterator> indices, 
                                             PairwiseCallback callback)
{
	timed_context context("Multidimensional scaling distance matrix computation");

	DenseMatrix distance_matrix(indices.size(),indices.size());

	for (typename std::vector<RandomAccessIterator>::const_iterator i_iter=indices.begin(); i_iter!=indices.end(); ++i_iter)
	{
		for (typename std::vector<RandomAccessIterator>::const_iterator j_iter=i_iter; j_iter!=indices.end(); ++j_iter)
		{
			DefaultScalarType d = callback(**i_iter,**j_iter);
			d *= d;
			distance_matrix(i_iter-indices.begin(),j_iter-indices.begin()) = d;
			distance_matrix(j_iter-indices.begin(),i_iter-indices.begin()) = d;
		}
	}
	return distance_matrix;
};

template <class RandomAccessIterator, class PairwiseCallback>
EmbeddingResult triangulate(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback distance_callback,
                            const std::vector<RandomAccessIterator>& landmarks, const DenseSymmetricMatrix& landmarks_distance_matrix, 
                            const EmbeddingResult& landmarks_embedding, unsigned int target_dimension)
{
	timed_context context("Landmark triangulation");
	
	bool* to_process = new bool[end-begin];
	fill(to_process,to_process+(end-begin),true);
	
	DenseMatrix embedding((end-begin),target_dimension);

	for (typename std::vector<RandomAccessIterator>::const_iterator iter=landmarks.begin(); 
			iter!=landmarks.end(); ++iter)
	{
		to_process[*iter-begin] = false;
		embedding.row(*iter-begin) = landmarks_embedding.first.row(iter-landmarks.begin());
	}

//	for (unsigned int i=0; i<target_dimension; ++i)
//		landmarks_embedding.first.col(i) /= 1.0;//sqrt(landmarks_embedding.second(i));

	DenseVector landmark_distances_squared = landmarks_distance_matrix.colwise().mean().array().square();
	DenseVector distances_to_landmarks(landmarks.size());

	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		if (!to_process[iter-begin])
			continue;

		for (unsigned int i=0; i<distances_to_landmarks.size(); ++i)
		{
			DefaultScalarType d = distance_callback(*iter,*landmarks[i]);
			distances_to_landmarks[i] = d*d;
		}

		distances_to_landmarks -= landmark_distances_squared;

		embedding.row(iter-begin) = -0.5*landmarks_embedding.first.transpose()*distances_to_landmarks;
	}

	delete[] to_process;

	return EmbeddingResult(embedding,DenseVector());
}

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
			DefaultScalarType d = callback(*i_iter,*j_iter);
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
	DefaultScalarType grand_mean = 0.0;
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
