#ifndef libedrt_neighbors_h_
#define libedrt_neighbors_h_

#include "../defines.hpp"
#include "covertree.hpp"
#include <vector>
#include <utility>

using std::pair;

template <class DistanceRecord>
struct distances_comparator
{
	bool operator()(const DistanceRecord& l, const DistanceRecord& r) 
	{
		return (l.second < r.second);
	}
};

template <class RandomAccessIterator, class KernelCallback>
struct kernel_distance
{
	kernel_distance(const KernelCallback& kc) : kc_(kc) {};
	KernelCallback kc_;
	double operator()(const pair<double, RandomAccessIterator>& l, const pair<double, RandomAccessIterator>& r) const
	{
		return l.first + r.first - 2*kc_(*(l.second),*(r.second));
	}
};

template <class RandomAccessIterator, class PairwiseCallback>
Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, 
                         PairwiseCallback callback, unsigned int k)
{
	timed_context context("Covertree-based neighbors search");

	kernel_distance<RandomAccessIterator, PairwiseCallback> kd(callback);
	CoverTree<double, pair<double, RandomAccessIterator>, kernel_distance<RandomAccessIterator, PairwiseCallback> > ct(kd);

	{
		timed_context ct_context("Covertree construction");
		for (RandomAccessIterator iter=begin; iter!=end; ++iter)
			ct.insert(make_pair(callback(*iter,*iter),iter));
	}
	
	Neighbors neighbors;
	neighbors.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		typedef std::vector< pair<double, RandomAccessIterator> > QueryResult;
		QueryResult query = ct.knn(make_pair(callback(*iter,*iter),iter),k);
		
		LocalNeighbors local_neighbors;
		local_neighbors.reserve(k);
		for (typename QueryResult::const_iterator neighbors_iter=query.begin(); 
				neighbors_iter!=query.end(); ++neighbors_iter)
			local_neighbors.push_back(neighbors_iter->second-begin);
		neighbors.push_back(local_neighbors);
	}
	return neighbors;
}

template <class RandomAccessIterator, class PairwiseCallback>
Neighbors find_neighbors_sort(RandomAccessIterator begin, RandomAccessIterator end, 
                         PairwiseCallback callback, unsigned int k)
{
	timed_context context("Distance sorting based neighbors search");
	typedef std::pair<RandomAccessIterator, double> DistanceRecord;
	typedef std::vector<DistanceRecord> Distances;

	Neighbors neighbors;
	neighbors.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		Distances distances;
		for (RandomAccessIterator around_iter=begin; around_iter!=end; ++around_iter)
			distances.push_back(make_pair(around_iter, callback(*around_iter,*around_iter) + callback(*iter,*iter) - 2*callback(*iter,*around_iter)));

		std::sort(distances.begin(),distances.end(),
		                  distances_comparator<DistanceRecord>());

		LocalNeighbors local_neighbors;
		local_neighbors.reserve(k);
		for (typename Distances::const_iterator neighbors_iter=distances.begin()+1; 
				neighbors_iter!=distances.begin()+k+1; ++neighbors_iter)
			local_neighbors.push_back(neighbors_iter->first - begin);
		neighbors.push_back(local_neighbors);
	}
	return neighbors;
}

#endif
