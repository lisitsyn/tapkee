#ifndef libedrt_neighbors_h_
#define libedrt_neighbors_h_

#include "../defines.hpp"
#include "covertree.hpp"
#include <vector>

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
	double operator()(const RandomAccessIterator& l, const RandomAccessIterator& r) const
	{
		return kc_(*l,*l) + kc_(*r,*r) - 2*kc_(*l,*r);
	}
};

template <class RandomAccessIterator, class PairwiseCallback>
Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, 
                         PairwiseCallback callback, unsigned int k)
{
	cout << "Using covertree" << endl;
	kernel_distance<RandomAccessIterator, PairwiseCallback> kd(callback);
	CoverTree<double, RandomAccessIterator, kernel_distance<RandomAccessIterator, PairwiseCallback> > ct(kd);

	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		ct.insert(iter);
	}
	
	Neighbors neighbors;
	neighbors.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		typedef std::vector< RandomAccessIterator > QueryResult;
		QueryResult query = ct.knn(iter,k);
		
		LocalNeighbors local_neighbors;
		local_neighbors.reserve(k);
		for (typename QueryResult::const_iterator neighbors_iter=query.begin(); 
				neighbors_iter!=query.end(); ++neighbors_iter)
			local_neighbors.push_back(*neighbors_iter-begin);
		neighbors.push_back(local_neighbors);
	}
	cout << "Done " << endl;
	return neighbors;

	/*
	typedef std::pair<RandomAccessIterator, double> DistanceRecord;
	typedef std::vector<DistanceRecord> Distances;
	
	cout << "K = " << k << endl;

	Neighbors neighbors;
	neighbors.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		Distances distances;
		for (RandomAccessIterator around_iter=begin; around_iter!=end; ++around_iter)
		distances.push_back(make_pair(around_iter, callback(*around_iter,*around_iter) + callback(*iter,*iter) - 2*callback(*iter,*around_iter)));
		
		std::sort(distances.begin(),distances.end(),
		                  distances_comparator<DistanceRecord>());

		for (typename Distances::iterator it=distances.begin(); it!=distances.begin()+k+1; ++it)
		{
			cout << "[" << it->second << "," << *(it->first) << "],";
		}
		cout << endl;

		LocalNeighbors local_neighbors;
		local_neighbors.reserve(k);
		for (typename Distances::const_iterator neighbors_iter=distances.begin()+1; 
				neighbors_iter!=distances.begin()+k+1; ++neighbors_iter)
			local_neighbors.push_back(neighbors_iter->first - begin);
		neighbors.push_back(local_neighbors);
	}
	return neighbors;
	*/
}

#endif
