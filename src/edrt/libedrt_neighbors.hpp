#ifndef libedrt_neighbors_h_
#define libedrt_neighbors_h_

template <class DistanceRecord>
struct distances_comparator
{
	bool operator()(const DistanceRecord& l, const DistanceRecord& r) 
	{
		return (l.second < r.second);
	}
};

template <class RandomAccessIterator, class PairwiseCallback>
Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, 
                         PairwiseCallback callback, unsigned int k)
{
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

		/*
		for (typename Distances::iterator it=distances.begin(); it!=distances.begin()+k+1; ++it)
		{
			cout << "[" << it->second << "," << *(it->first) << "],";
		}
		cout << endl;
		*/

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
