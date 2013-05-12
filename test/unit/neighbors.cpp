#include <gtest/gtest.h>

#include <tapkee/tapkee.hpp>
#include <tapkee/exceptions.hpp>

#include "callbacks.hpp"

#include <vector>
#include <algorithm>
#include <set>

#define TOLERANCE 1e-9

TEST(Neighbors,BruteDistanceNeighbors)
{
	typedef std::vector<float> Floats;
	const int N = 100;
	const int k = 10;
	ASSERT_EQ(k%2,0);

	Floats floats;
	for (int i=0;i<N;i++) 
		floats.push_back(float(i));

	float_distance_callback fdc;
	tapkee::tapkee_internal::Neighbors neighbors = 
		tapkee::tapkee_internal::find_neighbors(tapkee::Brute, floats.begin(), floats.end(), 
				tapkee::tapkee_internal::PlainDistance<Floats::iterator,float_distance_callback>(fdc), k, true);

	for (int i=0;i<N;i++)
	{
		// total number of found neighbors is k
		ASSERT_EQ(neighbors[i].size(),k);
		std::set<float> neighbors_set;
		for (int j=0;j<k;j++) 
			neighbors_set.insert(neighbors[i][j]);
		// there are no repeated values
		ASSERT_EQ(neighbors_set.size(),k);
		// the vector is not a neighbor of itself
		ASSERT_EQ(neighbors_set.find(floats[i]),neighbors_set.end());
		// check neighbors
		int k_left = std::min(i,k/2);
		int k_right = std::min(N-i-1,k/2);
		for (int j=0; j<k_left; j++) 
			ASSERT_NE(neighbors_set.find(floats[i-j-1]),neighbors_set.end());
		for (int j=0; j<k_right; j++) 
			ASSERT_NE(neighbors_set.find(floats[i+j+1]),neighbors_set.end());
	}
}

TEST(Neighbors,CoverTreeDistanceNeighbors)
{
	typedef std::vector<float> Floats;
	const int N = 100;
	const int k = 10;

	Floats floats;
	for (int i=0;i<N;i++) 
		floats.push_back(float(i));

	float_distance_callback fdc;
	tapkee::tapkee_internal::Neighbors neighbors = 
		tapkee::tapkee_internal::find_neighbors(tapkee::CoverTree, floats.begin(), floats.end(),
				tapkee::tapkee_internal::PlainDistance<Floats::iterator,float_distance_callback>(fdc), k, true);

	for (int i=0;i<N;i++)
	{
		// total number of found neighbors is k
		ASSERT_EQ(neighbors[i].size(),k);
		std::set<float> neighbors_set;
		for (int j=0;j<k;j++) 
			neighbors_set.insert(neighbors[i][j]);
		// there are no repeated values
		ASSERT_EQ(neighbors_set.size(),k);
		// the vector is not a neighbor of itself
		ASSERT_EQ(neighbors_set.find(floats[i]),neighbors_set.end());
		// check neighbors
		int k_left = std::min(i,k/2);
		int k_right = std::min(N-i-1,k/2);
		for (int j=0; j<k_left; j++) 
			ASSERT_NE(neighbors_set.find(floats[i-j-1]),neighbors_set.end());
		for (int j=0; j<k_right; j++) 
			ASSERT_NE(neighbors_set.find(floats[i+j+1]),neighbors_set.end());
	}
}

TEST(Neighbors,BruteKernelNeighbors)
{
	typedef std::vector<float> Floats;
	const int N = 100;
	const int k = 10;
	ASSERT_EQ(k%2,0);

	Floats floats;
	for (int i=0;i<N;i++) 
		floats.push_back(float(i));

	float_kernel_callback fkc;
	tapkee::tapkee_internal::Neighbors neighbors = 
		tapkee::tapkee_internal::find_neighbors(tapkee::Brute, floats.begin(), floats.end(),
				tapkee::tapkee_internal::KernelDistance<Floats::iterator,float_kernel_callback>(fkc), k, true);

	for (int i=0;i<N;i++)
	{
		// total number of found neighbors is k
		ASSERT_EQ(neighbors[i].size(),k);
		std::set<float> neighbors_set;
		for (int j=0;j<k;j++) 
			neighbors_set.insert(neighbors[i][j]);
		// there are no repeated values
		ASSERT_EQ(neighbors_set.size(),k);
		// the vector is not a neighbor of itself
		ASSERT_EQ(neighbors_set.find(floats[i]),neighbors_set.end());
		// check neighbors
		int k_left = std::min(i,k/2);
		int k_right = std::min(N-i-1,k/2);
		for (int j=0; j<k_left; j++) 
			ASSERT_NE(neighbors_set.find(floats[i-j-1]),neighbors_set.end());
		for (int j=0; j<k_right; j++) 
			ASSERT_NE(neighbors_set.find(floats[i+j+1]),neighbors_set.end());
	}
}

TEST(Neighbors,CoverTreeKernelNeighbors)
{
	typedef std::vector<float> Floats;
	const int N = 100;
	const int k = 10;

	Floats floats;
	for (int i=0;i<N;i++) 
		floats.push_back(float(i));

	float_kernel_callback fkc;
	tapkee::tapkee_internal::Neighbors neighbors = 
		tapkee::tapkee_internal::find_neighbors(tapkee::CoverTree, floats.begin(), floats.end(),
				tapkee::tapkee_internal::KernelDistance<Floats::iterator,float_kernel_callback>(fkc), k, true);

	for (int i=0;i<N;i++)
	{
		// total number of found neighbors is k
		ASSERT_EQ(neighbors[i].size(),k);
		std::set<float> neighbors_set;
		for (int j=0;j<k;j++) 
			neighbors_set.insert(neighbors[i][j]);
		// there are no repeated values
		ASSERT_EQ(neighbors_set.size(),k);
		// the vector is not a neighbor of itself
		ASSERT_EQ(neighbors_set.find(floats[i]),neighbors_set.end());
		// check neighbors
		int k_left = std::min(i,k/2);
		int k_right = std::min(N-i-1,k/2);
		for (int j=0; j<k_left; j++) 
			ASSERT_NE(neighbors_set.find(floats[i-j-1]),neighbors_set.end());
		for (int j=0; j<k_right; j++) 
			ASSERT_NE(neighbors_set.find(floats[i+j+1]),neighbors_set.end());
	}
}
