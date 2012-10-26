#ifndef libedrt_neighbors_h_
#define libedrt_neighbors_h_
struct covertree_point_t
{
public:

	covertree_point_t(int index, double (*measure)(int, int, const void*), const void* udata)
	{
		point_index = index;
		distance_measure = measure;
		user_data = udata;
	}

	inline bool operator==(const covertree_point_t& p) const
	{ 
		return (p.point_index==point_index); 
	}

	int point_index;
	double (*distance_measure)(int, int, const void*);
	const void* user_data;
};

struct covertree_kernel_point_t : covertree_point_t
{
	covertree_kernel_point_t(int index, double (*kernel_f)(int, int, const void*), const void* udata) :
		covertree_point_t(index,kernel_f,udata)
	{
		kii = distance_measure(point_index, point_index, user_data);
	}
	inline double distance(const covertree_kernel_point_t& p) const
	{
		return kii+p.kii-2.0*distance_measure(point_index, p.point_index, user_data); 
	}
	double kii;
};

struct covertree_distance_point_t : covertree_point_t
{
	covertree_distance_point_t(int index, double (*distance_f)(int, int, const void*), const void* udata) :
		covertree_point_t(index,distance_f,udata) { };
	inline double distance(const covertree_distance_point_t& p) const
	{
		return distance_measure(point_index, p.point_index, user_data); 
	}
};

int* neighbors_matrix(
		int N, 
		int k,
		double (*distance)(int, int, const void*),
		double (*kernel)(int, int, const void*), 
		const void* user_data);
#endif
