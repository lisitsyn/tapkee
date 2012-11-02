#ifndef COVERTREE_H_
#define COVERTREE_H_
#include <iosfwd>
#include <boost/unordered_map.hpp>
#include <set>
#include <boost/scoped_ptr.hpp>
#include <boost/ptr_container/ptr_list.hpp>

template<class DataType, class Point, class Distance>
class CoverTree
{
  static const int default_max_level = 0;
  int max_level;
  int min_level;

  Distance distance;

  struct Node
  {
    Node() : data(), children() 
    {
    }
    Point data;
    typedef boost::ptr_list<Node> ChildrenLevelContainer;
    typedef boost::unordered_map<int, ChildrenLevelContainer> ChildrenContainer;
    ChildrenContainer children;

    void add_child(const Point& new_data, long level)
    {
      std::auto_ptr<Node> node(new Node);
      node->data = new_data;
      children[level].push_back(node);
    }

    void dump(std::ostream& stream) const
    {
      stream << "Point " << data << "{" << std::endl;
      for(typename ChildrenContainer::const_iterator it = children.begin(); it != children.end(); ++it)
      {
        stream << "Level " << it->first << "{" << std::endl;
        for(typename ChildrenLevelContainer::const_iterator it_level = it->second.begin(); it_level != it->second.end(); ++it_level)
        {
          it_level->dump(stream);
        }
        stream << "}" << std::endl;
      }
      stream << "}" << std::endl;
    }
  };

  typedef std::vector<std::pair<DataType, const Node*> > NearestNodesStructure;
  boost::scoped_ptr<Node> root;

  DataType find_min_dist(const Point& data, const std::set<Node*>& node_set) const
  {
    DataType dist = std::numeric_limits<DataType>::max();
    for(typename std::set<Node*>::const_iterator it = node_set.begin(); it != node_set.end(); ++it)
    {
      DataType current_dist = distance(data, (*it)->data);
      if(current_dist < dist)
      {
        dist = current_dist;
      }
    }
    return dist;
  }

  void populate_set_from_node(const Point& data, Node* node, std::set<Node*>& node_set, int level) const
  {
    DataType max_dist = std::pow(static_cast<DataType>(2), level);
    if(distance(data, node->data) < max_dist)
      node_set.insert(node);
    typename Node::ChildrenContainer::iterator it = node->children.find(level);
    if(it == node->children.end())
    {
      return;
    }
    populate_set_from_list(data, node_set, it->second, max_dist);
  }

  void populate_set_from_list(const Point& data, std::set<Node*>& node_set, typename Node::ChildrenLevelContainer& level_container, DataType max_dist) const
  {
    for(typename Node::ChildrenLevelContainer::iterator it_level = level_container.begin(); it_level != level_container.end(); ++it_level)
    {
      if(distance(data, it_level->data) < max_dist)
      {
        node_set.insert(&*it_level);
      }
    }
  }

  void populate_node_structure_from_list(const Point& data, NearestNodesStructure& node_map, const typename Node::ChildrenLevelContainer& level_container, DataType max_dist) const
  {
    for(typename Node::ChildrenLevelContainer::const_iterator it_level = level_container.begin(); it_level != level_container.end(); ++it_level)
    {
      DataType dist = distance(data, it_level->data);
      if(dist < max_dist)
      {
        node_map.push_back(std::make_pair(dist, &*it_level));
      }
    }
  }

  bool try_insertion(const Point& data, const std::set<Node*>& node_set, int level)
  {
    std::set<Node*> new_node_set;
    for(typename std::set<Node*>::const_iterator it = node_set.begin(); it != node_set.end(); ++it)
    {
      populate_set_from_node(data, *it, new_node_set, level);
    }
    if(insert(data, new_node_set, level - 1))
    {
      return true;
    }
    return false;
  }

  bool insert(const Point& data, const std::set<Node*>& node_set, int level)
  {
    DataType dist = find_min_dist(data, node_set);
    if(dist > std::pow(static_cast<DataType>(2), level))
    {
      return false;
    }
    if(dist <= std::pow(static_cast<DataType>(2), level - 1))
    {
      if(try_insertion(data, node_set, level))
      {
        return true;
      }
    }
    (*node_set.begin())->add_child(data, level);
    min_level = std::min(min_level, level);
    return true;
  }

  DataType find_k_distance(NearestNodesStructure& nearest_nodes, std::size_t k) const
  {
    NearestNodesStructure new_nearest_nodes;
    std::size_t j = 0;
    for(typename NearestNodesStructure::const_iterator it = nearest_nodes.begin(); it != nearest_nodes.end() && j < k; ++it)
    {
      new_nearest_nodes.push_back(*it);
      ++j;
    }
    j = 0;
    return new_nearest_nodes.rbegin()->first;
  }

  void level_traversal(const Point& data, NearestNodesStructure& nearest_nodes, int level, std::size_t k) const
  {
    NearestNodesStructure new_nearest_nodes;
    DataType max_dist = find_k_distance(nearest_nodes, k);
    new_nearest_nodes.clear();
    for(typename NearestNodesStructure::const_iterator it = nearest_nodes.begin(); it != nearest_nodes.end(); ++it)
    {
      if(distance(data, it->second->data) < max_dist + std::pow(static_cast<DataType>(2), level))
        new_nearest_nodes.push_back(*it);
      typename Node::ChildrenContainer::const_iterator it_level = it->second->children.find(level);
      if(it_level != it->second->children.end())
      {
        populate_node_structure_from_list(data, new_nearest_nodes, it_level->second, max_dist + std::pow(static_cast<DataType>(2), level));
      }
    }
    nearest_nodes.swap(new_nearest_nodes);
  }

public:
  CoverTree(const Distance& distance)
    : max_level(default_max_level), min_level(default_max_level), distance(distance), root(NULL)
  {
  }

  void insert(const Point& data)
  {
    if(!root)
    {
      root.reset(new Node);
      root->data = data;
    }
    else
    {
      std::set<Node*> node_set;
      node_set.insert(root.get());
      if(!insert(data, node_set, max_level))
      {
        ++max_level;
        insert(data);
      };
    }
  }

  std::vector<Point> knn(const Point& data, std::size_t k) const
  {
    NearestNodesStructure nearest_nodes;
    nearest_nodes.push_back(std::make_pair(distance(data, root->data), root.get()));

    for(int i = max_level; i >= min_level; --i)
    {
      level_traversal(data, nearest_nodes, i, k);
      long partial_sort_position = std::min(k, nearest_nodes.size());
      std::partial_sort(nearest_nodes.begin(), nearest_nodes.begin() + partial_sort_position, nearest_nodes.end());
    }

    std::vector<Point> points;
    typename NearestNodesStructure::const_iterator it = nearest_nodes.begin();
    for(std::size_t i = 0; i < k && it != nearest_nodes.end(); ++i)
    {
      points.push_back(it->second->data);
      ++it;
    }
    return points;
  }

  void dump(std::ostream& stream) const
  {
    if(root)
    {
      root->dump(stream);
    }
  }
};

#endif
