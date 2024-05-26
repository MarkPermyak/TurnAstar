from heapq import heappop, heappush
from .SearchTools import Node

class FNode:
    '''
    Node class represents a search node

    - node: current Node with it's parameters  
    - h_focal: additional heuristic (PPM)
    '''

    def __init__(self, node : Node, h_focal : float):
        self.node = node 
        self.h_focal = h_focal
    
    def __eq__(self, other):
        return (self.node == other.node) 
    
    def __hash__(self):
        return hash(self.node)

    def __lt__(self, other): 
        return self.h_focal > other.h_focal or\
            (self.h_focal == other.h_focal and (self.node.h < other.node.h or\
                                                (self.node.h == other.node.h and (self.node.i < other.node.i or\
                                                                                  (self.node.i == other.node.i and self.node.j < other.node.j)))))


class FocalSearchTreePQS:        
    def __init__(self):
        self._open = []          
        self._closed = set()
        self._focal = []
        
        
    def open_is_empty(self):
        return len(self._open) == 0

    def add_to_open(self, item):
        heappush(self._open, item)
        return   

    def get_best_node_from_open(self):
        if self.open_is_empty():
            return None
        
        node = heappop(self._open)

        while node in self._closed:
            if self.open_is_empty():
                return None
        
            node = heappop(self._open)
        return node
    
    def get_fmin_from_open(self):
        if self.open_is_empty():
            return None

        return self._open[0].f

    def pop_node_from_open(self, node : Node):
        self._open.remove(node)
        return 

    def focal_is_empty(self):
        return len(self._focal) == 0

    def add_to_focal(self, item):
        heappush(self._focal, item)
        return   

    def get_best_node_from_focal(self):
        if self.focal_is_empty():
            return None
        
        node = heappop(self._focal)

        while node in self._closed:
            if self.focal_is_empty():
                return None
        
            node = heappop(self._focal)
        return node

    def add_to_closed(self, item):
        self._closed.add(item)
        
    def was_expanded(self, item):
        return item in self._closed

    def was_generated(self, item):
        return item in self._open

    @property
    def OPEN(self):
        return self._open
    
    @property
    def CLOSED(self):
        return self._closed
    
    @property
    def FOCAL(self):
        return self._focal