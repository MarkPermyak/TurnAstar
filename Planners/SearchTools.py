from heapq import heappop, heappush

class Map:

    def __init__(self, allow_diag_moves=False):
        '''
        Default constructor
        '''

        self._width = 0
        self._height = 0
        self._cells = []
        self.allow_diag_moves = allow_diag_moves

    def read_from_string(self, cell_str, width, height):
        '''
        Converting a string (with '#' representing obstacles and '.' representing free cells) to a grid
        '''
        self._width = width
        self._height = height
        self._cells = [[0 for _ in range(width)] for _ in range(height)]
        cell_lines = cell_str.split("\n")
        i = 0
        j = 0
        for l in cell_lines:
            if len(l) != 0:
                j = 0
                for c in l:
                    if c == '.':
                        self._cells[i][j] = 0
                    elif c == '#':
                        self._cells[i][j] = 1
                    else:
                        continue
                    j += 1
                if j != width:
                    raise Exception("Size Error. Map width = ", j, ", but must be", width )
                
                i += 1

        if i != height:
            raise Exception("Size Error. Map height = ", i, ", but must be", height )
    
     
    def set_grid_cells(self, width, height, grid_cells):
        '''
        Initialization of map by list of cells.
        '''
        self._width = width
        self._height = height
        self._cells = grid_cells


    def in_bounds(self, i, j):
        '''
        Check if the cell is on a grid.
        '''
        return (0 <= j < self._width) and (0 <= i < self._height)
    

    def traversable(self, i, j):
        '''
        Check if the cell is not an obstacle.
        '''
        return not self._cells[i][j]

    def get_turns(self, ori):
        if self.allow_diag_moves:
            return {(0, 1)   : [[-1, 1], [1, 1]],
                    (-1, 1)  : [[-1, 0], [0, 1]],
                    (-1, 0)  : [[-1, 1], [-1, -1]],
                    (-1, -1) : [[-1, 0], [0, -1]],
                    (0, -1)  : [[-1, -1], [1, -1]],
                    (1, -1)  : [[0, -1], [1, 0]],
                    (1, 0)   : [[1, -1], [1, 1]],
                    (1, 1)   : [[1, 0], [0, 1]]}[tuple(ori)]
        
        return {(0, 1)   : [[-1, 0], [1, 0]],
                (-1, 0)  : [[0, 1], [0, -1]],
                (0, -1)  : [[-1, 0], [1, 0]],
                (1, 0)   : [[0, -1], [0, 1]]}[tuple(ori)]
        

    def get_neighbors(self, i, j, ori=None):
        '''
        Get a list of neighbouring cells as (i,j,ori) tuples.
        If orientation is valid then you can either turn left or right
        or make a move in direction of ori
        '''   
        neighbors = []

        delta      = [[0, 1], [1, 0],  [0, -1],  [-1, 0]] 
        delta_diag = [[1, 1], [1, -1], [-1, -1], [-1, 1]]


        if ori is None:
            for d in delta:
                if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                    neighbors.append((i + d[0], j + d[1], None))

            if self.allow_diag_moves:        
                
                for d in delta_diag:
                    if self.in_bounds(i + 0, j + d[1]) and self.traversable(i + 0, j + d[1]): 
                        if self.in_bounds(i + d[0], j + 0) and self.traversable(i + d[0], j + 0):
                            if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                                neighbors.append((i + d[0], j + d[1], None))
        
        elif ori == [0, 0]:
            for d in delta:
                if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                    neighbors.append((i + d[0], j + d[1], d))
        
            if self.allow_diag_moves:
                for d in delta_diag:
                    if self.in_bounds(i + 0, j + d[1]) and self.traversable(i + 0, j + d[1]): 
                        if self.in_bounds(i + d[0], j + 0) and self.traversable(i + d[0], j + 0):
                            if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1]):
                                neighbors.append((i + d[0], j + d[1], d))
            
        else:
            if self.allow_diag_moves is False:
                assert ori in delta, 'Diagonal moves are not allowed'

            turns = self.get_turns(ori)

            neighbors.append((i, j, turns[0]))
            neighbors.append((i, j, turns[1]))

            if ori in delta:
                if self.in_bounds(i+ori[0], j+ori[1]) and self.traversable(i+ori[0], j+ori[1]):
                    neighbors.append((i+ori[0], j+ori[1], ori))

            if ori in delta_diag:
                if self.in_bounds(i + 0, j + ori[1]) and self.traversable(i + 0, j + ori[1]): 
                    if self.in_bounds(i + ori[0], j + 0) and self.traversable(i + ori[0], j + 0):
                        if self.in_bounds(i + ori[0], j + ori[1]) and self.traversable(i + ori[0], j + ori[1]):
                            neighbors.append((i + ori[0], j + ori[1], ori))

        return neighbors
    
    def get_size(self):
        return (self._height, self._width)


class Node:
    '''
    Node class represents a search node

    - i, j: coordinates of corresponding grid element
    - g: g-value of the node
    - h: h-value of the node // always 0 for Dijkstra
    - F: f-value of the node = g + h
    - parent: pointer to the parent-node 
    - ori: orienation of node
    '''

    def __init__(self, i, j, ori=None, g=0, h=0, f=None, g_no_turns=0, parent=None, tie_breaking_func=None):
        self.i = i
        self.j = j
        self.ori = ori
        
        # If ori = None, then no orientation is involved on the whole map
        # If ori = [0, 0], then we can go in any direction from here (ex. start or goal)
        
        self.g = g
        self.h = h

        if f is None:
            self.f = self.g + h
        else:
            self.f = f        
        
        self.g_no_turns = g_no_turns

        self.parent = parent

    
    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j) and (self.ori == other.ori)
    
    def __hash__(self):
        if self.ori:
            ij_ori = (self.i, self.j, self.ori[0], self.ori[1])
        else:
            ij_ori = (self.i, self.j, None)
        
        return hash(ij_ori)

    def __lt__(self, other): 
        return self.f < other.f or\
                (self.f == other.f and (self.g < other.g or\
                                        (self.g == other.g and (self.i < other.i or\
                                                                (self.i == other.i and self.j < other.j)))))
    

class SearchTreePQS:
        
    def __init__(self):
        self._open = []          
        self._closed = set()
        
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


class TurnMap:

    def __init__(self, cells=None):
        assert len(cells.shape) == 3 and cells.shape[0] == 4

        self._height, self._width = cells.shape[1:3]
        self._cells = cells
        # Down : 0
        # Right : 1
        # Up : 2
        # Left : 3
    
    def get_size(self):
        return (self._height, self._width)
    
    def in_bounds(self, i, j):
        '''
        Check if the cell is on a grid.
        '''
        return (0 <= j < self._width) and (0 <= i < self._height)
    

    def traversable(self, i, j, ori):
        '''
        Check if the cell is not an obstacle.
        '''
        ori_2_direction = {(1, 0)  : 0,
                           (0, 1)  : 1,
                           (-1, 0) : 2,
                           (0, -1) : 3}
        
        directon = ori_2_direction[tuple(ori)]

        return self._cells[directon][i][j] >= 0
        

    def get_neighbors(self, i, j, ori):
        '''
        Get a list of neighbouring cells as (i,j,ori) tuples.
        If ori is [0, 0] then we can move in any direction.
        '''   
        neighbors = []

        assert ori is not None, 'TurnMap must be with orientations '
        
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0]] 
        
        if ori == [0, 0]:
            for d in delta:
                if self.in_bounds(i + d[0], j + d[1]) and self.traversable(i + d[0], j + d[1], d):
                    neighbors.append((i + d[0], j + d[1], d))
        
        else:
            if ori[0] == 0:
                ortogonal_ori = [[1, 0], [-1, 0]]
            else:
                ortogonal_ori = [[0, 1], [0, -1]]

            # either turn left or right
            if self.traversable(i, j, ortogonal_ori[0]):
                neighbors.append((i, j, ortogonal_ori[0]))

            if self.traversable(i, j, ortogonal_ori[1]):
                neighbors.append((i, j, ortogonal_ori[1]))

            # or step in direction ori 
            if self.in_bounds(i+ori[0], j+ori[1]) and self.traversable(i+ori[0], j+ori[1], ori):
                neighbors.append((i+ori[0], j+ori[1], ori))

        return neighbors