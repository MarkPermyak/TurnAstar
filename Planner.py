from SearchTools import *
from FocalSearchTools import *
import numpy as np

def euclidean_distance(i1, j1, i2, j2):
    return np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)

def octile_distance(i1, j1, i2, j2):
    dx = np.abs(i1 -i2)
    dy = np.abs(j1 - j2)
    return 1.414 * min(dx ,dy) + np.abs(dx - dy)

def chebyshev_distance(i1, j1, i2, j2):
    return max(np.abs(i1 - i2), np.abs(i2 - j2))

def manhattan_distance(i1, j1, i2, j2):
    return (max(i1,i2) - min(i1,i2)) + (max(j1,j2) - min(j1,j2))

def compute_cost(i1, j1, ori1, i2, j2, ori2):
    '''
    Computes cost of simple moves between cells
    '''
    if abs(i1 - i2) + abs(j1 - j2) == 1: #cardinal move
        return 1

    if abs(i1 - i2) + abs(j1 - j2) == 2: #diagonal move
        return 2 ** 0.5

    if abs(i1 - i2) + abs(j1 - j2) == 0 and ori1 and ori2 and ori1 != ori2: #turn move
        return (abs(ori1[0] - ori2[0]) + abs(ori1[1] - ori2[1])) / 2
    
    raise Exception("Computing cost between non neigbours or problem with orientations")


class Planner:
    def __init__(self, grid_map : Map):
        
        self.grid_map = grid_map

    def set_new_grid(self, new : Map):
        self.grid_map = new

    def astar(self,
            start_node_ : Node, goal_node_ : Node,
            turns: bool=False,
            w=1,
            heuristic_func=manhattan_distance
        ):
        
        start_node = Node(start_node_.i, start_node_.j, start_node_.ori)
        goal_node = Node(goal_node_.i, goal_node_.j, goal_node_.ori)

        goal_i, goal_j = goal_node.i, goal_node.j
        
        ast = SearchTreePQS()

        steps = 0
        nodes_created = 0

        orientation_include_start = (start_node.ori is not None)
        orientation_include_goal = (goal_node.ori is not None)

        # either all 3 are True, or all 3 are False
        assert turns == orientation_include_start == orientation_include_goal, 'Turns and start/goal orientation should exist or not exist simultaneously'

        ast.add_to_open(start_node)
        nodes_created += 1

        extracted_node = Node(-1,-1)

        while not ast.was_expanded(goal_node):
            extracted_node = ast.get_best_node_from_open()

            if extracted_node is None:
                return {'found'         : False,
                        'last_node'     : start_node,
                        'steps'         : steps,
                        'nodes_created' : nodes_created,
                        'all_nodes'     : len(ast.OPEN) + len(ast.CLOSED),
                        'OPEN'          : ast.OPEN,
                        'CLOSED'        : ast.CLOSED}

            steps += 1
            ast.add_to_closed(extracted_node)

            neighbours = self.grid_map.get_neighbors(extracted_node.i, extracted_node.j, extracted_node.ori)
            # print('Node:', extracted_node.i,extracted_node.j,extracted_node.ori, 'g=', extracted_node.g, 'f=', extracted_node.f,
            #       'neighbours:', neighbours)

            for node_tuple in neighbours:
                i, j = node_tuple[0], node_tuple[1]
                
                ori = node_tuple[2]

                # either turn or step in direction ori
                g = extracted_node.g + compute_cost(i, j, ori, extracted_node.i, extracted_node.j, extracted_node.ori)
                h = heuristic_func(i, j, goal_i, goal_j)

                if ori:
                    # turn
                    if ori != extracted_node.ori:
                        g_no_turns = extracted_node.g_no_turns
                    # step in direction ori
                    else:
                        g_no_turns = extracted_node.g_no_turns + compute_cost(i, j, ori, extracted_node.i, extracted_node.j, extracted_node.ori)
                    
                    # doesn't care what orientation in goal node
                    if i == goal_i and j == goal_j:
                        new_ori = goal_node.ori
                    else:
                        new_ori = ori

                else:
                    g_no_turns = g
                    new_ori = None

                new_node = Node(i, j, new_ori, g, h, g + w * h, g_no_turns=g_no_turns, parent=extracted_node)
                
                
                if not ast.was_expanded(new_node): 
                    ast.add_to_open(new_node)
                    nodes_created += 1

        last_node = extracted_node    
        
        CLOSED = ast.CLOSED
        OPEN = ast.OPEN    
        all_nodes = len(OPEN) + len(CLOSED)

        return {'found'         : True,
                'cost'          : last_node.g,
                'last_node'     : last_node,
                'steps'         : steps,
                'nodes_created' : nodes_created,
                'all_nodes'     : all_nodes,
                'OPEN'          : OPEN,
                'CLOSED'        : CLOSED}
    
    
    def focal_astar(self,
            start_node_ : Node, goal_node_ : Node,
            heatmaps,
            turns: bool=False,
            w=1,
            heuristic_func=manhattan_distance
        ):
        f_min = np.inf
        
        h_start = heuristic_func(start_node_.i, start_node_.j, goal_node_.i, goal_node_.j)
        start_node = Node(start_node_.i, start_node_.j, h=h_start)
        goal_node = Node(goal_node_.i, goal_node_.j)

        start_fnode = FNode(start_node, heatmaps[start_node_.i][start_node_.j])
        goal_fnode = FNode(goal_node, heatmaps[goal_node_.i][goal_node_.j])

        goal_i, goal_j = goal_node.i, goal_node.j

        ast = FocalSearchTreePQS()
        steps = 0
        nodes_created_open = 0
        nodes_created_focal = 0
        
        ast.add_to_open(start_node)
        ast.add_to_focal(start_fnode)

        nodes_created_open += 1
        nodes_created_focal += 1
        
        extracted_fnode = FNode(Node(-1,-1), 0)
        
        while (not ast.focal_is_empty() and extracted_fnode != goal_fnode):
            
            f_min = ast.get_fmin_from_open()
            
            extracted_fnode = ast.get_best_node_from_focal()

            if extracted_fnode is None:
                return {'found'         : False,
                        'last_node'     : start_node,
                        'steps'         : steps,
                        'all_nodes'     : len(ast.OPEN) + len(ast.CLOSED),
                        'OPEN'          : ast.OPEN,
                        'CLOSED'        : ast.CLOSED,
                        'FOCAL'         : ast.FOCAL,
                        'nodes_created_open' : nodes_created_open,
                        'nodes_created_focal' : nodes_created_focal}


            ast.pop_node_from_open(extracted_fnode.node)

            steps += 1
            
            ast.add_to_closed(extracted_fnode)
            
            neighbours = self.grid_map.get_neighbors(extracted_fnode.node.i, extracted_fnode.node.j)
                
            for node_tuple in neighbours:
                i, j = node_tuple[0], node_tuple[1]
                
                ori = node_tuple[2]

                g = extracted_fnode.node.g + compute_cost(i, j, ori, extracted_fnode.node.i, extracted_fnode.node.j, extracted_fnode.node.ori)
                h = heuristic_func(i, j, goal_i, goal_j)

                g_no_turns = g
                new_ori = None

                new_node = Node(i, j, new_ori, g, h, g + w * h, g_no_turns=g_no_turns, parent=extracted_fnode.node)
                new_fnode = FNode(new_node, heatmaps[i][j])

                if not ast.was_expanded(new_fnode) and new_node not in ast._open:
                    ast.add_to_open(new_node)
                    nodes_created_open += 1
                    

                if new_node.f <= w * f_min: 
                    if not ast.was_expanded(new_fnode):
                        ast.add_to_focal(new_fnode)
                        nodes_created_focal += 1
                        

                
            if not ast.open_is_empty() and f_min < ast.get_fmin_from_open():
                old = w * f_min
                new = w * ast.get_fmin_from_open()
                for n in ast._open:
                    if n.f > old and n.f <= new:
                        new_fn = FNode(n, heatmaps[n.i][n.j])
                        if not ast.was_expanded(new_fn):
                            ast.add_to_focal(new_fn)
                            nodes_created_focal += 1
                            


        last_node = extracted_fnode    
        
        CLOSED = ast.CLOSED
        FOCAL = ast.FOCAL    
        OPEN = ast.OPEN
        all_nodes = len(OPEN) + len(CLOSED)
    
        return {'found'         : True,
                'last_node'     : last_node,
                'cost'          : last_node.node.g,
                'steps'         : steps,
                'all_nodes'     : all_nodes,
                'OPEN'          : OPEN,
                'CLOSED'        : CLOSED,
                'FOCAL'         : FOCAL,
                'nodes_created_focal' : nodes_created_focal,
                'nodes_created_open' : nodes_created_open}
    

    def focal_turnastar(self,
            start_node_ : Node, goal_node_ : Node,
            heatmaps : TurnMap,
            turns: bool=True,
            w=1,
            heuristic_func=manhattan_distance
        ):
        ori_2_channel = {(1, 0)  : 0,
                    (0, 1) : 1,
                    (-1, 0)  : 2,
                    (0, -1) : 3}
        
        h_start = heuristic_func(start_node_.i, start_node_.j, goal_node_.i, goal_node_.j)
        
        start_node = Node(start_node_.i, start_node_.j, start_node_.ori, h=h_start)
        goal_node = Node(goal_node_.i, goal_node_.j, goal_node_.ori)

        start_fnode = FNode(start_node, 1)
        goal_fnode = FNode(goal_node, 1)

        goal_i, goal_j = goal_node.i, goal_node.j

        ast = FocalSearchTreePQS()
        steps = 0
        nodes_created_open = 0
        nodes_created_focal = 0
        
        orientation_include_start = (start_node.ori is not None)
        orientation_include_goal = (goal_node.ori is not None)

        # either all 3 are True, or all 3 are False
        assert turns == orientation_include_start == orientation_include_goal, 'Turns and start/goal orientation should exist or not exist simultaneously'
        
        ast.add_to_open(start_node)
        ast.add_to_focal(start_fnode)
        nodes_created_open += 1
        nodes_created_focal += 1

        extracted_fnode = FNode(Node(-1,-1), 0)

        while (not ast.focal_is_empty() and extracted_fnode != goal_fnode):
            f_min = ast.get_fmin_from_open()

            extracted_fnode = ast.get_best_node_from_focal()
            
            if extracted_fnode is None:
                return {'found'         : False,
                        'last_node'     : start_node,
                        'steps'         : steps,
                        'all_nodes'     : len(ast.OPEN) + len(ast.CLOSED),
                        'OPEN'          : ast.OPEN,
                        'CLOSED'        : ast.CLOSED,
                        'FOCAL'         : ast.FOCAL,
                        'nodes_created_open' : nodes_created_open,
                        'nodes_created_focal' : nodes_created_focal}

            ast.pop_node_from_open(extracted_fnode.node)
            
            steps += 1
            
            ast.add_to_closed(extracted_fnode)
            
            neighbours = self.grid_map.get_neighbors(extracted_fnode.node.i, extracted_fnode.node.j, extracted_fnode.node.ori)
            
            for node_tuple in neighbours:
                i, j = node_tuple[0], node_tuple[1]
                
                ori = node_tuple[2]

                assert ori is not None
                
                g = extracted_fnode.node.g + compute_cost(i, j, ori, extracted_fnode.node.i, extracted_fnode.node.j, extracted_fnode.node.ori)
                h = heuristic_func(i, j, goal_i, goal_j)
    
                # doesn't care what orientation in goal node
                if i == goal_i and j == goal_j:
                    new_ori = goal_node.ori
                else:
                    new_ori = ori

                
                if ori == [0, 0]:
                    h_focal = 1
                else:
                    channel = ori_2_channel[tuple(ori)]
                    h_focal = heatmaps._cells[channel][i][j]

                new_node = Node(i, j, new_ori, g, h, g + w * h, parent=extracted_fnode.node)
                new_fnode = FNode(new_node, h_focal)

                if not ast.was_expanded(new_fnode) and new_node not in ast._open:
                    ast.add_to_open(new_node)
                    nodes_created_open += 1
           
                if new_node.f <= w * f_min: 
                    if not ast.was_expanded(new_fnode):
                        ast.add_to_focal(new_fnode)
                        nodes_created_focal += 1

            if not ast.open_is_empty() and f_min < ast.get_fmin_from_open():
                old = w * f_min
                new = w * ast.get_fmin_from_open()
                for n in ast._open:
                    if n.f > old and n.f <= new:
                        ori = n.ori
                        if ori == [0, 0]:
                            h_focal = 1
                        else:
                            channel = ori_2_channel[tuple(ori)]
                            h_focal = heatmaps._cells[channel][n.i][n.j]

                        new_fn = FNode(n, h_focal)
                        if not ast.was_expanded(new_fn):
                            nodes_created_focal += 1
                            ast.add_to_focal(new_fn)     
            
        last_node = extracted_fnode    
        
        CLOSED = ast.CLOSED
        FOCAL = ast.FOCAL   
        OPEN = ast.OPEN 
        all_nodes = len(OPEN) + len(CLOSED)
        
        return {'found'         : True,
                'cost'          : last_node.node.g,
                'last_node'     : last_node,
                'steps'         : steps,
                'all_nodes'     : all_nodes,
                'OPEN'          : OPEN,
                'CLOSED'        : CLOSED,
                'FOCAL'         : FOCAL,
                'nodes_created_focal' : nodes_created_focal,
                'nodes_created_open' : nodes_created_open}
    
    def gbfs_astar(self,
            start_node_ : Node, goal_node_ : Node,
            heatmaps,
            turns: bool=False,
            w=1,
            heuristic_func=manhattan_distance
        ):

        start_node = Node(start_node_.i, start_node_.j)
        goal_node = Node(goal_node_.i, goal_node_.j)

        start_fnode = FNode(start_node, heatmaps[start_node_.i][start_node_.j])
        goal_fnode = FNode(goal_node, heatmaps[goal_node_.i][goal_node_.j])

        goal_i, goal_j = goal_node.i, goal_node.j

        ast = FocalSearchTreePQS()
        steps = 0
        nodes_created_open = 0
        nodes_created_focal = 0
        
        # orientation_include_start = (start_node.ori is not None)
        # orientation_include_goal = (goal_node.ori is not None)

        # # either all 3 are True, or all 3 are False
        # assert turns == orientation_include_start == orientation_include_goal, 'Turns and start/goal orientation should exist or not exist simultaneously'
        
        ast.add_to_focal(start_fnode)
        nodes_created_focal = 0
        
        extracted_node = FNode(Node(-1,-1), 0)

        # while not ast.was_expanded(goal_node):
        while (not ast.focal_is_empty() and extracted_node != goal_fnode):
            extracted_node = ast.get_best_node_from_focal()

            if extracted_node is None:
                return {'found'         : False,
                        'last_node'     : start_node,
                        'steps'         : steps,
                        'all_nodes'     : len(ast.FOCAL) + len(ast.CLOSED),
                        'CLOSED'        : ast.CLOSED,
                        'FOCAL'         : ast.FOCAL,
                        'nodes_created_focal' : nodes_created_focal}

            steps += 1
            
            if ast.was_expanded(extracted_node):
                continue 
            
            ast.add_to_closed(extracted_node)
            
            neighbours = self.grid_map.get_neighbors(extracted_node.node.i, extracted_node.node.j)
                
            for node_tuple in neighbours:
                i, j = node_tuple[0], node_tuple[1]
                
                ori = node_tuple[2]

                # either turn or step in direction ori
                g = extracted_node.node.g + compute_cost(i, j, ori, extracted_node.node.i, extracted_node.node.j, extracted_node.node.ori)
                h = heuristic_func(i, j, goal_i, goal_j)

                # if ori:
                #     # turn
                #     if ori != extracted_node.ori:
                #         g_no_turns = extracted_node.g_no_turns
                #     # step in direction ori
                #     else:
                #         g_no_turns = extracted_node.g_no_turns + compute_cost(i, j, ori, extracted_node.i, extracted_node.j, extracted_node.ori)
                    
                #     # doesn't care what orientation in goal node
                #     if i == goal_i and j == goal_j:
                #         new_ori = goal_node.ori
                #     else:
                #         new_ori = ori

                # else:
                g_no_turns = g
                new_ori = None

                new_node = Node(i, j, new_ori, g, h, g + w * h, g_no_turns=g_no_turns, parent=extracted_node.node)
                new_fnode = FNode(new_node, heatmaps[i][j])
                if not ast.was_expanded(new_fnode): 
                    ast.add_to_focal(new_fnode)
                    nodes_created_focal += 1

        last_node = extracted_node    
        
        CLOSED = ast.CLOSED
        FOCAL = ast.FOCAL   
        all_nodes = len(FOCAL) + len(CLOSED) 
        
        return {'found'         : True,
                'cost'          : last_node.node.g,
                'last_node'     : last_node,
                'steps'         : steps,
                'all_nodes'     : all_nodes,
                'CLOSED'        : CLOSED,
                'FOCAL'         : FOCAL,
                'nodes_created_focal' : nodes_created_focal}
    
    
    def gbfs_turnastar(self,
            start_node_ : Node, goal_node_ : Node,
            heatmaps : TurnMap,
            turns: bool=True,
            w=1,
            heuristic_func=manhattan_distance
        ):
        ori_2_channel = {(1, 0)  : 0,
                    (0, 1) : 1,
                    (-1, 0)  : 2,
                    (0, -1) : 3}
        
        start_node = Node(start_node_.i, start_node_.j, start_node_.ori)
        goal_node = Node(goal_node_.i, goal_node_.j, goal_node_.ori)

        start_fnode = FNode(start_node, 1)
        goal_fnode = FNode(goal_node, 1)

        goal_i, goal_j = goal_node.i, goal_node.j

        ast = FocalSearchTreePQS()
        steps = 0
        nodes_created_focal = 0
        
        orientation_include_start = (start_node.ori is not None)
        orientation_include_goal = (goal_node.ori is not None)

        # either all 3 are True, or all 3 are False
        assert turns == orientation_include_start == orientation_include_goal, 'Turns and start/goal orientation should exist or not exist simultaneously'
        
        ast.add_to_focal(start_fnode)
        nodes_created_focal += 1
        
        extracted_node = FNode(Node(-1,-1), 0)

        # while not ast.was_expanded(goal_node):
        while (not ast.focal_is_empty() and extracted_node != goal_fnode):
            extracted_node = ast.get_best_node_from_focal()

            if extracted_node is None:
                return {'found'         : False,
                        'last_node'     : start_node,
                        'steps'         : steps,
                        'all_nodes'     : len(ast.FOCAL) + len(ast.CLOSED),
                        'CLOSED'        : ast.CLOSED,
                        'FOCAL'         : ast.FOCAL,
                        'nodes_created_focal' : nodes_created_focal}

            steps += 1
            
            if ast.was_expanded(extracted_node):
                continue 
            
            ast.add_to_closed(extracted_node)
            
            neighbours = self.grid_map.get_neighbors(extracted_node.node.i, extracted_node.node.j, extracted_node.node.ori)
            
            for node_tuple in neighbours:
                i, j = node_tuple[0], node_tuple[1]
                
                ori = node_tuple[2]
                assert ori is not None
                
                # either turn or step in direction ori
                g = extracted_node.node.g + compute_cost(i, j, ori, extracted_node.node.i, extracted_node.node.j, extracted_node.node.ori)
                h = heuristic_func(i, j, goal_i, goal_j)
    
                # doesn't care what orientation in goal node
                if i == goal_i and j == goal_j:
                    new_ori = goal_node.ori
                else:
                    new_ori = ori

                new_node = Node(i, j, new_ori, g, h, g + w * h, parent=extracted_node.node)
                
                if ori == [0, 0]:
                    h_focal = 1
                else:
                    channel = ori_2_channel[tuple(ori)]
                    h_focal = heatmaps._cells[channel][i][j]

                new_fnode = FNode(new_node, h_focal)

                if not ast.was_expanded(new_fnode): 
                    
                    ast.add_to_focal(new_fnode)
                    nodes_created_focal += 1
            
        last_node = extracted_node    
        
        CLOSED = ast.CLOSED
        FOCAL = ast.FOCAL
        OPEN = ast.OPEN    
        all_nodes = len(FOCAL) + len(CLOSED)
        
        return {'found'         : True,
                'cost'          : last_node.node.g,
                'last_node'     : last_node,
                'steps'         : steps,
                'all_nodes'     : all_nodes,
                'CLOSED'        : CLOSED,
                'FOCAL'         : FOCAL,
                'nodes_created_focal' : nodes_created_focal}
    
def all_turnastar(
        grid_map : Map,
        start_node_ : Node,
        turning_cost: bool=True,
        search_tree=SearchTreePQS
    ):

    h, w = grid_map.get_size()
    cost_cells = np.full((4, h, w), -1)
    # cost_cells = np.full_like(grid_map._cells, -1)
    cost_cells[:, start_node_.i, start_node_.j] = 0

    start_node = Node(start_node_.i, start_node_.j, start_node_.ori)

    ast = search_tree()
    steps = 0
    nodes_created = 0
     
    if turning_cost is False:
        start_node.ori = None

    ast.add_to_open(Node(start_node_.i, start_node_.j, [1, 0]))
    ast.add_to_open(Node(start_node_.i, start_node_.j, [-1, 0]))
    ast.add_to_open(Node(start_node_.i, start_node_.j, [0, 1]))
    ast.add_to_open(Node(start_node_.i, start_node_.j, [0, -1]))
    
    extracted_node = Node(-1,-1)

    ori_2_map = {(0, 0)  : (0, 1, 2, 3), 
             (1, 0)  : 0,
             (0, 1)  : 1,
             (-1, 0) : 2,
             (0, -1) : 3}

    while not ast.open_is_empty():
        extracted_node = ast.get_best_node_from_open()

        if extracted_node is None:
            return cost_cells, start_node, steps, nodes_created, ast.OPEN, ast.CLOSED

        steps += 1
        ast.add_to_closed(extracted_node)

        neighbours = grid_map.get_neighbors(extracted_node.i, extracted_node.j, extracted_node.ori)
               
        for node_tuple in neighbours:
            i, j = node_tuple[0], node_tuple[1]
            
            ori = node_tuple[2]

            # either turn or step in direction ori
            g = extracted_node.g + compute_cost(i, j, ori, extracted_node.i, extracted_node.j, extracted_node.ori)
            
            
            if ori:
                # turn
                if ori != extracted_node.ori:
                    g_no_turns = extracted_node.g_no_turns
                # step in direction ori
                else:
                    g_no_turns = extracted_node.g_no_turns + compute_cost(i, j, ori, extracted_node.i, extracted_node.j, extracted_node.ori)
                
                # doesn't care what orientation in goal node
                # if i == goal_i and j == goal_j:
                #     new_ori = goal_node.ori
                # else:
                new_ori = ori
                # new_ori = None
                
            else:
                g_no_turns = g
                new_ori = None

            new_node = Node(i, j, new_ori, g, 0, g, g_no_turns=g_no_turns, parent=extracted_node)
            
            
            if not ast.was_expanded(new_node): 
                node_ori = ori_2_map[tuple(new_node.ori)]
                cost_cells[node_ori, i, j] = new_node.g
                ast.add_to_open(new_node)
                
                nodes_created += 1

    last_node = extracted_node    
    
    CLOSED = ast.CLOSED
    OPEN = ast.OPEN    
    
    return cost_cells, last_node, steps, nodes_created, OPEN, CLOSED