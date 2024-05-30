import numpy as np
import argparse
import torch
import pickle
import os
from time import time

from Planners.Planner import *
from Planners.SearchTools import *
from Planners.FocalSearchTools import *

from Model.autoencoder import Autoencoder

def main(map_path, resolution, start, goal, algo, ckpt_path, w, heuristic, res_path):
    assert start[0] < resolution and\
        start[1] < resolution and\
        start[0] >= 0 and\
        start[1] >= 0, 'Invalid start'
    
    assert goal[0] < resolution and\
        goal[1] < resolution and\
        goal[0] >= 0 and\
        goal[1] >= 0, 'Invalid goal'
    
    t1 = time()


    map_np = np.load(map_path).astype('float32')
    
    assert len(map_np.shape) == 2 and map_np.shape == (resolution, resolution),\
        'Map should be size (resolution X resolution)'


    grid_map = Map()
    grid_map.set_grid_cells(map_np.shape[0], map_np.shape[1], map_np)

    str2ori = {'D' : [1, 0], 'R' : [0, 1], 'U' : [-1, 0], 'L' : [0, -1]}

    start_node = Node(start[0], start[1], str2ori[start[2]])
    goal_node = Node(goal[1], goal[1], str2ori[goal[2]])

    planner = Planner(grid_map)

    if algo in ['TurnA*', 'GBFS']:
        assert w is None, 'This algorithm is not weighted'
    else:
        assert w is not None, 'This algorithm needs weight'

    if algo in ['FocalTurnA*', 'GBFS']:
        assert ckpt_path is not None, 'Model path is needed'

        if resolution == 64:
            model = Autoencoder.load_from_checkpoint(ckpt_path, in_channels=3, out_channels=4)
        if resolution == 128:
            model = Autoencoder.load_from_checkpoint(ckpt_path,
                in_channels=3,
                out_channels=4,
                downsample_steps=4, 
                hidden_channels=32,
                attn_blocks=2,
                attn_heads=2,
                resolution=(128, 128))
        
        model = model.to('cpu')
        model.eval()

        map_torch = torch.from_numpy(map_np).unsqueeze(0)
        start_torch = torch.zeros_like(map_torch)
        goal_torch = torch.zeros_like(map_torch)

        start_torch[0, start[0], start[1]] = 1
        goal_torch[0, goal[0], goal[1]] = 1

        input = torch.cat([map_torch, start_torch, goal_torch], dim=0).unsqueeze(0)
        
        y = model.forward(input.cpu())[0]
        y = (y + 1) / 2

        prediction = y.detach().numpy() 
        turnmap = TurnMap(prediction)

    str2heuristic = {'euclidian' : euclidean_distance,
                     'octile'    : octile_distance, 
                     'chebyshev' : chebyshev_distance,
                     'manhattan' : manhattan_distance}
    
    h = str2heuristic[heuristic]

    if algo == 'TurnA*':
        res = planner.astar(start_node, goal_node, turns=True, heuristic_func=h)
    
    elif algo == 'WTurnA*':
        res = planner.astar(start_node, goal_node, turns=True, w=w, heuristic_func=h)

    elif algo == 'FocalTurnA*':
        res = planner.focal_turnastar(start_node, goal_node, turnmap, w=w, heuristic_func=h)

    elif algo == 'GBFS':
        res = planner.gbfs_turnastar(start_node, goal_node, turnmap, heuristic_func=h)

    t2 = time()

    os.makedirs(res_path, exist_ok=True)
    with open(res_path + 'RES_dict.pkl', 'wb+') as f:
        pickle.dump(res, f)

    print(f'Successfully found solution in {round(t2 - t1, 3)} seconds')

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, default='./map.npy')
    parser.add_argument('--resolution', type=int, choices=[64, 128], default=64)

    parser.add_argument('--startx', type=int)
    parser.add_argument('--starty', type=int)
    parser.add_argument('--startori', type=str, choices=['D', 'R', 'U', 'L'], default='D')

    parser.add_argument('--goalx', type=int)
    parser.add_argument('--goaly', type=int)
    parser.add_argument('--goalori', type=str, choices=['D', 'R', 'U', 'L'], default='D')

    parser.add_argument('--algo', type=str, choices=['TurnA*', 'WTurnA*', 'FocalTurnA*', 'GBFS'], default='TurnA*')

    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--w', type=int)
    parser.add_argument('--heuristic', type=str, choices=['euclidian', 'octile', 'chebyshev', 'manhattan'], default='manhattan')

    parser.add_argument('--res_path', type=str, default='./Results/')

    args = parser.parse_args()

    main(
        map_path=args.map_path,
        resolution=args.resolution,
        start=(args.startx, args.starty, args.startori),
        goal=(args.goalx, args.goaly, args.goalori),
        algo=args.algo,
        ckpt_path=args.ckpt_path,
        w=args.w,
        heuristic=args.heuristic,
        res_path=args.res_path
    )
