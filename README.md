# TurnAstar

Repository for "Machine Learning methods for Path Planning" diploma.


# Installation

### Clone repository

```bash
git clone git@github.com:MarkPermyak/TurnAstar.git
```

### Install requirements

```bash
python install -r requirements.txt
```

# Usage
```bash
python main.py [-h] [--map_path] [--resolution] [--startx] [--starty] [--startori] [--goalx] [--goaly] [--goalori] [--algo] [--ckpt_path] [--w] [--heuristic] [--res_path]
```
## Options
**`-h, --help`** \
List of all parameters.

**`--resolution`**\
Height and width of the map. For now, must be either 64 or 128. Default = `64`.

**`--map_path`** \
Path of map. Default = `./map.npy`. Map must be numpy.array of shape (resolution x resolution) consisting of 0's for free cells and 1's for obstacles.

**`--startx`**\
X coordinate of start cell.

**`--starty`**\
Y coordinate of start cell.

**`--startori`**\
Orientation of agent at the start. Must be one of the options: `[D, R, U, L]`, which means `Down, Right, Up and Left`. Default = `D`.

**`--goalx`**\
X coordinate of goal cell.

**`--goaly`**\
Y coordinate of goal cell.

**`--goalori`**\
Orientation of agent at the goal. Must be one of the options: `[D, R, U, L]`. Default = `D`.

**`--heuristic`**\
Heuristic function for estimation. Choices are: `[euclidian, octile, chebyshev, manhattan]`. Default = `manhattan`.

**`--res_path`**\
Path for result dictionary. Default = `./Results/`.

**`--algo`**\
Algorithm for path finding. Choices are: `[TurnA*, WTurnA*, FocalTurnA*, GBFS]`. Default = `TurnA*`. 

If `--algo=WTurnA*`, weight of estimation `--w` must be specified.

If `--algo=GBFS`, path for saved model `--ckpt_path` must be specified. Trained model paths are: `./Weights/PGM64.ckpt` for maps 64x64 and `./Weights/PGM128.ckpt` for maps 128x128.

If `--algo=FocalTurnA*`, both `--w` and `--ckpt_path` must be specified.