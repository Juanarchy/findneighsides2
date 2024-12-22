findneighsides2 is a simple python script that solves the "find neighbor" and "find neighboring sides" problems in a given triangulation avoiding loops and conditionals.

- "Find neighbor" problem: given a list of n-tuples of vertices of cells in a tiling, what are the neighbors of each cell?
- "Find neighboring sides" problem: given a list of n-tuples of vertices of cells in a tiling whose sides are numbered, which side of each cell is shared with which side of its neighbors?

The script is really memory-intensive, so it is recommended for tilings with a small number of cells. Otherwise memory requirements get too big. For a big number of cells I reccomend findneighsides or findneighsideswave.
