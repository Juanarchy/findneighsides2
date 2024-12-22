import cupy as np

"""
This script loads a shape (3,nElems) array from a text file whose i-th row contains the index of the (possibly 1-indexed) nodes at each vertex of the i-th triangle 
of a non-ramified triangulation, and saves 3 text files containing each:
    - A shape (3,nElems) array whose i-th row contains, at position j, the index of the cell that neighbors cell i from side j.
    - A shape (3,nElems) array whose i-th row contains, at position j, the limiting side of the cell that neighbors cell i from side j.
    - A shape (3,nElems) array whose i-th row contains the index of the (now 0-indexed) nodes at each vertex of the i-th triangle.

Its implemented algorithm compares the three sides of each cell to the three sides of each other cell and positions neighbors and sides accordingly through precise
array manipulation, thus avoiding loops and conditional statements of any kind. This means that it can efficiently be parallelized or ran on GPUs through CUDA imp-
lementations of numpy such as cupy. This algorithm can be easily extended to arbitrary (even higher-dimensional) non-ramified tilings by translating "sides" into 
"faces" and all their concerning instructions.

It also has the option to load a text file containing the indices of cells to remove from the tiling, but I haven't tested what happens if one removes enough cells
to leave nodes isolated. I suspect that would be fine, but the resulting answers would assume the existence of said unused, isolated nodes.

-Juan Andrés Fuenzalida A. Contact: juan[dot]fuenzalidaa[at]sansano[dot]usm[dot]cl
"""


elements=np.loadtxt("ElemNodes.txt",dtype=int)#load nodes faces array.
elements=elements-1 #turn into 0-index if 1-indexed.
bad_cells=None

#bad_cells=np.loadtxt("bad_cells.txt",dtype=int) #Bad cells to further remove.

if bad_cells is not None:
    elements=np.delete(elements,bad_cells,0)

#Get indices to compare cells to other cells but not to themselves.
l,r=np.meshgrid(np.array(range(len(elements))),np.array(range(len(elements))))
lr=np.vstack((l,r)).T
ii=np.vstack((np.array(range(len(elements))),np.array(range(len(elements))))).T

lr=np.setdiff1d(lr,ii)
l=lr[:,0]
r=lr[:,1]

#Get the nodes for sides 0,1,2 for each cell.
side0=np.take(elements,(0,1)).sort(axis=1)
side1=np.take(elements,(1,2)).sort(axis=1)
side2=np.take(elements,(2,0)).sort(axis=1)

#Compare sides of cells (cmpXY is vector with True at position i if side X of cell l[i] has the same nodes as side Y of cell r[i]).
cmp00=(side0[l]==side0[r]).all(axis=1)
cmp01=(side0[l]==side1[r]).all(axis=1)
cmp02=(side0[l]==side2[r]).all(axis=1)
cmp11=(side1[l]==side1[r]).all(axis=1)
cmp12=(side1[l]==side2[r]).all(axis=1)
cmp22=(side2[l]==side2[r]).all(axis=1)

#Get cells sharing sides with other cells (tlXY is vector with indices of cells whose side X is equal to side Y of another cell).
#                                         (Cell tlXY[i] has side X equal to side Y of cell trXY[i], i.e. cell tlXY[i] has trXY[i] as neighbor at side X)
tl00=l[cmp00]
tl01=l[cmp01]
tl02=l[cmp02]
tl11=l[cmp11]
tl12=l[cmp12]
tl22=l[cmp22]

#Get cells sharing sides with other cells (trXY is vector with indices of cells whose side Y is equal to side X of another cell).
tr00=r[cmp00]
tr01=r[cmp01]
tr02=r[cmp02]
tr11=r[cmp11]
tr12=r[cmp12]
tr22=r[cmp22]

#Initialize arrays of neighbors and neighoring sides. Will be -1 at end of computation if cell has no neighboring cells/sides at that side.
ElemNeighs=np.tile(np.array((-1,-1,-1)),(1,len(elements)))
ElemNeighSides=np.tile(np.array((-1,-1,-1)),(1,len(elements)))

ElemNeighs[tl00,0]=tr00     #Put cell tr00[i] at position 0 for neighbors of cell tl00[i] for all i
ElemNeighSides[tl00,0]=0    #Put side 0 at position 0 for neighboring sides of cell tl00[i] for all i
ElemNeighs[tr00,0]=tl00     #Put cell tl00[i] at position 0 for neighbors of cell tr00[i] for all i
ElemNeighSides[tr00,0]=0    #Put side 0 at position 0 for neighboring sides of cell tr00[i] for all i

ElemNeighs[tl01,0]=tr01     #Put cell tr01[i] at position 0 for neighbors of cell tl01[i] for all i
ElemNeighSides[tl01,0]=1    #Put side 1 at position 0 for neighboring sides of cell tl01[i] for all i
ElemNeighs[tr01,1]=tl01     #Put cell tl01[i] at position 1 for neighbors of cell tr01[i] for all i
ElemNeighSides[tr01,1]=0    #Put side 0 at position 1 for neighboring sides of cell tr01[i] for all i

ElemNeighs[tl02,0]=tr02     #Analogous...
ElemNeighSides[tl02,0]=2    
ElemNeighs[tr02,2]=tl02     
ElemNeighSides[tr02,2]=0    

ElemNeighs[tl11,1]=tr11     
ElemNeighSides[tl11,1]=1    
ElemNeighs[tr11,1]=tl11     
ElemNeighSides[tl11,1]=1    

ElemNeighs[tl12,1]=tr12     
ElemNeighSides[tl12,1]=2    
ElemNeighs[tr12,2]=tl12     
ElemNeighSides[tr12,2]=1    

ElemNeighs[tl22,2]=tr22     
ElemNeighSides[tl22,2]=2    
ElemNeighs[tr22,2]=tl22     
ElemNeighSides[tl22,2]=2    

#save results
np.savetxt('ElemNeighsFoundGPU.txt',ElemNeighs,fmt="%1d")
np.savetxt('ElemNeighSidesFoundGPU.txt',ElemNeighSides,fmt="%1d")
np.savetxt('ElemNodesNewGPU.txt',elements,fmt='%1d')
