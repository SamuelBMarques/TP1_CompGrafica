import numpy as np
import pyvista as pv

malha = pv.read("GRUPO_/data/tree2D_Nterm008.vtk")

pontos = malha.points                          # matriz (N, 3)
raios = malha.cell_data["raio"]                # vetor (M,)
segmentos = malha.lines.reshape(-1, 3)[:, 1:]  # matriz (M, 2)

print(malha.n_points, malha.n_cells)
print(raios.min(), raios.max())
print(malha.bounds)          