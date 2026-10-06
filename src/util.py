"""
util.py

Funcoes compartilhadas pelos modulos didaticos (rasterizacao e recorte).
"""

import numpy as np
import pyvista as pv


def carregar_segmentos_2d(caminho):
    """
    Le o VTK (PyVista) e devolve:
      segs  : matriz (M, 2, 2) -> para cada vaso, [[x0, y0], [x1, y1]]
      raios : vetor (M,)       -> raio de cada vaso
    So x e y sao usados (projecao 2D, descartando z).
    """
    malha = pv.read(caminho)
    nome = "raio" if "raio" in malha.cell_data else list(malha.cell_data.keys())[0]
    raios = np.asarray(malha.cell_data[nome], dtype=float)

    indices = malha.lines.reshape(-1, 3)[:, 1:]    # remove o "2" de cada celula
    segs = malha.points[indices][:, :, :2]         # (M, 2, 3) -> (M, 2, 2)
    return segs, raios


def criar_mapeamento(segs, largura, altura, margem=2):
    """
    Cria a funcao que converte coordenadas do MUNDO em coordenadas de PIXEL.

    E uma escala uniforme seguida de uma translacao:
        x_pix = (x - x_min) * s + margem + desloc_x
    A escala e a mesma em x e y, para nao deformar a arvore.
    Devolve uma funcao f(x, y) -> (x_pix, y_pix) em ponto flutuante.
    """
    pts = segs.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    util_x = largura - 1 - 2 * margem     # area util da grade
    util_y = altura - 1 - 2 * margem
    s = min(util_x / (x_max - x_min), util_y / (y_max - y_min))

    # Centraliza o desenho na grade
    desloc_x = (util_x - (x_max - x_min) * s) / 2
    desloc_y = (util_y - (y_max - y_min) * s) / 2

    def mapear(x, y):
        return ((x - x_min) * s + margem + desloc_x,
                (y - y_min) * s + margem + desloc_y)

    return mapear