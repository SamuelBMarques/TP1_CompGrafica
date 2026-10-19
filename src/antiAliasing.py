"""
antialiasing.py

Modulo 5 do TP1: aliasing x antialiasing por SUPERSAMPLING (SSAA),
implementado pelo grupo reutilizando o Bresenham (Modulo 3) e o
Scanline Fill (Modulo 4).

Ideia do supersampling com fator k:
  1. renderiza a cena numa grade k vezes maior em cada eixo (k*n x k*n);
  2. cada pixel final corresponde a um bloco k x k de amostras;
  3. o valor do pixel e a MEDIA do bloco (filtro de caixa).
Pixels de borda ficam com a cobertura parcial (tons intermediarios),
o que suaviza a escadinha.

Uso:
    python antialiasing.py ../data/exemplo2d.vtk
Gera ../report/figures/antialiasing.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from bresenham import bresenham
from scanlineFill import ordenar_por_angulo, scanline_fill
from util import carregar_modelo_2d, criar_mapeamento, extremos_arvore

LARANJA = np.array([1.0, 0.62, 0.1])


def pincel(raio):
    """Deslocamentos (dx, dy) de um pincel circular, em pixels de alta resolucao."""
    r = int(np.ceil(raio))
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= raio * raio + 1e-9]


def desenhar_segmento(camada, p0, p1, valor, deslocamentos):
    """Bresenham + pincel: da espessura ao segmento (pinta 'valor' na camada)."""
    px = np.array(bresenham(round(p0[0]), round(p0[1]), round(p1[0]), round(p1[1])))
    n = camada.shape[0]
    for dx, dy in deslocamentos:
        x, y = px[:, 0] + dx, px[:, 1] + dy
        ok = (x >= 0) & (x < n) & (y >= 0) & (y < n)
        camada[y[ok], x[ok]] = valor


def renderizar(pontos, indices, raios, vertices_poli, n, k, espessura=1.0):
    """
    Renderiza arvore + poligono numa imagem n x n usando supersampling k x k.
    Com k = 1 nao ha antialiasing (cada pixel recebe uma unica amostra).
    A geometria e a mesma para qualquer k: so muda a quantidade de amostras.
    """
    N = n * k
    mapear = criar_mapeamento(pontos[indices], n, n)

    def para_alta(x, y):
        # pixel final (x, y) -> coordenada na grade k vezes maior
        return k * x + (k - 1) / 2, k * y + (k - 1) / 2

    # Camada 1: poligono (Scanline Fill na grade de alta resolucao)
    verts = [para_alta(*v) for v in vertices_poli]
    poli, _ = scanline_fill(verts, N, N)

    # Camada 2: vasos (Bresenham + pincel); cada amostra guarda o raio
    arvore = np.full((N, N), np.nan)
    offs = pincel(espessura * k / 2)
    for (a, b), r in zip(pontos[indices], raios):
        desenhar_segmento(arvore, para_alta(*mapear(*a)), para_alta(*mapear(*b)), r, offs)

    # Reducao: media de cada bloco k x k
    def blocos(m):
        return m.reshape(n, k, n, k)

    cob_poli = blocos(poli.astype(float)).mean(axis=(1, 3))
    tem = ~np.isnan(arvore)
    cob_arv = blocos(tem.astype(float)).mean(axis=(1, 3))
    soma = blocos(np.where(tem, arvore, 0.0)).sum(axis=(1, 3))
    cont = blocos(tem.astype(float)).sum(axis=(1, 3))
    raio_medio = np.divide(soma, cont, out=np.zeros_like(soma), where=cont > 0)

    cor_arv = plt.get_cmap("viridis")((raio_medio - raios.min()) /
                                      (raios.max() - raios.min()))[..., :3]

    # Composicao: fundo branco -> poligono (alfa 0.45) -> arvore (alfa = cobertura)
    img = np.ones((n, n, 3))
    a = 0.45 * cob_poli[..., None]
    img = img * (1 - a) + LARANJA * a
    img = img * (1 - cob_arv[..., None]) + cor_arv * cob_arv[..., None]
    return img


def vertices_dos_extremos(pontos, indices, n):
    """Vertices (em pixels) do poligono raiz + folhas, ordenados por angulo."""
    raiz, folhas = extremos_arvore(indices)
    mapear = criar_mapeamento(pontos[indices], n, n)
    pix = [mapear(*pontos[i]) for i in [raiz] + folhas]
    return [pix[i] for i in ordenar_por_angulo(pix)]


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def figura(img_sem, img_com, k, caminho_saida, zoom=(30, 60, 38, 63)):
    x0, x1, y0, y1 = zoom
    fig, eixos = plt.subplots(2, 2, figsize=(11, 11))
    titulos = ["Sem antialiasing (1 amostra por pixel)",
               f"Supersampling {k}x{k} ({k * k} amostras por pixel)"]

    for col, (img, tit) in enumerate(zip([img_sem, img_com], titulos)):
        ax = eixos[0, col]
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title(tit)

        az = eixos[1, col]
        az.imshow(img[y0:y1, x0:x1], origin="lower", interpolation="nearest",
                  extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5))
        az.set_title("Detalhe (regiao em vermelho)")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python antialiasing.py arquivo.vtk")
        sys.exit(1)

    n, k = 64, 4
    pontos, indices, raios = carregar_modelo_2d(sys.argv[1])
    poli = vertices_dos_extremos(pontos, indices, n)

    sem = renderizar(pontos, indices, raios, poli, n, 1)
    com = renderizar(pontos, indices, raios, poli, n, k)

    saida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)
    figura(sem, com, k, os.path.join(saida, "antialiasing.png"))

    # Medida objetiva: erro em relacao a uma referencia com 16x16 amostras
    ref = renderizar(pontos, indices, raios, poli, n, 16)
    print("Erro quadratico medio em relacao a referencia (16x16 amostras):")
    print(f"  sem antialiasing (k=1): {mse(sem, ref):.5f}")
    for fator in (2, 4, 8):
        img = com if fator == k else renderizar(pontos, indices, raios, poli, n, fator)
        print(f"  supersampling k={fator}:     {mse(img, ref):.5f}")

    # Quantidade de tons distintos (luminancia) na imagem
    def tons(img):
        return len(np.unique(np.round(img.mean(axis=2), 3)))
    print(f"\nTons distintos: sem antialiasing = {tons(sem)}, com k={k} = {tons(com)}")
    print(f"Figura salva em {os.path.normpath(saida)}")