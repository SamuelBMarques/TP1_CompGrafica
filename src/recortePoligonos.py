"""
recortePoligonos.py

Modulo 7 do TP1: recorte de poligonos pelo algoritmo de Sutherland-Hodgman,
implementado pelo grupo, contra uma janela retangular alinhada aos eixos.
O preenchimento dos poligonos usa o Scanline Fill (Modulo 4) e a arvore usa
o Bresenham (Modulo 3); o Matplotlib e usado apenas para MOSTRAR a grade.

Tres demonstracoes (todas em grade de pixels, como nos demais modulos):
  1. passo a passo: o poligono e recortado contra uma borda por vez;
  2. poligono CONCAVO em U: o resultado tem dois pedacos e o algoritmo os
     liga por trechos de borda sobrepostos sobre a borda (limitacao que motiva o
     Weiler-Atherton, discutido no fim deste arquivo);
  3. poligono formado pela raiz + folhas da arvore CCO fornecida, recortado
     por uma janela (so se um arquivo .vtk for informado).

Convencao: janela = (xmin, xmax, ymin, ymax), borda pertence a janela;
poligono = lista de vertices [(x, y), ...] sem repetir o primeiro no fim.

Uso:
    python recortePoligonos.py                               # demonstracoes 1 e 2
    python recortePoligonos.py ../data/arvore.vtk            # + demonstracao 3
    python recortePoligonos.py ../data/arvore.vtk --janela XMIN XMAX YMIN YMAX
Gera recorte_poligono_*.png em ../report/figures
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from bresenham import desenhar_grade, rasterizar_arvore
from scanlineFill import (POLIGONO_U, area_shoelace, ordenar_por_angulo,
                          ponto_dentro, scanline_fill, sobrepor_poligono)
from util import criar_mapeamento, extremos_arvore

BORDAS = ("esquerda", "direita", "abaixo", "acima")


# ---------------------------------------------------------------------------
# Sutherland-Hodgman
# ---------------------------------------------------------------------------

def dentro(p, borda, janela):
    """True se o ponto esta do lado de dentro do semiplano da borda."""
    xmin, xmax, ymin, ymax = janela
    x, y = p
    if borda == "esquerda":
        return x >= xmin
    if borda == "direita":
        return x <= xmax
    if borda == "abaixo":
        return y >= ymin
    return y <= ymax                                 # acima


def intersecao(s, e, borda, janela):
    """Ponto onde o segmento s-e cruza a reta da borda."""
    xmin, xmax, ymin, ymax = janela
    if borda in ("esquerda", "direita"):
        xb = xmin if borda == "esquerda" else xmax
        t = (xb - s[0]) / (e[0] - s[0])              # s e e em lados opostos: divisor != 0
        return (xb, s[1] + t * (e[1] - s[1]))
    yb = ymin if borda == "abaixo" else ymax
    t = (yb - s[1]) / (e[1] - s[1])
    return (s[0] + t * (e[0] - s[0]), yb)


def recortar_por_borda(poligono, borda, janela):
    """
    Um passo do algoritmo: recorta o poligono contra UMA borda.
    Percorre cada aresta (s -> e) e aplica as 4 regras:
      s dentro, e dentro : emite e
      s dentro, e fora   : emite a intersecao
      s fora,   e dentro : emite a intersecao e depois e
      s fora,   e fora   : nada
    """
    saida = []
    if not poligono:
        return saida
    s = poligono[-1]
    for e in poligono:
        if dentro(e, borda, janela):
            if not dentro(s, borda, janela):
                saida.append(intersecao(s, e, borda, janela))
            saida.append(e)
        elif dentro(s, borda, janela):
            saida.append(intersecao(s, e, borda, janela))
        s = e
    return saida


def sutherland_hodgman(poligono, janela, rastro=None):
    """
    Recorta o poligono contra as 4 bordas da janela, uma de cada vez; a saida
    de uma borda e a entrada da seguinte. Devolve a lista de vertices
    (vazia se nada aparece). Se 'rastro' for uma lista, guarda
    (borda, vertices_apos_a_borda) de cada passo (didatico).
    """
    atual = list(poligono)
    for borda in BORDAS:
        atual = recortar_por_borda(atual, borda, janela)
        if rastro is not None:
            rastro.append((borda, list(atual)))
    return limpar(atual)


def limpar(poligono, tol=1e-9):
    """Remove vertices consecutivos repetidos (surgem quando um vertice cai na borda)."""
    saida = []
    for p in poligono:
        if not saida or abs(p[0] - saida[-1][0]) > tol or abs(p[1] - saida[-1][1]) > tol:
            saida.append(p)
    if len(saida) > 1 and abs(saida[0][0] - saida[-1][0]) <= tol \
            and abs(saida[0][1] - saida[-1][1]) <= tol:
        saida.pop()
    return saida


def arestas_sobrepostas(poligono, janela, tol=1e-9):
    """
    Trechos da borda da janela percorridos nos DOIS sentidos pelo poligono
    (duas arestas colineares, de sentidos opostos, que se sobrepoem). Surgem
    quando o recorte de um poligono concavo gera varios pedacos que o
    Sutherland-Hodgman liga por trechos sobre a borda. Devolve a lista de
    segmentos ((x0, y0), (x1, y1)) da regiao sobreposta.
    """
    xmin, xmax, ymin, ymax = janela
    n = len(poligono)
    arestas = [(poligono[i], poligono[(i + 1) % n]) for i in range(n)]
    achados = []
    for i in range(n):
        for j in range(i + 1, n):
            (a, b), (c, d) = arestas[i], arestas[j]
            # fixo = coordenada constante na borda; livre = a outra
            for fixo, valores in ((1, (ymin, ymax)), (0, (xmin, xmax))):
                livre = 1 - fixo
                for v in valores:
                    if not all(abs(p[fixo] - v) <= tol for p in (a, b, c, d)):
                        continue
                    if (b[livre] - a[livre]) * (d[livre] - c[livre]) >= 0:
                        continue                      # mesmo sentido
                    lo = max(min(a[livre], b[livre]), min(c[livre], d[livre]))
                    hi = min(max(a[livre], b[livre]), max(c[livre], d[livre]))
                    if hi - lo > tol:
                        p0, p1 = [0.0, 0.0], [0.0, 0.0]
                        p0[fixo] = p1[fixo] = v
                        p0[livre], p1[livre] = lo, hi
                        achados.append((tuple(p0), tuple(p1)))
    return achados


def preencher(vertices, largura, altura):
    """Scanline Fill que aceita poligono vazio ou degenerado (< 3 vertices)."""
    if len(vertices) < 3:
        return np.zeros((altura, largura), dtype=bool)
    return scanline_fill(vertices, largura, altura)[0]


# ---------------------------------------------------------------------------
# Verificacao automatica
# ---------------------------------------------------------------------------

def na_janela(x, y, janela):
    return janela[0] <= x <= janela[1] and janela[2] <= y <= janela[3]


def verificar_pixels(poligono, janela, largura, altura):
    """
    Compara com um metodo INDEPENDENTE: um pixel pertence ao poligono
    recortado se e somente se esta dentro do poligono original (teste do raio)
    E dentro da janela.
    """
    grade = preencher(sutherland_hodgman(poligono, janela), largura, altura)
    for y in range(altura):
        for x in range(largura):
            esperado = ponto_dentro(x, y, poligono) and na_janela(x, y, janela)
            assert grade[y, x] == esperado, f"pixel ({x},{y}) em {poligono} / {janela}"


def testar_casos_fixos():
    janela = (4, 12, 3, 9)
    tri_dentro = [(5, 4), (10, 4), (7, 8)]
    assert sutherland_hodgman(tri_dentro, janela) == tri_dentro, "totalmente dentro"
    assert sutherland_hodgman([(0, 0), (2, 0), (1, 2)], janela) == [], "totalmente fora"
    cobre = [(0, 0), (20, 0), (20, 15), (0, 15)]               # janela dentro do poligono
    assert sorted(sutherland_hodgman(cobre, janela)) == sorted(
        [(4, 3), (12, 3), (12, 9), (4, 9)]), "janela dentro do poligono"
    quad = sutherland_hodgman([(2.0, 5.0), (8.0, 1.0), (14.0, 5.0), (8.0, 11.0)], janela)
    assert all(na_janela(x, y, janela) for x, y in quad) and len(quad) >= 4, "cruzando"
    return 4


def testar_aleatorios(n=300, semente=2):
    """Poligonos simples (possivelmente concavos) e janelas aleatorias."""
    random.seed(semente)
    for _ in range(n):
        k = random.randint(3, 9)
        pts = [(random.uniform(-4, 36), random.uniform(-4, 36)) for _ in range(k)]
        poli = [pts[i] for i in ordenar_por_angulo(pts)]
        x0, y0 = random.uniform(2, 20), random.uniform(2, 20)
        janela = (x0, x0 + random.uniform(3, 12), y0, y0 + random.uniform(3, 12))
        verificar_pixels(poli, janela, 32, 32)
    return n


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def contorno_janela(ax, janela, destaque=None):
    xmin, xmax, ymin, ymax = janela
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                               edgecolor="green", linestyle="--", linewidth=1.4))
    if destaque:                                     # borda aplicada neste passo
        seg = {"esquerda": ([xmin, xmin], [ymin, ymax]), "direita": ([xmax, xmax], [ymin, ymax]),
               "abaixo": ([xmin, xmax], [ymin, ymin]), "acima": ([xmin, xmax], [ymax, ymax])}
        ax.plot(*seg[destaque], color="green", linewidth=4, solid_capstyle="butt")


def painel_poligono(ax, vertices, janela, w, h, cor, titulo, destaque=None):
    grade = preencher(vertices, w, h)
    desenhar_grade(ax, w, h)
    if len(vertices) >= 2:
        sobrepor_poligono(ax, list(vertices), grade, cor)
    contorno_janela(ax, janela, destaque)
    ax.set_title(f"{titulo}\n{len(vertices)} vertices, {int(grade.sum())} pixels", fontsize=10)


POLIGONO_EXEMPLO = [(3.5, 9.5), (11.5, 2.5), (20.5, 9.5), (13.5, 18.5)]
JANELA_EXEMPLO = (7.5, 17.5, 5.5, 14.5)


def figura_passos(caminho_saida, w=24, h=20):
    """Poligono recortado contra uma borda por vez."""
    poli, janela = POLIGONO_EXEMPLO, JANELA_EXEMPLO
    rastro = []
    final = sutherland_hodgman(poli, janela, rastro)

    print(f"Exemplo passo a passo: janela x in [{janela[0]}, {janela[1]}], "
          f"y in [{janela[2]}, {janela[3]}]; poligono de {len(poli)} vertices")
    for borda, verts in rastro:
        print(f"  apos borda {borda:8s}: {len(verts)} vertices")
    print("  vertices finais: " + ", ".join(f"({x:.2f},{y:.2f})" for x, y in final))
    print(f"  area do resultado (cadarco): {area_shoelace(final):.2f}\n")

    fig, eixos = plt.subplots(2, 3, figsize=(15, 9.5))
    paineis = [(poli, "Original", None)] + [(v, f"Apos a borda {b}", b) for b, v in rastro]
    for ax, (verts, tit, dest) in zip(eixos.ravel(), paineis):
        painel_poligono(ax, verts, janela, w, h, "tab:blue", tit, dest)
    eixos[1, 2].axis("off")
    eixos[1, 2].text(0.0, 0.5, "A saida de cada borda e a\nentrada da seguinte.\n\n"
                     "Linha verde grossa: borda\naplicada no passo.\nTracejado: janela completa.",
                     fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def figura_concavo(caminho_saida, w=24, h=20):
    """Poligono em U cujo recorte tem DOIS pedacos: limitacao do Sutherland-Hodgman."""
    janela = (0.5, 23.5, 10.5, 19.5)                 # corta so os dois bracos do U
    rastro = []
    res = sutherland_hodgman(POLIGONO_U, janela, rastro)
    dup = arestas_sobrepostas(res, janela)

    grade = preencher(res, w, h)
    esperado = np.array([[ponto_dentro(x, y, POLIGONO_U) and na_janela(x, y, janela)
                          for x in range(w)] for y in range(h)])
    assert (grade == esperado).all(), "pixels do U recortado"
    print(f"Poligono em U recortado por y >= {janela[2]}: ficam 2 bracos separados,")
    print(f"  mas a saida e UMA lista de {len(res)} vertices com "
          f"{len(dup)} trecho(s) de borda percorrido(s) nos dois sentidos:")
    for a, b in dup:
        print(f"    de {a} ate {b}")
    print(f"  pixels corretos: {int(grade.sum())} (conferido com o teste do raio)\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    painel_poligono(a1, POLIGONO_U, janela, w, h, "tab:blue", "Poligono concavo (U) e janela")
    painel_poligono(a2, res, janela, w, h, "tab:blue",
                    "Sutherland-Hodgman: um unico poligono, dois pedacos")
    for (x0, y0), (x1, y1) in dup:
        a2.plot([x0, x1], [y0, y1], color="magenta", linewidth=3, linestyle=":")
    a2.plot([], [], color="magenta", linewidth=3, linestyle=":",
            label="trecho percorrido nos dois sentidos (liga os pedacos)")
    a2.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return len(dup)


def poligono_extremos(pontos, indices):
    """Raiz + folhas, ordenados pelo angulo em torno do centroide (como no Modulo 4)."""
    raiz, folhas = extremos_arvore(indices)
    verts = [tuple(pontos[i]) for i in [raiz] + folhas]
    return [verts[k] for k in ordenar_por_angulo(verts)]


def figura_arvore(pontos, indices, raios, janela, caminho_saida, n=64):
    """
    Poligono raiz+folhas da arvore recortado pela janela. O recorte e feito em
    coordenadas do mundo; como o mapeamento mundo->pixel e afim, o resultado
    mapeado e o mesmo que recortar em pixels.
    """
    mapear = criar_mapeamento(pontos[indices], n, n)
    mundo = poligono_extremos(pontos, indices)
    recortado = sutherland_hodgman(mundo, janela)
    orig_pix = [mapear(*p) for p in mundo]
    rec_pix = [mapear(*p) for p in recortado]
    jan_pix = (*mapear(janela[0], janela[2]), *mapear(janela[1], janela[3]))
    jan_pix = (jan_pix[0], jan_pix[2], jan_pix[1], jan_pix[3])      # xmin, xmax, ymin, ymax

    g0 = preencher(orig_pix, n, n)
    g1 = preencher(rec_pix, n, n)
    for y in range(n):                               # conferencia com o teste do raio
        for x in range(n):
            assert g1[y, x] == (ponto_dentro(x, y, orig_pix) and na_janela(x, y, jan_pix)), \
                f"pixel ({x},{y})"
    dup = arestas_sobrepostas(recortado, janela) if recortado else []

    print(f"Poligono raiz+folhas: {len(mundo)} vertices, {int(g0.sum())} pixels")
    print(f"Apos o recorte      : {len(recortado)} vertices, {int(g1.sum())} pixels, "
          f"{len(dup)} trecho(s) sobreposto(s)\n")

    arvore, _ = rasterizar_arvore(pontos[indices], raios, n)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1, 1, 1, 0))
    fig, eixos = plt.subplots(1, 2, figsize=(14, 7))
    for ax, verts, grade, cor, tit in (
            (eixos[0], orig_pix, g0, "orange", f"Antes: poligono raiz + folhas ({int(g0.sum())} pixels)"),
            (eixos[1], rec_pix, g1, "tab:green", f"Depois: Sutherland-Hodgman ({int(g1.sum())} pixels)")):
        img = ax.imshow(arvore, origin="lower", cmap=cmap, vmin=raios.min(),
                        vmax=raios.max(), interpolation="nearest", zorder=1.5)
        desenhar_grade(ax, n, n)
        if len(verts) >= 2:
            sobrepor_poligono(ax, verts, grade, cor)
        ax.add_patch(plt.Rectangle((jan_pix[0], jan_pix[2]), jan_pix[1] - jan_pix[0],
                                   jan_pix[3] - jan_pix[2], fill=False, edgecolor="green",
                                   linestyle="--", linewidth=1.6, zorder=3))
        ax.set_title(tit)
    fig.colorbar(img, ax=eixos[1], shrink=0.7, label="raio")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def carregar_modelo(caminho):
    """(pontos (N,2), indices (M,2), raios (M,)); cai no readVtk.py se nao houver PyVista."""
    try:
        from util import carregar_modelo_2d
        return carregar_modelo_2d(caminho)
    except ImportError:
        from readVtk import ler_vtk
        m = ler_vtk(caminho)
        raios = m.raios if m.raios is not None else np.ones(m.num_segmentos)
        return m.pontos[:, :2], m.segmentos, np.asarray(raios, dtype=float)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", nargs="?", help="arquivo .vtk (opcional; ativa a demonstracao 3)")
    ap.add_argument("--janela", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="janela de recorte da arvore, em coordenadas do mundo")
    ap.add_argument("--tamanho", type=float, default=0.5,
                    help="lado da janela automatica (fracao da caixa envolvente)")
    ap.add_argument("--saida", help="pasta de saida das figuras")
    ap.add_argument("--n", type=int, default=64, help="lado da grade da arvore (padrao 64)")
    a = ap.parse_args()

    saida = a.saida or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "report", "figures")
    os.makedirs(saida, exist_ok=True)

    # 1) Testes automaticos
    n1, n2 = testar_casos_fixos(), testar_aleatorios()
    print(f"Sutherland-Hodgman: {n1} casos fixos e {n2} poligonos aleatorios "
          "conferidos pixel a pixel com o teste par-impar.\n")

    # 2) Demonstracoes didaticas
    figura_passos(os.path.join(saida, "recorte_poligono_exemplo.png"))
    figura_concavo(os.path.join(saida, "recorte_poligono_concavo.png"))

    # 3) Arvore fornecida
    if a.arquivo:
        pontos, indices, raios = carregar_modelo(a.arquivo)
        if a.janela:
            janela = tuple(a.janela)
        else:
            from recorteArvore import escolher_janela
            janela = escolher_janela(pontos[indices], a.tamanho)
        print("Janela da arvore: x in [%.4f, %.4f], y in [%.4f, %.4f]" % janela)
        figura_arvore(pontos, indices, raios, janela,
                      os.path.join(saida, "recorte_poligono_arvore.png"), a.n)
    else:
        print("(Sem arquivo .vtk: demonstracao com a arvore ignorada.)")
    print(f"Figuras salvas em {os.path.normpath(saida)}")


# ---------------------------------------------------------------------------
# Discussao (para o relatorio): Sutherland-Hodgman x Weiler-Atherton
# ---------------------------------------------------------------------------
# Sutherland-Hodgman recorta o poligono contra UMA borda por vez, entrada e
# saida sendo listas de vertices. Funciona para qualquer poligono, mas a
# JANELA precisa ser convexa. Para poligonos convexos o resultado e sempre
# correto. Para poligonos CONCAVOS, se o resultado correto tem varios pedacos
# desconexos (como o U recortado acima), o algoritmo devolve uma unica lista
# de vertices em que os pedacos ficam ligados por arestas sobre a borda da
# janela, percorridas nos dois sentidos (trechos sobrepostos de area nula).
# Preencher essa lista com a regra par-impar da os pixels certos, mas o
# poligono nao e "limpo": nao da para contar componentes, calcular contorno,
# furos ou usa-lo como entrada de outras operacoes.
#
# Weiler-Atherton e mais adequado quando:
#   - o poligono e concavo e o recorte gera MULTIPLAS regioes: devolve um
#     poligono separado por pedaco, sem arestas fantasmas;
#   - ha casos de intersecao complexos (a janela tambem pode ser concava, ou
#     ser um poligono qualquer, e o recorte pode gerar buracos);
#   - o resultado sera usado como geometria (area, contorno, interseccao /
#     uniao / diferenca entre poligonos arbitrarios).
# Ele monta duas listas circulares (sujeito e recorte) com os pontos de
# intersecao inseridos e marcados como "entrada" ou "saida"; o contorno e
# percorrido seguindo o sujeito a cada entrada e trocando para a janela a
# cada saida, ate fechar cada regiao. Custo: estrutura de dados e tratamento
# de casos degenerados (vertices sobre a borda) bem mais complexos.