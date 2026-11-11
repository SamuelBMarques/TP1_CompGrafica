"""
app.py

Aplicacao principal do TP1 (Secao 5 - Integracao).

Estende o Visualizador (visualizador.py: Modulos 1 e 2) com:
  - rotacao do modelo tambem em X e Y (nao so em Z);
  - janela de recorte desenhada sobre o modelo (Modulo 6), com a escolha do
    algoritmo (Cohen-Sutherland ou Liang-Barsky) e a janela movel/redimensionavel.
    O recorte aparece em TODOS os modos de desenho, inclusive nos tubos com
    espessura proporcional ao raio (tecla m troca o modo).
    O recorte e feito sobre a PROJECAO em xy; o z de cada extremo recortado
    e interpolado ao longo do segmento.

Os demais modulos didaticos (Bresenham, Scanline, Antialiasing, Recorte de
poligonos) continuam sendo executados pelos seus scripts (ver README).

Uso:
    python app.py ../data/arvore.vtk
    python app.py ../data/arvore.vtk --recorte --algoritmo 1
    python app.py ../data/arvore.vtk --janela XMIN XMAX YMIN YMAX
    python app.py ../data/arvore.vtk --captura fig.png --recorte     (salva em ../figures/fig.png, sem janela)
    python app.py ../data/arvore.vtk --captura fig.png --recorte --modo 1   (tubos)
    python app.py ../data/arvore.vtk --captura v.png --modo 1 --vista xz --rotx 10 --roty 10 --projecao ortografica
"""

import argparse
import os

import numpy as np
import pyvista as pv

from recorteSegmentos import cohen_sutherland, liang_barsky
from visualizador import (Visualizador, matriz_escala, matriz_rotacao_z,
                          matriz_translacao)

# Pasta padrao das imagens: ../figures (relativa a este arquivo, em src/)
PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "figures")

ALGORITMOS = [("Cohen-Sutherland", cohen_sutherland), ("Liang-Barsky", liang_barsky)]

AJUDA = ("TECLAS\n"
         "mouse: esq = girar | roda = zoom | meio/shift+esq = mover camera\n"
         "m modo | v vista | o ortografica/perspectiva | 0 reset\n"
         "setas: transladar modelo | + / - : escala\n"
         "z x: gira em Z | a d: gira em Y | u i: gira em X\n"
         "1 liga/desliga recorte (linhas e tubos) | 2 troca algoritmo\n"
         "h j k l: move janela | , . : reduz/amplia janela")


def matriz_rotacao_x(graus):
    a = np.radians(graus)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def matriz_rotacao_y(graus):
    a = np.radians(graus)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


class Aplicacao(Visualizador):

    def __init__(self, caminho, off_screen=False, janela=None, recorte=False,
                 algoritmo=0):
        super().__init__(caminho, off_screen)
        self.recorte_on = recorte
        self.algoritmo = algoritmo
        self.janela = tuple(janela) if janela else self._janela_padrao()
        self.zjan = float(self.centro[2])      # plano da janela (z do centro do modelo)
        self.overlays = []                     # atores do recorte (seguem as transformacoes)
        self.n_visiveis = 0
        self._ids_visiveis = []

    # -- Janela de recorte -------------------------------------------------
    def _janela_padrao(self):
        """Janela que corta varios segmentos (mesma busca do Modulo 6)."""
        segs = self.seg.points[:, :2].reshape(-1, 2, 2)
        try:
            from recorteArvore import escolher_janela
            return tuple(escolher_janela(segs))
        except Exception:
            pts = segs.reshape(-1, 2)
            lo, tam = pts.min(axis=0), pts.max(axis=0) - pts.min(axis=0)
            return (lo[0] + 0.25 * tam[0], lo[0] + 0.70 * tam[0],
                    lo[1] + 0.30 * tam[1], lo[1] + 0.75 * tam[1])

    @staticmethod
    def _z_em(p0, p1, x, y):
        """z do ponto (x, y) sobre o segmento p0-p1 (interpolacao linear)."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        if abs(dx) >= abs(dy):
            t = (x - p0[0]) / dx if dx else 0.0
        else:
            t = (y - p0[1]) / dy
        return p0[2] + t * (p1[2] - p0[2])

    def _calcular_recorte(self):
        """Recorta todos os segmentos; devolve um PolyData com os trechos visiveis."""
        fn = ALGORITMOS[self.algoritmo][1]
        pts = self.seg.points.reshape(-1, 2, 3)        # um segmento por linha
        novos, ids = [], []
        for i, (p0, p1) in enumerate(pts):
            r = fn(p0[0], p0[1], p1[0], p1[1], self.janela)
            if r is None:
                continue
            novos.append((r[0], r[1], self._z_em(p0, p1, r[0], r[1])))
            novos.append((r[2], r[3], self._z_em(p0, p1, r[2], r[3])))
            ids.append(i)
        self.n_visiveis = len(ids)
        self._ids_visiveis = ids
        if not ids:
            return None
        n = len(ids)
        celulas = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2),
                                   np.arange(1, 2 * n, 2)]).ravel()
        malha = pv.PolyData(np.array(novos), lines=celulas)
        malha.cell_data["raio"] = self.raios[ids]
        return malha

    def _remover(self, nome):
        try:
            self.p.remove_actor(nome)
        except Exception:
            pass                                       # nao existia

    def _tubos_do_recorte(self, malha):
        """
        Mesmos tubos do visualizador (espessura proporcional ao raio), mas so
        para os trechos que sobraram do recorte.
        """
        fator = 0.015 * self.tamanho / self.raios.max()   # igual ao Visualizador
        r = np.repeat(self.raios[self._ids_visiveis], 2)
        malha.point_data["raio"] = r
        malha.point_data["raio_visual"] = r * fator
        return malha.tube(radius=0.001 * self.tamanho, scalars="raio_visual",
                          absolute=True, n_sides=12)

    def _recorte_overlay(self):
        """
        Desenha (ou remove) a janela e o resultado do recorte conforme recorte_on.
        Acompanha o modo de desenho:
          modo 0 (linhas)           : trechos visiveis como linhas coloridas;
          modo 1 (tubos + cor)      : trechos visiveis como tubos coloridos pelo raio;
          modo 2 (tubos, espessura) : trechos visiveis como tubos de uma cor so.
        Nos modos de tubos, a arvore inteira fica como tubos cinza translucidos
        (o que foi descartado) e a janela ganha uma area translucida.
        """
        if not self.recorte_on:
            for nome in ("janela", "janela_area", "cinza", "recorte"):
                self._remover(nome)
            self.overlays = []
            return

        tubular = self.modo in (1, 2)
        self.ator.SetVisibility(False)                 # esconde a arvore original
        x0, x1, y0, y1 = self.janela
        z = self.zjan
        cantos = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
        quadro = pv.PolyData(cantos, lines=np.array([5, 0, 1, 2, 3, 0]))
        atores = [self.p.add_mesh(quadro, color="red", line_width=3, name="janela",
                                  reset_camera=False)]

        if tubular:
            area = pv.PolyData(cantos, faces=np.array([4, 0, 1, 2, 3]))
            atores.append(self.p.add_mesh(area, color="red", opacity=0.12,
                                          name="janela_area", reset_camera=False))
            atores.append(self.p.add_mesh(self.tubos, color="lightgray", opacity=0.25,
                                          name="cinza", reset_camera=False))
        else:
            self._remover("janela_area")
            atores.append(self.p.add_mesh(self.seg, color="lightgray", line_width=1,
                                          name="cinza", reset_camera=False))

        visivel = self._calcular_recorte()
        if visivel is None:
            self._remover("recorte")
        elif not tubular:
            atores.append(self.p.add_mesh(
                visivel, scalars="raio", cmap="viridis",
                clim=[self.raios.min(), self.raios.max()], line_width=5,
                show_scalar_bar=False, name="recorte", reset_camera=False))
        else:
            tubos = self._tubos_do_recorte(visivel)
            if self.modo == 1:
                atores.append(self.p.add_mesh(
                    tubos, scalars="raio", cmap="viridis",
                    clim=[self.raios.min(), self.raios.max()],
                    show_scalar_bar=False, name="recorte", reset_camera=False))
            else:
                atores.append(self.p.add_mesh(tubos, color="steelblue", name="recorte",
                                              reset_camera=False))
        self.overlays = atores

    # -- Integracao com o visualizador base ---------------------------------
    def _aplicar_matriz(self):
        for ator in [self.ator] + self.overlays:
            if ator is not None:
                ator.user_matrix = self.matriz

    def desenhar(self):
        super().desenhar()
        self._recorte_overlay()
        self._aplicar_matriz()
        self.atualizar_texto()          # de novo: agora com a contagem do recorte

    def atualizar(self):
        self._aplicar_matriz()
        self.p.render()

    def atualizar_texto(self):
        proj = "ortografica" if self.ortografica else "perspectiva"
        texto = (f"Modo: {self.MODOS[self.modo]}\n"
                 f"Vista: {self.VISTAS[self.vista]} | Projecao: {proj}")
        if self.recorte_on:
            texto += (f"\nRecorte: {ALGORITMOS[self.algoritmo][0]} "
                      f"({self.n_visiveis}/{self.num_segmentos} segmentos visiveis)")
        self.p.add_text(texto, position="upper_left", font_size=10,
                        color="black", name="info")

    @property
    def num_segmentos(self):
        return len(self.raios)

    # -- Transformacoes extras --------------------------------------------
    def _girar(self, eixo, graus):
        """Acumula em self.matriz uma rotacao em torno do centro do modelo: T(c) * R * T(-c)."""
        r = {"x": matriz_rotacao_x, "y": matriz_rotacao_y, "z": matriz_rotacao_z}[eixo](graus)
        c = self.centro_atual()
        m = matriz_translacao(*c) @ r @ matriz_translacao(*-c)
        self.matriz = m @ self.matriz

    def rotacionar_eixo(self, eixo, graus):
        self._girar(eixo, graus)
        self.atualizar()

    # -- Controles do recorte ------------------------------------------------
    def _refazer_recorte(self):
        if not self.recorte_on:
            return
        self._recorte_overlay()
        self._aplicar_matriz()
        self.atualizar_texto()
        self.p.render()

    def alternar_recorte(self):
        self.recorte_on = not self.recorte_on
        self.desenhar()
        self.p.render()

    def proximo_algoritmo(self):
        self.algoritmo = (self.algoritmo + 1) % len(ALGORITMOS)
        self._refazer_recorte()

    def mover_janela(self, dx, dy):
        x0, x1, y0, y1 = self.janela
        px, py = dx * 0.05 * (x1 - x0), dy * 0.05 * (y1 - y0)
        self.janela = (x0 + px, x1 + px, y0 + py, y1 + py)
        self._refazer_recorte()

    def redimensionar_janela(self, f):
        x0, x1, y0, y1 = self.janela
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx, hy = f * (x1 - x0) / 2, f * (y1 - y0) / 2
        self.janela = (cx - hx, cx + hx, cy - hy, cy + hy)
        self._refazer_recorte()

    # -- Execucao ------------------------------------------------------------
    def executar(self, captura=None):
        ev = self.p.add_key_event
        ev("a", lambda: self.rotacionar_eixo("y", 10))
        ev("d", lambda: self.rotacionar_eixo("y", -10))
        ev("u", lambda: self.rotacionar_eixo("x", 10))
        ev("i", lambda: self.rotacionar_eixo("x", -10))
        ev("1", self.alternar_recorte)
        ev("2", self.proximo_algoritmo)
        ev("h", lambda: self.mover_janela(-1, 0))
        ev("l", lambda: self.mover_janela(1, 0))
        ev("j", lambda: self.mover_janela(0, -1))
        ev("k", lambda: self.mover_janela(0, 1))
        ev("comma", lambda: self.redimensionar_janela(0.9))
        ev("period", lambda: self.redimensionar_janela(1.1))
        self.p.add_text(AJUDA, position="lower_left", font_size=8,
                        color="black", name="ajuda")
        super().executar(captura)     # desenha, registra as teclas do base e abre a janela


def caminho_captura(nome):
    """
    So o nome ('vista.png')    -> salva em ../figures/vista.png
    Caminho com pasta          -> usa o caminho informado
    Sem extensao               -> acrescenta .png
    A pasta de destino e criada se nao existir.
    """
    if not os.path.splitext(nome)[1]:
        nome += ".png"
    pasta = os.path.dirname(nome)
    if pasta == "":
        pasta = PASTA_FIGURAS
        nome = os.path.join(pasta, nome)
    os.makedirs(pasta, exist_ok=True)
    return os.path.normpath(nome)


def _vista(valor):
    """Aceita o indice (0 a 3) ou o nome da vista (xy, xz, yz, isometrica)."""
    nomes = Visualizador.VISTAS
    if valor in nomes:
        return nomes.index(valor)
    if valor.isdigit() and int(valor) < len(nomes):
        return int(valor)
    raise argparse.ArgumentTypeError(f"vista invalida: {valor} (use 0-3 ou {', '.join(nomes)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", help="arquivo .vtk")
    ap.add_argument("--captura", metavar="NOME.png",
                    help="salva uma imagem e fecha (sem janela); so o nome vai para "
                         "../figures/")
    ap.add_argument("--modo", type=int, default=0, help="0, 1 ou 2")
    ap.add_argument("--vista", type=_vista, default=0,
                    help="0/xy, 1/xz, 2/yz ou 3/isometrica (vale para --captura)")
    ap.add_argument("--rotx", type=float, default=0.0,
                    help="rotacao do modelo em X, em graus (cada 10 = uma vez a tecla u)")
    ap.add_argument("--roty", type=float, default=0.0,
                    help="rotacao do modelo em Y, em graus (cada 10 = uma vez a tecla a)")
    ap.add_argument("--rotz", type=float, default=0.0,
                    help="rotacao do modelo em Z, em graus (cada 10 = uma vez a tecla z)")
    ap.add_argument("--projecao", choices=["perspectiva", "ortografica"],
                    default="perspectiva", help="projecao da camera (tecla o)")
    ap.add_argument("--recorte", action="store_true", help="inicia com o recorte ligado")
    ap.add_argument("--algoritmo", type=int, default=0, choices=[0, 1],
                    help="0 = Cohen-Sutherland, 1 = Liang-Barsky")
    ap.add_argument("--janela", type=float, nargs=4,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="janela de recorte em coordenadas do mundo")
    a = ap.parse_args()
    if a.captura:
        a.captura = caminho_captura(a.captura)

    app = Aplicacao(a.arquivo, off_screen=a.captura is not None, janela=a.janela,
                    recorte=a.recorte, algoritmo=a.algoritmo)
    app.modo, app.vista = a.modo, a.vista
    for eixo, graus in (("x", a.rotx), ("y", a.roty), ("z", a.rotz)):
        if graus:
            app._girar(eixo, graus)              # entra na matriz; desenhar() a aplica
    if a.projecao == "ortografica":
        app.alternar_projecao()
    if a.captura:
        app.desenhar()
        app.aplicar_vista()
        app.p.show(screenshot=a.captura)
        print(f"Imagem salva em {a.captura}")
    else:
        app.executar()