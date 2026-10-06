"""
visualizador.py

Visualizador de arvores CCO (Modulos 1 e 2 do TP1).

- Leitura do VTK: PyVista (biblioteca).
- Transformacoes (translacao, rotacao, escala): matrizes 4x4 em
  coordenadas homogeneas, escritas pelo grupo e aplicadas ao modelo.
- Camera/projecao: PyVista (vistas e projecao ortografica/perspectiva).

Uso:
    python visualizador.py ../data/exemplo2d.vtk
    python visualizador.py ../data/exemplo2d.vtk --captura fig.png --modo 1
"""

import argparse

import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Matrizes de transformacao (coordenadas homogeneas 4x4)
# ---------------------------------------------------------------------------

def matriz_translacao(dx, dy, dz=0.0):
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m


def matriz_escala(s):
    return np.diag([s, s, s, 1.0])


def matriz_rotacao_z(angulo_graus):
    a = np.radians(angulo_graus)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


# ---------------------------------------------------------------------------
# Visualizador
# ---------------------------------------------------------------------------

class Visualizador:
    MODOS = ["Linhas coloridas pelo raio",
             "Tubos (espessura + cor pelo raio)",
             "Tubos (somente espessura)"]
    VISTAS = ["xy", "xz", "yz", "isometrica"]

    def __init__(self, caminho, off_screen=False):
        # --- Leitura (PyVista) ---
        malha = pv.read(caminho)
        nome = "raio" if "raio" in malha.cell_data else list(malha.cell_data.keys())[0]
        self.raios = np.asarray(malha.cell_data[nome], dtype=float)

        # LINES vem "achatado": [2, a, b, 2, a, b, ...]. Pegamos so os indices.
        if malha.lines.size != 3 * malha.n_cells:
            raise ValueError("Este visualizador espera apenas celulas de 2 pontos.")
        segmentos = malha.lines.reshape(-1, 3)[:, 1:]

        self.resumo(malha)

        # --- Malha com um segmento independente por vaso ---
        # Cada vaso ganha seus proprios 2 pontos. Assim cada um pode ter
        # o seu raio na hora de gerar o tubo (um ponto compartilhado nao
        # poderia ter tres raios ao mesmo tempo).
        n = len(segmentos)
        pontos = malha.points[segmentos].reshape(-1, 3)
        celulas = np.column_stack([np.full(n, 2),
                                   np.arange(0, 2 * n, 2),
                                   np.arange(1, 2 * n, 2)]).ravel()
        self.seg = pv.PolyData(pontos, lines=celulas)
        self.seg.cell_data["raio"] = self.raios
        self.seg.point_data["raio"] = np.repeat(self.raios, 2)

        # --- Tubos: o raio visual e proporcional ao raio do arquivo ---
        # Os raios do arquivo (0.25 a 0.67) estao em outra escala que as
        # coordenadas (~0.1), entao reescalamos: o vaso mais grosso fica
        # com raio visual de 1,5% do tamanho do dominio.
        tamanho = max(malha.bounds[1] - malha.bounds[0],
                      malha.bounds[3] - malha.bounds[2],
                      malha.bounds[5] - malha.bounds[4])
        fator = 0.015 * tamanho / self.raios.max()
        self.seg.point_data["raio_visual"] = np.repeat(self.raios, 2) * fator
        self.tubos = self.seg.tube(radius=0.001 * tamanho, scalars="raio_visual",
                                   absolute=True, n_sides=12)

        # --- Estado da aplicacao ---
        self.tamanho = tamanho
        self.centro = np.array(malha.center)
        self.matriz = np.eye(4)      # transformacao atual do modelo
        self.modo = 0
        self.vista = 0
        self.ortografica = False
        self.ator = None

        self.p = pv.Plotter(off_screen=off_screen, window_size=(1000, 800))
        self.p.set_background("white")

    # -- Modulo 1: informacoes basicas --
    def resumo(self, malha):
        print("=== Resumo do modelo CCO ===")
        print(f"Pontos   : {malha.n_points}")
        print(f"Segmentos: {malha.n_cells}")
        print(f"Raio min : {self.raios.min():.7f}")
        print(f"Raio max : {self.raios.max():.7f}")
        print(f"Limites  : {malha.bounds}")

    # -- Desenho --
    def desenhar(self):
        args = dict(name="arvore", clim=[self.raios.min(), self.raios.max()],
                    scalar_bar_args={"title": "Raio", "color": "black"})
        if self.modo == 0:
            self.ator = self.p.add_mesh(self.seg, scalars="raio", cmap="viridis",
                                        line_width=4, **args)
        elif self.modo == 1:
            self.ator = self.p.add_mesh(self.tubos, scalars="raio", cmap="viridis",
                                        **args)
        else:
            try:
                self.p.remove_scalar_bar("Raio")   # sem cor, sem legenda
            except (StopIteration, KeyError):
                pass                               # ainda nao havia barra
            self.ator = self.p.add_mesh(self.tubos, color="steelblue", name="arvore")
        self.ator.user_matrix = self.matriz    # aplica as transformacoes
        self.atualizar_texto()

    def atualizar_texto(self):
        proj = "ortografica" if self.ortografica else "perspectiva"
        self.p.add_text(f"Modo: {self.MODOS[self.modo]}\n"
                        f"Vista: {self.VISTAS[self.vista]} | Projecao: {proj}",
                        position="upper_left", font_size=10, color="black",
                        name="info")

    # -- Modulo 2: transformacoes sobre o modelo --
    def centro_atual(self):
        # Centro do modelo depois das transformacoes ja aplicadas
        return (self.matriz @ np.append(self.centro, 1.0))[:3]

    def transladar(self, dx, dy):
        passo = 0.05 * self.tamanho
        self.matriz = matriz_translacao(dx * passo, dy * passo) @ self.matriz
        self.atualizar()

    def rotacionar(self, graus):
        # Gira em torno do centro do modelo: T(c) * R * T(-c)
        c = self.centro_atual()
        m = matriz_translacao(*c) @ matriz_rotacao_z(graus) @ matriz_translacao(*-c)
        self.matriz = m @ self.matriz
        self.atualizar()

    def escalar(self, s):
        c = self.centro_atual()
        m = matriz_translacao(*c) @ matriz_escala(s) @ matriz_translacao(*-c)
        self.matriz = m @ self.matriz
        self.atualizar()

    def resetar(self):
        self.matriz = np.eye(4)
        self.atualizar()

    def atualizar(self):
        self.ator.user_matrix = self.matriz
        self.p.render()

    # -- Modulo 2: observador --
    def aplicar_vista(self):
        v = self.VISTAS[self.vista]
        {"xy": self.p.view_xy, "xz": self.p.view_xz,
         "yz": self.p.view_yz, "isometrica": self.p.view_isometric}[v]()
        self.atualizar_texto()
        self.p.render()

    def proxima_vista(self):
        self.vista = (self.vista + 1) % len(self.VISTAS)
        self.aplicar_vista()

    def alternar_projecao(self):
        self.ortografica = not self.ortografica
        if self.ortografica:
            self.p.enable_parallel_projection()
        else:
            self.p.disable_parallel_projection()
        self.atualizar_texto()
        self.p.render()

    def proximo_modo(self):
        self.modo = (self.modo + 1) % len(self.MODOS)
        self.desenhar()
        self.p.render()

    # -- Execucao --
    def executar(self, captura=None):
        self.desenhar()
        self.p.view_xy()

        # Teclas (evitamos letras que o VTK ja usa: w, s, r, f, e, q, p...)
        ev = self.p.add_key_event
        ev("m", self.proximo_modo)
        ev("v", self.proxima_vista)
        ev("o", self.alternar_projecao)
        ev("Left", lambda: self.transladar(-1, 0))
        ev("Right", lambda: self.transladar(1, 0))
        ev("Up", lambda: self.transladar(0, 1))
        ev("Down", lambda: self.transladar(0, -1))
        ev("z", lambda: self.rotacionar(10))
        ev("x", lambda: self.rotacionar(-10))
        ev("plus", lambda: self.escalar(1.1))
        ev("equal", lambda: self.escalar(1.1))
        ev("minus", lambda: self.escalar(1 / 1.1))
        ev("0", self.resetar)

        self.p.show(screenshot=captura)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", help="arquivo .vtk")
    ap.add_argument("--captura", help="salva uma imagem e fecha (sem janela)")
    ap.add_argument("--modo", type=int, default=0, help="0, 1 ou 2")
    ap.add_argument("--vista", type=int, default=0, help="0 a 3")
    a = ap.parse_args()

    vis = Visualizador(a.arquivo, off_screen=a.captura is not None)
    vis.modo, vis.vista = a.modo, a.vista
    if a.captura:
        vis.desenhar()
        vis.aplicar_vista()
    vis.executar(a.captura) if not a.captura else vis.p.show(screenshot=a.captura)