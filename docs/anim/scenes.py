"""Manim animations for the slides/README (https://www.manim.community).

render: manim -qm docs/anim/scenes.py AngleScene PinholeScene FilterScene TwoCamerasScene
"""
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Angle, Arrow, Axes, Create, DashedLine, Dot, FadeIn, Line,
                   MarkupText, Scene, Text, ValueTracker, VGroup, Write, always_redraw, config)

BG, TEXT, SUB = "#0A1121", "#E8EEF8", "#8CA0C0"
SKY, INDIGO, PINK, AMBER, GREEN = "#38BDF8", "#818CF8", "#F472B6", "#FBBF24", "#34D399"
config.background_color = BG
FONT = "Noto Sans"


def label(s, size=28, color=TEXT):
    return Text(s, font=FONT, font_size=size, color=color)


def math(s, size=34, color=TEXT):
    """Formula as Pango markup (<i>, <sub>, <sup>) in a serif font; no LaTeX needed."""
    return MarkupText(s, font="DejaVu Serif", font_size=size, color=color)


class AngleScene(Scene):
    """Three keypoints -> two vectors -> dot product -> clinical flexion angle."""

    def construct(self):
        title = label("Ângulo a partir de 3 keypoints", 34).to_edge(UP)
        self.play(FadeIn(title))
        B = np.array([-3.6, -2.0, 0])
        A = np.array([-6.0, 0.2, 0])
        flex = ValueTracker(20)   # flexion in degrees

        def C():
            # forearm direction = extension of AB rotated by the flexion angle
            d = (B - A) / np.linalg.norm(B - A)
            a = np.radians(flex.get_value())
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            return B + 3.0 * rot @ d

        ba = always_redraw(lambda: Arrow(B, A, buff=0, color=SKY, stroke_width=6))
        bc = always_redraw(lambda: Arrow(B, C(), buff=0, color=PINK, stroke_width=6))
        dots = always_redraw(lambda: VGroup(*[Dot(p, radius=0.11, color=TEXT) for p in (A, B, C())]))
        names = always_redraw(lambda: VGroup(
            label("A ombro", 22, SUB).next_to(A, UP),
            label("B cotovelo", 22, SUB).next_to(B, DOWN + LEFT * 0.5),
            label("C punho", 22, SUB).next_to(C(), RIGHT)))
        def interior():
            ccw = (np.degrees(np.arctan2(*(A - B)[1::-1])) - np.degrees(np.arctan2(*(C() - B)[1::-1]))) % 360
            return Angle(Line(B, C()), Line(B, A), radius=0.8, color=AMBER, stroke_width=5, other_angle=ccw > 180)

        theta = always_redraw(interior)
        ext = always_redraw(lambda: DashedLine(B, B + 1.8 * (B - A) / np.linalg.norm(B - A), color=SUB))
        self.play(Create(ba), Create(bc), FadeIn(dots), FadeIn(names))
        self.play(Create(theta), Create(ext))

        f1 = math("<i>θ</i> = arccos( <b>BA</b> · <b>BC</b> / |<b>BA</b>| |<b>BC</b>| )", 30).to_corner(UP + RIGHT).shift(DOWN * 1.0)
        f2 = math("flexão = 180° − <i>θ</i>", 30).next_to(f1, DOWN, aligned_edge=LEFT)
        val = always_redraw(lambda: VGroup(
            math(f"<i>θ</i> = {180 - flex.get_value():.0f}°", 34, AMBER),
            math(f"flexão = {flex.get_value():.0f}°", 34, GREEN),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(f2, DOWN * 1.5, aligned_edge=LEFT))
        self.play(Write(f1), Write(f2), FadeIn(val))
        self.play(flex.animate.set_value(120), run_time=3)
        self.play(flex.animate.set_value(0), run_time=2.5)
        note = label("braço esticado = 0°, a mesma convenção do fisioterapeuta", 24, SUB).to_edge(DOWN)
        self.play(FadeIn(note))
        self.play(flex.animate.set_value(75), run_time=2)
        self.wait(1.5)


class PinholeScene(Scene):
    """Pinhole back-projection: pixel u and depth Z give the 3D coordinate X."""

    def construct(self):
        title = label("2D → 3D: modelo pinhole", 34).to_edge(UP)
        self.play(FadeIn(title))
        O = np.array([-5.5, -1.2, 0])
        f = 1.6
        axis = Line(O, O + RIGHT * 10.5, color=SUB, stroke_width=2)
        plane = Line(O + RIGHT * f + DOWN * 0.7, O + RIGHT * f + UP * 3.2, color=TEXT, stroke_width=6)
        cam = Dot(O, radius=0.14, color=TEXT)
        self.play(Create(axis), Create(plane), FadeIn(cam),
                  FadeIn(label("câmera", 22, SUB).next_to(O, DOWN)),
                  FadeIn(label("plano da imagem", 22, SUB).next_to(plane, UP)))
        Z = ValueTracker(6.5)
        X = 2.6

        def P():
            return O + RIGHT * Z.get_value() + UP * X

        def U():
            return O + RIGHT * f + UP * (f * X / Z.get_value())

        ray = always_redraw(lambda: DashedLine(O, P(), color=SKY))
        pt = always_redraw(lambda: Dot(P(), radius=0.14, color=SKY))
        px = always_redraw(lambda: Dot(U(), radius=0.12, color=PINK))
        zline = always_redraw(lambda: Line(O + RIGHT * Z.get_value(), P(), color=SKY, stroke_width=2))
        zlab = always_redraw(lambda: math(f"<i>Z</i> = {Z.get_value() / 2:.2f} m", 26, SKY)
                             .next_to(O + RIGHT * Z.get_value(), DOWN))
        ulab = always_redraw(lambda: math("<i>u</i> − <i>c<sub>x</sub></i>", 26, PINK).next_to(U(), LEFT))
        flab = math("<i>f</i>", 26).next_to(O + RIGHT * f / 2, UP * 0.4)
        self.play(Create(ray), FadeIn(pt), FadeIn(px), Create(zline), FadeIn(zlab), FadeIn(ulab), FadeIn(flab))
        eq = math("<i>X</i>/<i>Z</i> = (<i>u</i> − <i>c<sub>x</sub></i>)/<i>f</i>   ⇒   "
                  "<i>X</i> = (<i>u</i> − <i>c<sub>x</sub></i>)·<i>Z</i> / <i>f</i>", 32).to_edge(DOWN).shift(UP * 0.2)
        self.play(Write(eq))
        self.play(Z.animate.set_value(9.5), run_time=2.5)
        self.play(Z.animate.set_value(4.0), run_time=2.5)
        note = label("Z vem do YOLO26-depth; f vem do campo de visão da webcam", 22, SUB).next_to(eq, UP, buff=0.25)
        self.play(FadeIn(note))
        self.wait(2)


class FilterScene(Scene):
    """Real elbow-angle signal from the webcam replayed through the filters (data: docs/anim/filtros.csv)."""

    def construct(self):
        import pandas as pd
        df = pd.read_csv(Path(__file__).with_name("filtros.csv"))
        t = df["t"].to_numpy() - df["t"].iloc[0]
        title = label("Mesmo sinal, filtros diferentes (meu cotovelo, dados reais)", 30).to_edge(UP)
        ax = Axes(x_range=[0, float(t[-1]), 2], y_range=[-10, 160, 40], x_length=12, y_length=5.2,
                  axis_config={"color": SUB, "include_tip": False}).shift(DOWN * 0.4)
        ticks = VGroup(*[label(f"{v}°", 20, SUB).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0, 40, 80, 120, 160)])
        ticks.add(label("tempo (s)", 20, SUB).next_to(ax.x_axis, DOWN, buff=0.1).align_to(ax.x_axis, RIGHT))
        self.play(FadeIn(title), Create(ax), FadeIn(ticks))
        series = [("sem filtro", "sem filtro", "#64748B", 2), ("média móvel", "media movel (10)", AMBER, 4),
                  ("exponencial", "exponencial (a=0.15)", INDIGO, 4), ("Kalman", "Kalman (vel. const.)", SKY, 4),
                  ("1€", "1 euro", PINK, 5)]
        legend = VGroup()
        curves = []
        for name, col, color, w in series:
            y = df[col].to_numpy()
            ok = np.isfinite(y)
            curves.append(ax.plot_line_graph(t[ok], y[ok], line_color=color, stroke_width=w, add_vertex_dots=False))
            legend.add(VGroup(Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=6), label(name, 20, TEXT)).arrange(RIGHT, buff=0.15))
        legend.arrange(RIGHT, buff=0.5).next_to(ax, DOWN, buff=0.35)
        self.play(FadeIn(legend))
        self.play(Create(curves[0]), run_time=3)
        for c in curves[1:]:
            self.play(Create(c), run_time=2.2)
        x0 = ax.c2p(19.3, 128)
        msg = VGroup(label("média móvel e exponencial:", 22, AMBER), label("ADM cai de 145° para 91°", 22, AMBER),
                     label("1€: 139°, 59 ms de atraso", 22, PINK)).arrange(DOWN, aligned_edge=LEFT).move_to(x0, aligned_edge=LEFT + UP)
        self.play(FadeIn(msg))
        self.wait(2.5)


class TwoCamerasScene(Scene):
    """Top view: a forearm rotating toward camera A is foreshortened for A but seen in full by B."""

    def construct(self):
        title = label("Por que duas câmeras", 34).to_edge(UP)
        self.play(FadeIn(title))
        E = np.array([0.0, 0.3, 0])
        ang = ValueTracker(0)  # 0 = forearm parallel to camera A's image plane (pointing right)

        def W():
            a = np.radians(ang.get_value())
            return E + 2.4 * np.array([np.cos(a), -np.sin(a), 0])

        camA = VGroup(Dot(E + DOWN * 3.2, radius=0.2, color=SKY), label("câmera A (frente)", 22, SKY)).arrange(RIGHT)
        camA.move_to(E + DOWN * 3.2 + RIGHT * 1.0)
        camB = VGroup(Dot(E + RIGHT * 5.0, radius=0.2, color=AMBER), label("câmera B (lado)", 22, AMBER)).arrange(DOWN)
        camB.move_to(E + RIGHT * 5.4)
        arm = always_redraw(lambda: Line(E, W(), color=PINK, stroke_width=10))
        elbow = Dot(E, radius=0.14, color=TEXT)
        # what each camera "sees": projection of the forearm on its image axis
        projA = always_redraw(lambda: Line(np.array([E[0], -2.2, 0]), np.array([W()[0], -2.2, 0]), color=SKY, stroke_width=10))
        projB = always_redraw(lambda: Line(np.array([3.9, E[1], 0]), np.array([3.9, W()[1], 0]), color=AMBER, stroke_width=10))
        txtA = always_redraw(lambda: label(f"A vê {abs(np.cos(np.radians(ang.get_value()))) * 100:.0f}% do antebraço", 22, SKY)
                             .move_to(np.array([-3.5, -2.2, 0])))
        txtB = always_redraw(lambda: label(f"B vê {abs(np.sin(np.radians(ang.get_value()))) * 100:.0f}%", 22, AMBER)
                             .move_to(np.array([3.9, 2.4, 0])))
        self.play(FadeIn(camA), FadeIn(camB), Create(arm), FadeIn(elbow))
        self.play(Create(projA), Create(projB), FadeIn(txtA), FadeIn(txtB))
        self.play(ang.animate.set_value(85), run_time=4)
        note = label("cada articulação usa a câmera que vê o membro menos encurtado", 22, TEXT).to_edge(DOWN).shift(UP * 0.1)
        self.play(FadeIn(note))
        self.play(ang.animate.set_value(20), run_time=3)
        self.wait(1.5)


class TiltScene(Scene):
    """Posture: trunk and neck tilt against the vertical, with the attention/bad thresholds."""

    def construct(self):
        title = label("Postura: inclinação em relação à vertical", 34).to_edge(UP)
        self.play(FadeIn(title))
        hip = np.array([-2.5, -2.6, 0])
        trunk, neck = ValueTracker(4), ValueTracker(8)

        def sh():
            a = np.radians(trunk.get_value())
            return hip + 3.2 * np.array([np.sin(a), np.cos(a), 0])

        def ear():
            a = np.radians(neck.get_value())
            return sh() + 1.3 * np.array([np.sin(a), np.cos(a), 0])

        v1 = DashedLine(hip, hip + UP * 4.9, color=SUB)
        v2 = always_redraw(lambda: DashedLine(sh(), sh() + UP * 1.6, color=SUB))
        t_line = always_redraw(lambda: Line(hip, sh(), color=SKY, stroke_width=9))
        n_line = always_redraw(lambda: Line(sh(), ear(), color=PINK, stroke_width=9))
        dots = always_redraw(lambda: VGroup(*[Dot(p, radius=0.12, color=TEXT) for p in (hip, sh(), ear())]))
        a1 = always_redraw(lambda: Angle(Line(hip, sh()), Line(hip, hip + UP), radius=1.1, color=SKY, other_angle=False)
                           if trunk.get_value() > 1 else VGroup())
        a2 = always_redraw(lambda: Angle(Line(sh(), ear()), Line(sh(), sh() + UP), radius=0.7, color=PINK)
                           if neck.get_value() > 1 else VGroup())

        def level(v, warn, bad):
            return ("OK", GREEN) if v < warn else ("ATENÇÃO", AMBER) if v < bad else ("RUIM", "#F87171")

        def panel():
            rows = VGroup()
            for name, v, (w, b), c in (("tronco", trunk.get_value(), (10, 20), SKY), ("pescoço", neck.get_value(), (20, 30), PINK)):
                lv, lc = level(v, w, b)
                rows.add(VGroup(label(f"{name}: {v:4.0f}°", 28, c), label(lv, 28, lc)).arrange(RIGHT, buff=0.4))
            rows.add(label("limiares: tronco 10°/20° · pescoço 20°/30°", 20, SUB))
            return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(RIGHT).shift(LEFT * 0.4)

        info = always_redraw(panel)
        self.play(Create(v1), Create(v2), Create(t_line), Create(n_line), FadeIn(dots), FadeIn(info))
        self.add(a1, a2)
        self.play(neck.animate.set_value(26), run_time=2)
        self.play(neck.animate.set_value(38), trunk.animate.set_value(16), run_time=2)
        self.play(trunk.animate.set_value(24), run_time=1.5)
        self.wait(0.8)
        self.play(neck.animate.set_value(8), trunk.animate.set_value(4), run_time=2)
        self.wait(1)


class OneEuroScene(Scene):
    """The 1-euro filter equations: cutoff frequency grows with speed."""

    def construct(self):
        title = label("Filtro 1€ (Casiez, Roussel e Vogel, CHI 2012)", 32).to_edge(UP)
        eqs = VGroup(
            math("<i>x̂<sub>i</sub></i> = <i>α x<sub>i</sub></i> + (1 − <i>α</i>) <i>x̂<sub>i−1</sub></i>", 34),
            math("<i>α</i> = 1 / (1 + <i>τ</i>/<i>T<sub>e</sub></i>),    <i>τ</i> = 1 / (2π <i>f<sub>c</sub></i>)", 34),
            math("<i>f<sub>c</sub></i> = <i>f<sub>c,min</sub></i> + <i>β</i> |<i>ẋ̂<sub>i</sub></i>|", 34, SKY),
        ).arrange(DOWN, buff=0.6, aligned_edge=LEFT).shift(LEFT * 2.2)
        notes = VGroup(label("passa-baixa exponencial", 22, SUB), label("suavização depende do corte", 22, SUB),
                       label("o corte cresce com a velocidade", 22, SKY))
        for n, e in zip(notes, eqs):
            n.next_to(e, RIGHT, buff=0.8)
        bottom = VGroup(label("parado → corte baixo → sem tremor", 24, GREEN),
                        label("rápido → corte alto → pouco atraso", 24, PINK)).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(title))
        for e, n in zip(eqs, notes):
            self.play(Write(e), FadeIn(n), run_time=1.2)
        self.play(FadeIn(bottom))
        self.wait(2)
