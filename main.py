import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from eq.parsing import EqParser


class Viewport:

    def __init__(self, parser, size=(600, 400), initial="0.1*x^2"):
        self.parser = parser
        self.size = size
        self.current_function = None
        self.set_function(initial)

    def set_function(self, expr):
        self.current_function = self.parser.parse(expr, name="f")

    def setup(self):
        fig, ax = plt.subplots()
        axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
        text_box = TextBox(axbox, "Plotten: ")
        text_box.on_submit(self.submit)
        text_box.set_val(self.current_function.eqstr)

    def submit(self, content):
        self.set_function(content)

    def start_animation(self):
        plt.show()


def launch():
    parser = EqParser()
    vp = Viewport(parser)
    vp.setup()
    vp.start_animation()


if __name__ == "__main__":
    launch()
