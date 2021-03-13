import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from eq.parsing import EqParser, ParserError


class Viewport:

    def __init__(self, parser, size=(600, 400), initial="0.1*x^2"):
        self.parser = parser
        self.size = size
        self.current_function = None
        self.ax = None
        self.fig = None
        self.text_box = None
        self.set_function(initial)

    def set_function(self, expr):
        try:
            f = self.parser.parse(expr, name="f")
            assert len(f.var) == 1
            self.current_function = f
        except (ParserError, AssertionError):
            self.text_box.set_val("invalid expression")

    def setup(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        plt.xlim([-10, 10])
        plt.ylim([0, 10])
        axbox = self.fig.add_axes([0.15, 0.05, 0.7, 0.06])
        self.text_box = TextBox(axbox, "Plotten: ", initial=self.current_function.eqstr)
        self.text_box.on_submit(self.submit)

        # disable keyboard shortcuts
        manager, canvas = self.fig.canvas.manager, self.fig.canvas
        canvas.mpl_disconnect(manager.key_press_handler_id)

    def submit(self, expression):
        self.set_function(expression)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()

    def start_animation(self):
        plt.show()


def launch():
    parser = EqParser()
    vp = Viewport(parser)
    vp.setup()
    vp.start_animation()


if __name__ == "__main__":
    launch()
