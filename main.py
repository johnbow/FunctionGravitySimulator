import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation
from eq.parsing import EqParser, ParserError


class Ball:

    def __init__(self, pos: np.ndarray, radius):
        self.pos = pos
        self.radius = radius
        self.shape = None


class Viewport:

    def __init__(self, parser, size=(600, 400), initial="0.1*x^2", limits=(-10, 10, -1, 19)):
        self.parser = parser
        self.size = size
        self.limits = limits
        self.current_function = None
        self.ax = None
        self.fig = None
        self.anim = None
        self.text_box = None
        self.suppress_submit = False
        self.x_data, self.y_data = None, None
        self.line = None
        self.pixel_size = (800, 600)
        self.num_points = 200
        self.dpi = 100
        self.update_rate = 1000 // 30   # in milliseconds (30 fps)
        self.ball_radius = 0.5          # in coordinate units
        self.line_color = "red"
        self.title = "Balls on functions - Simulation"
        self.balls = []
        self.ball_shapes = []
        self.set_function(initial)
        self.setup(initial)
        self.setup_line_data()

    def update(self, i):
        return self.ball_shapes

    def set_function(self, expr):
        try:
            f = self.parser.parse(expr, name="f")
            assert len(f.var) == 1
            self.current_function = f
        except (ParserError, AssertionError):
            self.text_box.set_val("invalid")

    def setup_line_data(self):
        self.x_data = np.linspace(*self.limits[:2], self.num_points)
        self.y_data = self.current_function(self.x_data)
        self.line, = self.ax.plot(self.x_data, self.y_data, color=self.line_color)

    def compute_and_plot_line_data(self):
        self.x_data = np.linspace(*self.limits[:2], self.num_points)
        self.y_data = self.current_function(self.x_data)
        self.line.set_data(self.x_data, self.y_data)

    def setup(self, initial):
        self.fig, self.ax = plt.subplots(figsize=(self.pixel_size[0]//self.dpi, self.pixel_size[1]//self.dpi),
                                         num=self.title, dpi=self.dpi)
        self.fig.subplots_adjust(bottom=0.2)
        self.ax.set_aspect(1)
        plt.xlim(self.limits[:2])
        plt.ylim(self.limits[2:])
        axbox = self.fig.add_axes([0.15, 0.05, 0.7, 0.06])
        self.text_box = TextBox(axbox, "Plotten: ", initial=initial)
        self.text_box.on_submit(self.submit)
        self.connect_events()

    def connect_events(self):
        self.text_box.connect_event("key_release_event", self.custom_key_release)
        self.fig.canvas.mpl_connect("button_press_event", self.new_ball)

    def custom_key_release(self, event):
        # workaround of ^-key matplotlib bug:
        if event.key is None:
            self.suppress_submit = True
            self.text_box.set_val(self.text_box.text + "^")
            self.text_box.cursor_index += 1
            self.text_box._rendercursor()

    def submit(self, expression):
        if self.suppress_submit:
            self.suppress_submit = False
            return
        self.set_function(expression)
        self.compute_and_plot_line_data()

    def start_animation(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=self.update_rate, blit=True)
        plt.show()

    def new_ball(self, event):
        if event.inaxes is not self.ax:
            return
        self.add_ball(event.xdata, event.ydata)

    def add_ball(self, x, y):
        ball = Ball(np.array((x, y)), self.ball_radius)
        self.balls.append(ball)
        ball.shape = plt.Circle(ball.pos, ball.radius)
        self.ax.add_artist(ball.shape)
        self.ball_shapes.append(ball.shape)


def launch():
    parser = EqParser()
    vp = Viewport(parser)
    vp.start_animation()


if __name__ == "__main__":
    launch()
