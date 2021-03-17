from time import time_ns
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation
from eq.parsing import EqParser, ParserError


def dist(x, y, pos) -> np.ndarray:
    return np.sqrt((x - pos[0])**2 + (y - pos[1])**2)


@np.vectorize
def clamp(min_v, v, max_v) -> np.ndarray:
    return min(max(v, min_v), max_v)


class Ball:

    def __init__(self, pos: np.ndarray, radius):
        self.pos = pos
        self.radius = radius
        self.shape = None
        self.velocity = np.array((0.0, 0.0))
        self.bounce_timeout = 0

    def set_pos(self, new_pos):
        self.pos = new_pos
        self.shape.set_center(self.pos)

    def increase_radius_by(self, off):
        self.radius += off
        self.shape.set_radius(self.radius)

    def move(self, v):
        self.pos += v
        self.shape.set_center(self.pos)

    def apply_physics(self):
        self.move(self.velocity)

    def dist(self, ball):
        return np.linalg.norm(self.pos - ball.pos)


class Viewport:

    def __init__(self, parser, size=(600, 400), initial="0.1*x^2", limits=(-10, 10, -1, 14)):
        self.parser = parser
        self.size = size
        self.limits = limits
        self.current_function = None

        self.zoom_pressed = False
        self.orig_press_zoom = None
        self.orig_release_zoom = None

        self.pan_pressed = False
        self.orig_press_pan = None
        self.orig_release_pan = None

        self.ax = None
        self.fig = None
        self.anim = None
        self.text_box = None
        self.suppress_submit = False
        self.x_data, self.y_data = None, None
        self.line = None
        self.button_pressed = False
        self.dragged_ball = None
        self.resize_ball_pressed = False

        self.last_drag_pos = np.array((0.0, 0.0))
        self.last_drag_time = 0     # in nanoseconds
        self.drag_sling_factor = 7.0
        self.bounce_timeout = 1    # in frames
        self.terminal_velocity = -1.0
        self.bounce_factor = 0.9
        self.ball_collision_factor = 0.5
        self.fps = 40
        self.gravity = 1.0      # in coord-units per second
        self.pixel_size = (800, 600)
        self.num_points = 200
        self.dpi = 100
        self.update_rate = 1000 // self.fps   # in milliseconds (30 fps)
        self.ball_radius = 0.5          # in coord-units
        self.line_color = "red"
        self.title = "Balls on functions - Simulation"
        self.resize_ball_increase = 0.05   # in coord-units per frame

        self.balls = []
        self.ball_shapes = []
        self.set_function(initial)
        self.setup(initial)
        self.setup_line_data()

    def update(self, frame):
        if self.resize_ball_pressed:
            self.dragged_ball.increase_radius_by(self.resize_ball_increase)

        g_acceleration = self.gravity / self.fps
        for i, ball in enumerate(self.balls):
            self.apply_physics(ball, g_acceleration)
            if ball.bounce_timeout > 0:
                ball.bounce_timeout -= 1
            else:
                self.check_ball_on_function(ball)
                self.check_in_borders(ball, i)
        # check ball collisions
        for b1, b2 in combinations(self.balls, 2):
            if b1.dist(b2) <= b1.radius + b2.radius:
                self.ball_collision(b1, b2)
        return self.line, *self.ball_shapes

    def apply_physics(self, ball, g_accel):
        ball.velocity[1] -= g_accel
        ball.velocity[1] = max(ball.velocity[1], self.terminal_velocity)
        ball.apply_physics()

    def ball_collision(self, b1, b2):
        # mirror velocities and apply to position
        tangent = -1 / ((b1.pos[1] - b2.pos[1]) / (b1.pos[0] - b2.pos[0]))
        alpha2 = 2 * np.arctan(tangent)
        mirror = np.array([[np.cos(alpha2), np.sin(alpha2)],
                           [np.sin(alpha2), -np.cos(alpha2)]])
        b1.velocity = np.dot(b1.velocity, mirror) * self.ball_collision_factor
        b2.velocity = np.dot(b2.velocity, mirror) * self.ball_collision_factor
        b1.apply_physics()
        b2.apply_physics()

        # set balls outside of each other
        dst = b1.dist(b2)
        if dst < b1.radius + b2.radius:
            diff = b1.radius + b2.radius - dst
            b = b1 if np.linalg.norm(b1.velocity) > np.linalg.norm(b2.velocity) else b2
            v = b1.pos - b2.pos if b is b1 else b2.pos - b1.pos
            b.move(v / np.linalg.norm(v) * diff)
            b.velocity[1] *= self.bounce_factor

    def check_in_borders(self, ball, i):
        if not (self.limits[0] - ball.radius <= ball.pos[0] <= self.limits[1] + ball.radius) \
                or ball.pos[1] + ball.radius < self.limits[2]:
            self.remove_ball(i)

    def check_ball_on_function(self, ball):
        points_per_unit = self.num_points / (self.limits[1] - self.limits[0])
        try:
            left = max(int((ball.pos[0] - ball.radius - self.limits[0]) * points_per_unit), 0)
            right = min(int((ball.pos[0] + ball.radius - self.limits[0]) * points_per_unit)+1, self.num_points-1)
        except ValueError as e:
            print(e)
            print(ball.pos)
            return
        mask = dist(self.x_data[left:right], self.y_data[left:right], ball.pos) <= ball.radius
        if mask.any():
            self.make_ball_bounce(ball, mask, left)

    def make_ball_bounce(self, ball, mask, left):
        # calculate proportion of circle for tangent where collision occured
        pos = np.mean(np.where(mask))
        x = pos / len(mask)
        # calculate tangent by derivate of circle
        index = int(left + pos)
        tangent = (self.y_data[index + 1] - self.y_data[index - 1]) / (self.x_data[index + 1] - self.x_data[index - 1])
        # mirror velocity at tangent
        alpha2 = 2 * np.arctan(tangent)
        mirror = np.array([[np.cos(alpha2), np.sin(alpha2)],
                       [np.sin(alpha2), -np.cos(alpha2)]])
        ball.velocity = np.dot(ball.velocity, mirror) * self.bounce_factor
        ball.bounce_timeout = self.bounce_timeout
        # set ball onto function
        a = np.array((self.x_data[index], self.y_data[index]))
        alpha = np.arctan(-1 / tangent)
        dv = np.array((np.cos(alpha), np.sin(alpha))) * -np.sign(tangent)
        ball.set_pos(a + ball.radius * dv)

    def set_function(self, expr):
        try:
            f = self.parser.parse(expr, name="f")
            assert len(f.var) == 1
            self.current_function = f
        except (ParserError, AssertionError):
            self.text_box.set_val("invalid")

    def setup_line_data(self):
        self.x_data = np.linspace(self.limits[0]-self.ball_radius, self.limits[1]+self.ball_radius, self.num_points)
        self.y_data = self.current_function(self.x_data)
        self.line, = self.ax.plot(self.x_data, self.y_data, color=self.line_color)

    def update_line_data(self):
        self.x_data = np.linspace(self.limits[0] - self.ball_radius, self.limits[1] + self.ball_radius, self.num_points)
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

    def submit(self, expression):
        if self.suppress_submit:
            self.suppress_submit = False
            return
        self.set_function(expression)
        self.update_line_data()

    def connect_events(self):
        self.text_box.connect_event("key_release_event", self.custom_key_release)
        self.fig.canvas.mpl_connect("button_press_event", self.new_ball)
        self.fig.canvas.mpl_connect("button_release_event", self.add_ball)
        self.fig.canvas.mpl_connect("motion_notify_event", self.drag_ball)
        self.fig.canvas.mpl_connect("draw_event", self.draw_event)

        # connect toolbar functions to disable events
        self.orig_press_zoom = self.fig.canvas.manager.toolbar.press_zoom
        self.fig.canvas.manager.toolbar.press_zoom = self.press_zoom
        self.orig_release_zoom = self.fig.canvas.manager.toolbar.release_zoom
        self.fig.canvas.manager.toolbar.release_zoom = self.release_zoom
        self.orig_press_pan = self.fig.canvas.manager.toolbar.press_pan
        self.fig.canvas.manager.toolbar.press_pan = self.press_pan
        self.orig_release_pan = self.fig.canvas.manager.toolbar.release_pan
        self.fig.canvas.manager.toolbar.release_pan = self.release_pan

    def draw_event(self, event):
        new_limits = self.ax.get_xlim() + self.ax.get_ylim()
        if new_limits == self.limits:
            return
        self.limits = new_limits
        self.update_line_data()

    def press_pan(self, event):
        self.pan_pressed = True
        return self.orig_press_pan(event)

    def release_pan(self, event):
        self.pan_pressed = False
        return self.orig_release_pan(event)

    def press_zoom(self, event):
        self.zoom_pressed = True
        return self.orig_press_zoom(event)

    def release_zoom(self, event):
        self.zoom_pressed = False
        return self.orig_release_zoom(event)

    def custom_key_release(self, event):
        # workaround of ^-key matplotlib bug:
        if event.key is None:
            self.suppress_submit = True
            self.text_box.set_val(self.text_box.text + "^")
            self.text_box.cursor_index += 1
            self.text_box._rendercursor()

    def start_animation(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=self.update_rate, blit=True)
        plt.show()

    def new_ball(self, event):
        if event.inaxes is not self.ax or self.zoom_pressed or self.pan_pressed:
            return
        if event.button != 1:
            if self.button_pressed:
                self.resize_ball_pressed = True
            return
        self.last_drag_time = time_ns()
        self.last_drag_pos = np.array((event.xdata, event.ydata))
        self.dragged_ball = Ball(self.last_drag_pos, self.ball_radius)
        self.dragged_ball.shape = plt.Circle(self.dragged_ball.pos, self.dragged_ball.radius, edgecolor="black")
        self.ax.add_artist(self.dragged_ball.shape)
        self.ball_shapes.append(self.dragged_ball.shape)
        self.button_pressed = True

    def drag_ball(self, event):
        if not self.button_pressed or not event.inaxes:
            return
        pos_now = np.array((event.xdata, event.ydata))
        time_now = time_ns()
        dt = (time_now - self.last_drag_time) / 1e6
        self.dragged_ball.velocity = self.drag_sling_factor * (pos_now - self.last_drag_pos) / dt
        self.last_drag_pos = pos_now
        self.last_drag_time = time_now
        self.dragged_ball.set_pos(self.last_drag_pos)

    def add_ball(self, event):
        if not self.button_pressed:
            return
        if event.button != 1:
            self.resize_ball_pressed = False
            return
        if not event.inaxes:
            self.dragged_ball.shape.remove()
            del self.ball_shapes[-1]
        else:
            self.dragged_ball.set_pos(np.array((event.xdata, event.ydata)))
            self.dragged_ball.velocity = clamp(self.terminal_velocity,
                                               self.dragged_ball.velocity,
                                               -self.terminal_velocity)
            self.balls.append(self.dragged_ball)
        self.resize_ball_pressed = False
        self.button_pressed = False
        self.dragged_ball = None

    def remove_ball(self, index):
        self.balls[index].shape.remove()
        del self.balls[index]
        del self.ball_shapes[index]


def launch():
    parser = EqParser()
    vp = Viewport(parser)
    vp.start_animation()


if __name__ == "__main__":
    launch()
