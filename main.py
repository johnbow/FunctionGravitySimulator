import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox


class Viewport:

    def __init__(self, size=(600, 400)):
        self.size = size

    def display(self):
        pass

    def start_animation(self):
        pass


def launch():
    vp = Viewport()


if __name__ == "__main__":
    launch()
