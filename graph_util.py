import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from collections import deque
from typing import Callable
import math

plt.ion()

fig = plt.figure()

data: dict[str, tuple[deque[float], deque[float]]] = {}
callbacks: dict[str, Callable[[], tuple[float, float]]] = {}
axes: dict[str, Axes] = {}
lines: dict[str, Line2D] = {}

def close():
    plt.close('all')
    
def add_plot(name: str, callback: Callable[[], tuple[float, float]]):
    if name in callbacks:
        return

    data[name] = (deque(maxlen=50), deque(maxlen=50))
    callbacks[name] = callback

    rebuild_plots()


def rebuild_plots():
    fig.clear()
    axes.clear()
    lines.clear()

    count = len(callbacks)
    if count == 0:
        return

    cols = min(2, count)
    rows = math.ceil(count / cols)

    new_axes = fig.subplots(rows, cols, squeeze=False).flatten()

    for i, name in enumerate(callbacks):
        ax = new_axes[i]

        axes[name] = ax
        ax.set_title(name)
        ax.grid(True)

        line, = ax.plot([], [], marker='o', markersize=2)
        lines[name] = line

        x, y = data[name]
        line.set_data(list(x), list(y))
        ax.relim()
        # ax.set_xlim(0, 100)    # Zakres osi X
        ax.set_ylim(-50, 50)   # Zakres osi Y

    for ax in new_axes[count:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.canvas.draw_idle()


def update_plots():
    for name, callback in callbacks.items():
        new_x, new_y = callback()

        if name not in data.keys():
            continue
        x, y = data[name]
        x.append(new_x)
        y.append(new_y)

        if name not in lines.keys():
            continue
        line = lines[name]
        line.set_data(list(x), list(y))

        if name not in axes.keys():
            continue

        ax = axes[name]
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.01)


counter = 0

def fun():
    global counter
    counter += 1
    return counter, math.sin(counter * 0.1)


def fun2():
    return 1, 1


if __name__ == "__main__":
    add_plot("Sinus", fun)
    add_plot("Constant", fun2)

    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        update_plots()