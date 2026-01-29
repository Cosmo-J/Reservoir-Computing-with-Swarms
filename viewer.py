import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input")


def SimViewer(data):
    positions = data['positions']
    velocities = data['velocities']
    predator_positions = data['predator_positions']
    spawn_bounds = data['bounds']
    boid_count = data['boid_count']
    time_steps= data['time_steps']
    print(f'Found file!')
    
    n_frames = time_steps

    # --- Figure & scatters ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.22)  # room for slider/buttons

    ax.set_title("Boids + Predator")
    print(spawn_bounds)


    left, right = float(spawn_bounds[0]) * 10, float(spawn_bounds[1]) * 10

    ax.set_xlim(left,right)
    ax.set_ylim(left,right)
    ax.set_aspect("equal", adjustable="box")

    boids_scatter = ax.scatter([], [], s=5)   # small points
    pred_scatter  = ax.scatter([], [], s=10)  # bigger point

    boids0 = positions[0]
    vel0 = velocities[0]

    arrow_scale = 0.2  # tune if needed

    boids_quiver = ax.quiver(boids0[:, 0], boids0[:, 1],vel0[:, 0], vel0[:, 1],angles='xy', scale_units='xy', scale=1.0/arrow_scale, width=0.0025)

    # --- UI widgets ---
    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    slider = Slider(ax=ax_slider, label="t", valmin=0, valmax=n_frames - 1, valinit=0, valstep=1)

    ax_btn = fig.add_axes([0.02, 0.06, 0.10, 0.08])
    btn = Button(ax_btn, "Play")
    state = {"playing": False, "frame": 0}

    def draw_frame(i: int):
        i = int(np.clip(i, 0, n_frames - 1))
        state["frame"] = i

        boids = positions[i]
        vel = velocities[i]
        pred = predator_positions[i]

        boids_scatter.set_offsets(boids)
        pred_scatter.set_offsets(pred)

        boids_quiver.set_offsets(boids)
        boids_quiver.set_UVC(vel[:, 0], vel[:, 1])

        ax.set_xlabel(f"frame {i+1}/{n_frames}")
        return boids_scatter, pred_scatter

    def on_slider_change(val):
        state["playing"] = False
        btn.label.set_text("Play")
        draw_frame(int(val))
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    def on_button_clicked(event):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")

    btn.on_clicked(on_button_clicked)

    def tick(_):
        # This function is called repeatedly by FuncAnimation.
        if state["playing"]:
            nxt = (state["frame"] + 1) % n_frames
            # update slider without firing callbacks too aggressively
            slider.eventson = False
            slider.set_val(nxt)
            slider.eventson = True
            draw_frame(nxt)
        return boids_scatter, pred_scatter

    # Initialize and run
    draw_frame(0)
    ani = FuncAnimation(fig, tick, interval=33, blit=False)  # ~30 FPS
    plt.show()

def load_run(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "positions": z["positions"],
        "velocities": z["velocities"],
        "predator_positions": z["predator_positions"],
        "time_steps": int(z["time_steps"]),
        "boid_count": int(z["boid_count"]),
        "bounds": z["bounds"],
    }




def main():
    args = parser.parse_args()
    filename = args.input
    data = load_run(filename)
    SimViewer(data)



if __name__ == "__main__":
    main()