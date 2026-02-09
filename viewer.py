import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("input",nargs='+', help='path to .npz simulation runs')
parser.add_argument("--overlay", "-o", action="store_true", help="Overlay multiple simulations on one axis instead of besides eachother.")

DARK_MODE = False
ZOOM = 0.1
BOID_SIZE = 50*ZOOM
PRED_SIZE = 100*ZOOM
ARROW_SCALE = 1/(0.5*ZOOM)
ARROW_WIDTH = 0.015 * ZOOM


#TODO Add ability to overlay plots instead of side by side
def View1(datas:list):
    times = [d.get('time_steps') for d in datas]
    if len(set(times)) > 1:
        print(f"Times: {times}")
        sys.tracebacklimit = 0
        raise Exception('The given simulations have inequal time sets, therefore cannot be compared.')
    else:
        time_steps = times[0]
        

    if DARK_MODE: plt.style.use('dark_background')

    


    fig, axs = plt.subplots(1, len(datas),figsize=(5 * len(datas), 5))
    axs = np.atleast_1d(axs)
    
    artists = [GetArtist(ax,data) for ax, data in zip(axs,datas)]
    
    state = {"playing": False, "frame": 0, "speed":33}
    plt.subplots_adjust(bottom=0.5)
    
    ax_time = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    playback_slider = Slider(ax=ax_time, label="t", valmin=0, valmax=time_steps - 1, valinit=0, valstep=1)

    ax_speed = fig.add_axes([0.15, 0.2, 0.7, 0.04])
    speed_slider = Slider(ax=ax_speed, label="s", valmin=0.01, valmax=60, valinit=33, valstep=0.01)

    ax_btn = fig.add_axes([0.02, 0.06, 0.10, 0.08])
    btn = Button(ax_btn, "Play")


    def draw_frame(i: int):
        i = int(np.clip(i, 0, time_steps - 1))
        state["frame"] = i
        for art, data, ax in zip(artists, datas, axs):
            # get the position of each thing at a given time step
            boids = data['positions'][i]
            vel   = data['velocities'][i]
            pred  = data['predator_positions'][i]

            # update the artists scatter based on these timesteps
            art['boids'].set_offsets(boids)
            art['pred'].set_offsets(pred)
            art['quiver'].set_offsets(boids)
            art['quiver'].set_UVC(vel[:, 0], vel[:, 1])

            ax.set_xlabel(f"frame {i+1}/{time_steps}")

    def on_speed_scrub(val):
        ms = int(val)
        state["speed"] = ms
        ani._interval = ms
        ani.event_source.stop()
        ani.event_source.interval = ms
        ani.event_source.start()

        fig.canvas.draw_idle()

    def on_time_scrub(val):
        state["playing"] = False
        btn.label.set_text("Play")
        draw_frame(int(val))
        fig.canvas.draw_idle()

    def on_button_clicked(_):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")

    def tick(_):
        if state["playing"]:
            nxt = (state["frame"] + 1) % time_steps
            playback_slider.eventson = False
            playback_slider.set_val(nxt)
            playback_slider.eventson = True

            draw_frame(nxt)
    
    playback_slider.on_changed(on_time_scrub)
    speed_slider.on_changed(on_speed_scrub)

    btn.on_clicked(on_button_clicked)

    draw_frame(0)
    ani = FuncAnimation(fig, tick, interval=state['speed'], blit=False,cache_frame_data=False)  # ~30 FPS
    plt.show()


def View(datas: list, overlay=True):
    times = [d.get('time_steps') for d in datas]
    if len(set(times)) > 1:
        print(f"Times: {times}")
        sys.tracebacklimit = 0
        raise Exception('The given simulations have inequal time sets, therefore cannot be compared.')
    time_steps = times[0]

    if DARK_MODE:
        plt.style.use('dark_background')

    # --- choose axes layout ---
    if overlay:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig, ax = plt.subplots(1, len(datas), figsize=(5 * len(datas), 5))
    
    axs = np.array( [ax for _ in range (len(datas))] ).flatten()

    cmap = plt.get_cmap("tab10")
    colours = [cmap(i % cmap.N) for i in range(len(datas))]

    artists = [GetArtist(ax, data, colour=c) for ax, data, c in zip(axs, datas, colours)]

    state = {"playing": False, "frame": 0, "speed": 33}
    plt.subplots_adjust(bottom=0.5)

    ax_time = fig.add_axes([0.15, 0.08, 0.7, 0.04])
    playback_slider = Slider(ax=ax_time, label="t", valmin=0, valmax=time_steps - 1, valinit=0, valstep=1)

    ax_speed = fig.add_axes([0.15, 0.2, 0.7, 0.04])
    speed_slider = Slider(ax=ax_speed, label="s", valmin=1, valmax=60, valinit=33, valstep=1)

    ax_btn = fig.add_axes([0.02, 0.06, 0.10, 0.08])
    btn = Button(ax_btn, "Play")

    def draw_frame(i: int):
        i = int(np.clip(i, 0, time_steps - 1))
        state["frame"] = i

        for art, data in zip(artists, datas):
            boids = data['positions'][i]
            vel   = data['velocities'][i]
            pred  = data['predator_positions'][i]

            art['boids'].set_offsets(boids)
            art['pred'].set_offsets(pred)
            art['quiver'].set_offsets(boids)
            art['quiver'].set_UVC(vel[:, 0], vel[:, 1])

        # label only once if overlay; otherwise each subplot already has its own
        if overlay:
            axs[0].set_xlabel(f"frame {i+1}/{time_steps}")
        else:
            for ax in axs:
                ax.set_xlabel(f"frame {i+1}/{time_steps}")

    def on_speed_scrub(val):
        ms = int(val)
        state["speed"] = ms
        ani.event_source.interval = ms
        fig.canvas.draw_idle()

    def on_time_scrub(val):
        state["playing"] = False
        btn.label.set_text("Play")
        draw_frame(int(val))
        fig.canvas.draw_idle()

    def on_button_clicked(_):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")

    def tick(_):
        if state["playing"]:
            nxt = (state["frame"] + 1) % time_steps
            playback_slider.eventson = False
            playback_slider.set_val(nxt)
            playback_slider.eventson = True
            draw_frame(nxt)

    playback_slider.on_changed(on_time_scrub)
    speed_slider.on_changed(on_speed_scrub)
    btn.on_clicked(on_button_clicked)

    draw_frame(0)
    ani = FuncAnimation(fig, tick, interval=state['speed'], blit=False, cache_frame_data=False)

    # keep reference alive (important!)
    fig._ani = ani

    plt.show()

    
def GetArtist(ax,data,colour):
    positions = data['positions']
    velocities = data['velocities']
    spawn_bounds = data['bounds']
    config_title = data.get('config_title',None)
    config_title = "No Title Found!" if config_title is None else f"{config_title[0]}.ini"
    
    #boid_count = data['boid_count']
    #predator_positions = data['predator_positions']
    #time_steps= data['time_steps']

    ax.set_title(config_title)
    spawn_bounds = np.array(spawn_bounds).flatten()
    left, right = float(spawn_bounds[0])/ZOOM, float(spawn_bounds[1])/ZOOM
    ax.set_xlim(left,right)
    ax.set_ylim(left,right)

    ax.set_aspect("equal", adjustable="box")

    boids_scatter = ax.scatter([], [], s=BOID_SIZE,color=colour)
    pred_scatter  = ax.scatter([], [], s=PRED_SIZE,color=colour)

    boids0 = positions[0]
    vel0 = velocities[0]

    boids_quiver = ax.quiver(boids0[:, 0], boids0[:, 1],vel0[:, 0], vel0[:, 1], angles='xy', scale_units='xy', scale=ARROW_SCALE, width=ARROW_WIDTH,color=colour)

    return {
        "boids": boids_scatter,
        "pred": pred_scatter,
        "quiver": boids_quiver,
    }

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
    filenames = args.input
    filenames = np.array(filenames).flatten()
    datas =[load_run(f) for f in filenames]
    View(datas)

if __name__ == "__main__":
    main()