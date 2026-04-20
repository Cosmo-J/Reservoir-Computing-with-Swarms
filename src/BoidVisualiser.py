from __future__ import annotations

from .ConfigManager import compare_params
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

STYLE_DEFAULTS = {
    "boid_size": 10,
    "boid_alpha": 0.85,
    "arrow_scale": 10,
    "arrow_width": 0.003,
    "signal_size": 25,
    "signal_color": "red",
    "signal_marker": "o",
    "signal_trail": 4,
    "trail_alpha": 0.45,
    "trail_width": 1.5,
    "colors": None,
    "interval": 33,

    "wireframe_color": "gray",
    "wireframe_alpha": 0.12,
    "wireframe_width": 0.5,
    "major_radius": 10,
    "minor_radius": 4,
    
    "ticks":[],
    "time_label":True,

    "time_per_frame":0.02,

    "max_columns":5
}


def flat_render(replicas:list, frames=[0], bounds=(-8,8), overlay=True, animate=False, **kwargs):
    """        
        Parameters
        ----------
        boid_simulation : dict
            dict which expects items for 'positions','velocities','input_signal'. 
            Example: 'positions':(T,N,2),'velocites':(T,N,2) where T is time steps, N is number of boid agents. 'input_signal' may be empty.
        frame : int, optional
            _description_, by default 0
        bounds : _type_, optional
            _description_, by default None
        overlay : bool, optional
            _description_, by default False
        animate : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
    """

    style = {**STYLE_DEFAULTS, **kwargs}
    num_replicas = len(replicas)
    num_frames = len(frames)
    
    multi_replica = num_replicas>1
    multi_frame = num_frames>1

    if multi_frame and multi_replica:
        raise TypeError("Cannot render multiple frames across multiple simulations")

    simulation_length = len(replicas[0]['positions'])

    if len(bounds) == 2:
        bounds = [float(bounds[0]), float(bounds[1]), float(bounds[0]), float(bounds[1])]
    else:
        raise ValueError("bounds must be in the shape (min, max)")

    colors = style["colors"]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(num_replicas)]

    max_cols = style["max_columns"]

    if multi_frame:
        num_plots = num_frames
    elif overlay:
        num_plots = 1
    else:
        num_plots = num_replicas

    ncols = min(max_cols, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for i,ax in enumerate(axes):
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xticks(style["ticks"])
        ax.set_yticks(style["ticks"])
        ax.set_aspect("equal", adjustable="box")

        if style["time_label"] and not animate:
            time = style["time_per_frame"]
            ax.set_title(f"t = {int(time*frames[i])}")

    def make_artists(ax, replica, color,frame_index=0):
        p_init = replica["positions"][frame_index]
        velocities = replica.get("velocities")
        signal = replica.get("input_signal")

        boids = ax.scatter(p_init[:, 0], p_init[:, 1], s=style["boid_size"], alpha=style["boid_alpha"], color=color, animated=animate)

        arrows=None
        if velocities is not None:
            v_init = velocities[frame_index]
            arrows = ax.quiver(p_init[:,0],p_init[:,1],v_init[:,0],v_init[:,1],color=color,alpha=0.4,angles="xy", scale_units="xy", scale=style["arrow_scale"],width=style["arrow_width"],animated=animate)

        singal_point = None
        signal_trail = None
        if signal is not None:
            singal_point = ax.scatter(signal[frame_index, 0], signal[frame_index, 1], s=style["signal_size"], color=style["signal_color"], marker=style["signal_marker"], zorder=3, animated=animate)
            signal_trail, = ax.plot([], [], color=style["signal_color"], alpha=style["trail_alpha"], linewidth=style["trail_width"], animated=animate)

        return {"boids": boids, "arrows": arrows, "signal": singal_point, "signal_trail": signal_trail}

    def update_artist(replica, artist, frame_index):
        updated = []

        pos_now = replica["positions"][frame_index]
        velocities = replica.get("velocities")
        input_signal = replica.get("input_signal")

        artist["boids"].set_offsets(pos_now)
        updated.append(artist["boids"])

        if velocities is not None:
            vel_now = velocities[frame_index]
            artist["arrows"].set_offsets(pos_now)
            artist["arrows"].set_UVC(vel_now[:, 0], vel_now[:, 1])
            updated.append(artist["arrows"])

        if input_signal is not None:
            signal_now = input_signal[frame_index]
            artist["signal"].set_offsets(signal_now[None, :])
            updated.append(artist["signal"])

            if style["signal_trail"] > 0 and artist["signal_trail"] is not None:
                start = max(0, frame_index - int(style["signal_trail"]))
                trail_xy = input_signal[start:frame_index + 1]
                artist["signal_trail"].set_data(trail_xy[:, 0], trail_xy[:, 1])
                updated.append(artist["signal_trail"])

        return [a for a in updated if a is not None]
    
    artists = []

    if multi_frame:
        replica_ptrs = [0] * num_frames
        frame_ptrs = list(frames)
        for i, (replica_idx, frame_idx) in enumerate(zip(replica_ptrs, frame_ptrs)):
            artists.append(make_artists(axes[i], replicas[replica_idx], colors[replica_idx], frame_idx))
    elif overlay:
        replica_ptrs = list(range(num_replicas))
        for replica_idx in replica_ptrs:
            artists.append( make_artists(axes[0], replicas[replica_idx], colors[replica_idx], frames[0]))
    else:
        replica_ptrs = list(range(num_replicas))
        for i, replica_idx in enumerate(replica_ptrs):
            artists.append(make_artists(axes[i], replicas[replica_idx], colors[replica_idx], frames[0]))

    def draw(frame_index):
        updated_artists = []
        for replica_idx, artist in zip(replica_ptrs, artists):
            updated_artists.extend(update_artist(replicas[replica_idx], artist, frame_index))
        return updated_artists

    if animate:
        animation = FuncAnimation(fig,draw,frames=simulation_length,interval=style["interval"],blit=True,cache_frame_data=False,)
        return fig, axes[:num_plots], animation

    else:
        if multi_frame:
            for replica_idx, frame_idx, artist in zip(replica_ptrs, frame_ptrs, artists):
                update_artist(replicas[replica_idx], artist, frame_idx)
        else:
            draw(frames[0])

        return fig, axes[:num_plots]


def torus_render(replicas:list, sim_width, frames=[0], bounds=(-8,8), overlay=True, animate=False, **kwargs):
    style = {**STYLE_DEFAULTS, **kwargs}

    num_replicas = len(replicas)
    num_frames = len(frames)
    
    multi_replica = num_replicas>1
    multi_frame = num_frames>1
    frame

    if multi_frame and multi_replica:
        raise TypeError("Cannot render multiple frames across multiple simulations")

    simulation_length = len(replicas[0]['positions'])

    if len(bounds) == 2:
        bounds = [float(bounds[0]), float(bounds[1]), float(bounds[0]), float(bounds[1])]
    else:
        raise ValueError("bounds must be None, or in the shape (min, max)")

    colors = style["colors"]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(num_replicas)]

    max_cols = style["max_columns"]
    if multi_frame:
        num_plots = num_frames
    elif overlay:
        num_plots = 1
    else:
        num_plots = num_replicas

    ncols = min(max_cols, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for i,ax in enumerate(axes):
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xticks(style["ticks"])
        ax.set_yticks(style["ticks"])
        ax.set_aspect("equal", adjustable="box")

        if style["time_label"] and not animate:
            time = style["time_per_frame"]
            ax.set_title(f"t = {int(time*frames[i])}")


    axis_count = 1 if overlay else len(replicas)
    fig, axes = plt.subplots(1, axis_count, figsize=(6 * axis_count, 6), subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes).tolist()

    limit = style["major_radius"] + style["minor_radius"] + 2
    u, v = np.mgrid[0:2 * np.pi:25j, 0:2 * np.pi:20j]
    wire_x = (style["major_radius"] + style["minor_radius"] * np.cos(v)) * np.cos(u)
    wire_y = (style["major_radius"] + style["minor_radius"] * np.cos(v)) * np.sin(u)
    wire_z = style["minor_radius"] * np.sin(v)

    for axis_index, ax in enumerate(axes):
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        ax.plot_wireframe(wire_x, wire_y, wire_z, color=style["wireframe_color"], alpha=style["wireframe_alpha"], linewidth=style["wireframe_width"])

    def make_artists(ax, replica, color):
        p_init = replica["positions"][0]
        signal = replica.get("input_signal")
        x, y, z = to_torus_3d(p_init, sim_width, style["major_radius"], style["minor_radius"])
        boids = ax.scatter(x, y, z, s=style["boid_size"], alpha=style["boid_alpha"], color=color, animated=animate)
        signal_point = None
        signal_trail = None
        if signal is not None:
            sx, sy, sz = to_torus_3d(signal[0:1], sim_width, style["major_radius"], style["minor_radius"])
            signal_point = ax.scatter(sx, sy, sz, s=style["signal_size"], color=style["signal_color"], marker=style["signal_marker"], zorder=3, animated=animate)
            signal_trail, = ax.plot([], [], [], color=style["signal_color"], alpha=style["trail_alpha"], linewidth=style["trail_width"], animated=animate)
        return {"boids": boids, "signal": signal_point, "signal_trail": signal_trail}

    artists = []
    for i, replica in enumerate(replicas):
        if overlay:
            artists.append(make_artists(axes[0], replica, colors[i]))
        else:
            artists.append(make_artists(axes[i], replica, colors[i]))

    def draw(frame_index):
        updated_artist = []
        for i, (replica, artist) in enumerate(zip(replicas, artists)):
            ax = axes[0] if overlay else axes[i]
            ax.set_title(f"frame {frame_index}" if overlay else f"replica {i}\n frame {frame_index}")
            pos_now = replica["positions"][frame_index]
            input_signal = replica.get("input_signal", None)
            x, y, z = to_torus_3d(pos_now, sim_width, style["major_radius"], style["minor_radius"])
            artist["boids"]._offsets3d = (x, y, z)
            updated_artist.append(artist["boids"])
            if input_signal is not None:
                signal_now = input_signal[frame_index:frame_index+1]
                sx, sy, sz = to_torus_3d(signal_now, sim_width, style["major_radius"], style["minor_radius"])
                artist["signal"]._offsets3d = (sx, sy, sz)
                updated_artist.append(artist["signal"])
                if style["signal_trail"] > 0:
                    start = max(0, frame_index - int(style["signal_trail"]))
                    trail_xy = input_signal[start:frame_index + 1]
                    tx, ty, tz = to_torus_3d(trail_xy, sim_width, style["major_radius"], style["minor_radius"])
                    artist["signal_trail"].set_data(tx, ty)
                    artist["signal_trail"].set_3d_properties(tz)
                    updated_artist.append(artist["signal_trail"])
        return [i for i in updated_artist if i is not None]

    if animate:
        animation = FuncAnimation(fig, draw, frames=total_frames, interval=style["interval"], blit=True, cache_frame_data=False)
        return fig, axes, animation
    else:
        draw(frame)
        return fig, axes


def to_torus_3d(positions, sim_width, major_radius=10, minor_radius=4):
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    phi = 2 * np.pi * (points[:, 0] / sim_width)
    theta = 2 * np.pi * (points[:, 1] / sim_width)
    x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = minor_radius * np.sin(theta)
    return x, y, z


def stupid_test_code():
    from .BoidSimulator import BoidSimulator
    from .SaverLoader import find_npzs

    datas = find_npzs("tests/donut")
    sim_w = datas[0].get('config').item().get('sim_width')
    print(sim_w)
    nice = BoidSimulator.construct_view_dict(datas)
    ax = torus_render(nice,sim_w,animate=True,overlay=False)
    plt.show()
    exit()

#stupid_test_code()