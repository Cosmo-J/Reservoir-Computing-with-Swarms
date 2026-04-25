import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

SIM_STYLE = {
    "boid_size": 10,
    "boid_alpha": 0.85,
    "arrow_scale": 10,
    "arrow_width": 0.003,

    "boid_trail": 5,
    "boid_trail_alpha": 0.45,
    "boid_trail_width": 0.5,
    "boid_trail_color": None,

    "signal_size": 25,
    "signal_color": "red",
    "signal_marker": "o",

    "signal_trail": 4,
    "signal_trail_alpha": 0.45,
    "signal_trail_width": 1.5,

    "colors": None,
    "interval": 33,

    "wireframe_color": "gray",
    "wireframe_alpha": 0.12,
    "wireframe_width": 0.5,
    "major_radius": 10,
    "minor_radius": 4,
    "torus_padding": 2,
    "torus_u_res": 25,
    "torus_v_res": 20,
    "torus_view_elev": 25,
    "torus_view_azim": -60,
    "torus_axis_off": False,

    "ticks": [],
    "time_label": True,
    "time_per_frame": 0.02,

    "max_columns": 5,
    "dpi": 200,
}

def flat_render(replicas:list, frames=None, bounds=(-8,8), overlay=True, animate=False, **kwargs):
    if frames is None:
        frames = [0]
    frames = list(frames)

    style = {**SIM_STYLE, **kwargs}
    num_replicas = len(replicas)
    num_frames = len(frames)

    if num_replicas == 0:
        raise ValueError("replicas must not be empty")

    simulation_length = len(replicas[0]["positions"])
    if any(len(replica["positions"]) != simulation_length for replica in replicas):
        raise ValueError("all replicas must have the same simulation length")

    if any(frame < 0 or frame >= simulation_length for frame in frames):
        raise ValueError(f"frame indices must lie in [0, {simulation_length - 1}]")

    if len(bounds) == 2:
        bounds = [float(bounds[0]), float(bounds[1]), float(bounds[0]), float(bounds[1])]
    else:
        raise ValueError("bounds must be in the shape (min, max)")

    colors = style["colors"]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(num_replicas)]

    num_plots = num_frames * (1 if overlay else num_replicas)

    if overlay:
        ncols = min(style["max_columns"], num_plots)
        nrows = int(np.ceil(num_plots / ncols))
    else:
        nrows = num_replicas
        ncols = num_frames

    fig, axes = plt.subplots(nrows, ncols, dpi=style["dpi"], figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xticks(style["ticks"])
        ax.set_yticks(style["ticks"])
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[num_plots:]:
        ax.set_visible(False)

    if overlay:
        total_artists = num_frames * num_replicas
        axis_of = lambda artist_idx: artist_idx // num_replicas
        replica_of = lambda artist_idx: artist_idx % num_replicas
        frame_slot_of = lambda artist_idx: artist_idx // num_replicas
    else:
        total_artists = num_frames * num_replicas
        axis_of = lambda artist_idx: artist_idx
        replica_of = lambda artist_idx: artist_idx // num_frames
        frame_slot_of = lambda artist_idx: artist_idx % num_frames

    if style["time_label"] and not animate:
        if overlay:
            for frame_slot, frame_idx in enumerate(frames):
                axes[frame_slot].set_title(f"t = {np.round(style['time_per_frame'] * frame_idx, decimals=1)}")
        else:
            for plot_idx in range(num_plots):
                replica_idx = plot_idx // num_frames
                frame_slot = plot_idx % num_frames
                frame_idx = frames[frame_slot]
                axes[plot_idx].set_title(f"replica {replica_idx}, t = {np.round(style['time_per_frame'] * frame_idx, decimals=1)}")

    def make_artists(ax, replica, color, frame_index=0):
        p_init = replica["positions"][frame_index]
        velocities = replica.get("velocities")
        signal = replica.get("input_signal")

        boids = ax.scatter(p_init[:, 0], p_init[:, 1], s=style["boid_size"], alpha=style["boid_alpha"], color=color, animated=animate)
        boid_trail_color = color if style["boid_trail_color"] is None else style["boid_trail_color"]
        boid_trail, = ax.plot([], [], color=boid_trail_color, alpha=style["boid_trail_alpha"], linewidth=style["boid_trail_width"], animated=animate)

        arrows = None
        if velocities is not None:
            v_init = velocities[frame_index]
            arrows = ax.quiver(p_init[:, 0], p_init[:, 1], v_init[:, 0], v_init[:, 1], color=color, alpha=0.4, angles="xy", scale_units="xy", scale=style["arrow_scale"], width=style["arrow_width"], animated=animate)

        signal_point = None
        signal_trail = None
        if signal is not None:
            signal_point = ax.scatter(signal[frame_index, 0], signal[frame_index, 1], s=style["signal_size"], color=style["signal_color"], marker=style["signal_marker"], zorder=3, animated=animate)
            signal_trail, = ax.plot([], [], color=style["signal_color"], alpha=style["signal_trail_alpha"], linewidth=style["signal_trail_width"], animated=animate)

        return {"boids": boids, "boid_trail": boid_trail, "arrows": arrows, "signal": signal_point, "signal_trail": signal_trail}

    def update_artist(replica, artist, frame_index):
        updated = []

        positions = replica["positions"]
        pos_now = positions[frame_index]
        velocities = replica.get("velocities")
        input_signal = replica.get("input_signal")

        artist["boids"].set_offsets(pos_now)
        updated.append(artist["boids"])

        if style["boid_trail"] and artist["boid_trail"] is not None:
            start = max(0, frame_index - int(style["boid_trail"]))
            trail_segment = positions[start:frame_index + 1]
            n_boids = trail_segment.shape[1]
            nan_separator = np.full((1, n_boids, 2), np.nan)
            separated_trails = np.vstack([trail_segment, nan_separator])
            flat_trail = separated_trails.transpose(1, 0, 2).reshape(-1, 2)
            artist["boid_trail"].set_data(flat_trail[:, 0], flat_trail[:, 1])
            updated.append(artist["boid_trail"])

        if velocities is not None and artist["arrows"] is not None:
            vel_now = velocities[frame_index]
            artist["arrows"].set_offsets(pos_now)
            artist["arrows"].set_UVC(vel_now[:, 0], vel_now[:, 1])
            updated.append(artist["arrows"])

        if input_signal is not None and artist["signal"] is not None:
            signal_now = input_signal[frame_index]
            artist["signal"].set_offsets(signal_now[None, :])
            updated.append(artist["signal"])

            if style["signal_trail"] > 0 and artist["signal_trail"] is not None:
                start = max(0, frame_index - int(style["signal_trail"]))
                trail_xy = input_signal[start:frame_index + 1]
                artist["signal_trail"].set_data(trail_xy[:, 0], trail_xy[:, 1])
                updated.append(artist["signal_trail"])

        return [artist_obj for artist_obj in updated if artist_obj is not None]

    artists = []
    for artist_idx in range(total_artists):
        replica_idx = replica_of(artist_idx)
        frame_slot = frame_slot_of(artist_idx)
        axis_idx = axis_of(artist_idx)
        artists.append(make_artists(axes[axis_idx], replicas[replica_idx], colors[replica_idx], frames[frame_slot]))

    def draw(global_frame):
        updated_artists = []
        for artist_idx, artist in enumerate(artists):
            replica_idx = replica_of(artist_idx)
            frame_slot = frame_slot_of(artist_idx)
            frame_index = min(frames[frame_slot] + global_frame, simulation_length - 1) if animate else frames[frame_slot]
            updated_artists.extend(update_artist(replicas[replica_idx], artist, frame_index))
        return updated_artists

    if animate:
        animation = FuncAnimation(fig, draw, frames=simulation_length, interval=style["interval"], blit=True, cache_frame_data=False)
        return fig, axes[:num_plots], animation

    draw(0)
    return fig, axes[:num_plots]

def torus_render(replicas:list, sim_width, frames=[0], bounds=None, overlay=True, animate=False, **kwargs):
    style = {**SIM_STYLE, **kwargs}

    num_replicas = len(replicas)
    num_frames = len(frames)

    multi_replica = num_replicas > 1
    multi_frame = num_frames > 1

    if multi_frame and multi_replica:
        raise TypeError("Cannot render multiple frames across multiple simulations")

    simulation_length = len(replicas[0]["positions"])

    if bounds is None:
        limit = style["major_radius"] + style["minor_radius"] + style["torus_padding"]
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be None, or in the shape (min, max)")
        limit = max(abs(float(bounds[0])), abs(float(bounds[1])))

    colors = style["colors"]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(num_replicas)]

    max_cols = style["max_columns"]

    if multi_frame:
        num_plots = num_frames
    else:
        if overlay:
            num_plots = 1
        else:
            num_plots = num_replicas

    ncols = min(max_cols, num_plots)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, dpi=style["dpi"], figsize=(4 * ncols, 4 * nrows), squeeze=False, subplot_kw={"projection": "3d"})
    axes = axes.ravel()

    u, v = np.mgrid[0:2 * np.pi:complex(style["torus_u_res"]), 0:2 * np.pi:complex(style["torus_v_res"])]
    wire_x = (style["major_radius"] + style["minor_radius"] * np.cos(v)) * np.cos(u)
    wire_y = (style["major_radius"] + style["minor_radius"] * np.cos(v)) * np.sin(u)
    wire_z = style["minor_radius"] * np.sin(v)

    for i, ax in enumerate(axes):
        if i >= num_plots:
            ax.set_visible(False)
            continue

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(style["torus_view_elev"], style["torus_view_azim"])

        if style["torus_axis_off"]:
            ax.axis("off")

        ax.plot_wireframe(wire_x, wire_y, wire_z, color=style["wireframe_color"], alpha=style["wireframe_alpha"], linewidth=style["wireframe_width"])

        if style["time_label"] and not animate:
            time = style["time_per_frame"]
            ax.set_title(f"t = {np.round(time * frames[i], decimals=1)}")

    def make_artists(ax, replica, color, frame_index=0):
        p_init = replica["positions"][frame_index]
        signal = replica.get("input_signal")

        x, y, z = to_torus_3d(p_init, sim_width, style["major_radius"], style["minor_radius"])
        boids = ax.scatter(x, y, z, s=style["boid_size"], alpha=style["boid_alpha"], color=color, animated=animate)

        boid_trail_color = style["boid_trail_color"]
        if boid_trail_color is None:
            boid_trail_color = color

        boid_trail = None
        if style["boid_trail"]:
            boid_trail, = ax.plot([], [], [], color=boid_trail_color, alpha=style["boid_trail_alpha"], linewidth=style["boid_trail_width"], animated=animate)

        signal_point = None
        signal_trail = None
        if signal is not None:
            sx, sy, sz = to_torus_3d(signal[frame_index:frame_index + 1], sim_width, style["major_radius"], style["minor_radius"])
            signal_point = ax.scatter(sx, sy, sz, s=style["signal_size"], color=style["signal_color"], marker=style["signal_marker"], zorder=3, animated=animate)
            signal_trail, = ax.plot([], [], [], color=style["signal_color"], alpha=style["signal_trail_alpha"], linewidth=style["signal_trail_width"], animated=animate)

        return {"boids": boids, "boid_trail": boid_trail, "signal": signal_point, "signal_trail": signal_trail}

    def update_artist(replica, artist, frame_index):
        updated = []

        positions = replica["positions"]
        pos_now = positions[frame_index]
        input_signal = replica.get("input_signal")

        x, y, z = to_torus_3d(pos_now, sim_width, style["major_radius"], style["minor_radius"])
        artist["boids"]._offsets3d = (x, y, z)
        updated.append(artist["boids"])

        if style["boid_trail"] and artist["boid_trail"] is not None:
            start = max(0, frame_index - int(style["boid_trail"]))
            trail_segment = positions[start:frame_index + 1]
            n_boids = trail_segment.shape[1]
            nan_separator = np.full((1, n_boids, 2), np.nan)
            separated_trails = np.vstack([trail_segment, nan_separator])
            flat_trail = separated_trails.transpose(1, 0, 2).reshape(-1, 2)
            tx, ty, tz = to_torus_3d(flat_trail, sim_width, style["major_radius"], style["minor_radius"])
            artist["boid_trail"].set_data(tx, ty)
            artist["boid_trail"].set_3d_properties(tz)
            updated.append(artist["boid_trail"])

        if input_signal is not None:
            signal_now = input_signal[frame_index:frame_index + 1]
            sx, sy, sz = to_torus_3d(signal_now, sim_width, style["major_radius"], style["minor_radius"])
            artist["signal"]._offsets3d = (sx, sy, sz)
            updated.append(artist["signal"])

            if style["signal_trail"] > 0 and artist["signal_trail"] is not None:
                start = max(0, frame_index - int(style["signal_trail"]))
                trail_xy = input_signal[start:frame_index + 1]
                tx, ty, tz = to_torus_3d(trail_xy, sim_width, style["major_radius"], style["minor_radius"])
                artist["signal_trail"].set_data(tx, ty)
                artist["signal_trail"].set_3d_properties(tz)
                updated.append(artist["signal_trail"])

        clean = []
        for item in updated:
            if item is not None:
                clean.append(item)

        return clean

    artists = []

    if multi_frame:
        replica_ptrs = [0] * num_frames
        frame_ptrs = list(frames)
        for i, frame_idx in enumerate(frame_ptrs):
            artists.append(make_artists(axes[i], replicas[0], colors[0], frame_idx))
    else:
        replica_ptrs = list(range(num_replicas))
        if overlay:
            for replica_idx in replica_ptrs:
                artists.append(make_artists(axes[0], replicas[replica_idx], colors[replica_idx], frames[0]))
        else:
            for i, replica_idx in enumerate(replica_ptrs):
                artists.append(make_artists(axes[i], replicas[replica_idx], colors[replica_idx], frames[0]))

    def draw(frame_index):
        updated_artists = []

        if animate and style["time_label"]:
            time = style["time_per_frame"]
            for ax in axes[:num_plots]:
                ax.set_title(f"t = {np.round(time * frame_index, decimals=1)}")

        for replica_idx, artist in zip(replica_ptrs, artists):
            updated_artists.extend(update_artist(replicas[replica_idx], artist, frame_index))

        return updated_artists

    if animate:
        animation = FuncAnimation(fig, draw, frames=simulation_length, interval=style["interval"], blit=True, cache_frame_data=False)
        return fig, axes[:num_plots], animation

    if multi_frame:
        for frame_idx, artist in zip(frame_ptrs, artists):
            update_artist(replicas[0], artist, frame_idx)
    else:
        draw(frames[0])

    return fig, axes[:num_plots]

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



RIDGE_STYLE = {
    "dpi": 100,
    "plot_ratio": (2, 1),
    "fig_width": 20,
    "grid":False,
    "font_size": 34,

    "y_ticks": [-5, 0, 5],
    "x_ticks": range(0,1250,250),
    "axes_linewidth":2,

    "delta_t":None,

    "signal_label": r"$\bar{x}_L$",
    "signal_color": "black",
    "signal_linestyle":"solid",
    "signal_linewidth":2.5,


    "predictions_line_styles":"dashed",
    "predictions_labels":None,
    "predictions_colors":None,
    "predictions_linewidths":2.5,
}

def plot_ridge_predictions(predictions, signal, **kwargs):
    style = {**RIDGE_STYLE, **kwargs}
    
    
    plt.rcParams.update(
        {"font.family": "serif",
        "mathtext.fontset": "cm",
        'axes.linewidth': style["axes_linewidth"]
        })

    num_predictions = len(predictions)
    if num_predictions == 0:
        raise ValueError("Predictions must contain at least one prediction dictionary")
    
    if not isinstance(predictions, list):
        predictions = [predictions]

    try:
        signal = np.asarray(signal)
        assert len(signal.shape)==1
    except:
        raise ValueError("Invalid signal. Must be array like and shape (T,1) where T is at least as many as your plot range.")

    pred_dists = [p["prediction_distance"].item() for p in predictions]
    if len(set(pred_dists))!=1:
        print(f"WARNING: Predictions generated with different prediction distances so plot will be incorrect: {pred_dists}")
        print(f"Using the first in list.")

    prediction_distance = pred_dists[0]
    signal_target = signal[prediction_distance:]


    ratio_width, ratio_height = style["plot_ratio"]
    fig_height = style["fig_width"] * ratio_height / ratio_width

    fig, ax = plt.subplots(figsize=(style["fig_width"], fig_height), dpi=style["dpi"])
    plt.subplots_adjust(bottom=0.2)

    ax.plot(signal_target, color=style["signal_color"], label=style["signal_label"],linewidth=style["signal_linewidth"])

    
    p_colors = style["predictions_colors"]
    if isinstance(p_colors, list):
        if len(p_colors) != num_predictions:
            raise ValueError(f"predictions_colors={p_colors} did not have the same number of options as number of predictions.")
    else:
        cmap = plt.get_cmap("tab10")
        p_colors = [cmap(i % cmap.N) for i in range(num_predictions)]

    p_line_styles = style["predictions_line_styles"]
    if isinstance(p_line_styles, list):
        if len(p_line_styles) != num_predictions:
            raise ValueError(f"predictions_line_styles={p_line_styles} did not have the same number of options as number of predictions.")
    else:
        p_line_styles = [p_line_styles for _ in range(num_predictions)]

    p_linewidth = style["predictions_linewidths"]
    if isinstance(p_linewidth, list):
        if len(p_linewidth) != num_predictions:
            raise ValueError(f"prediction_linewidth={p_linewidth} did not have the same number of options as number of predictions.")
    else:
        p_linewidth = [p_linewidth for _ in range(num_predictions)]



    for i, p in enumerate(predictions):
        prediction = p["prediction"]
        prediction_offset = len(signal_target) - len(prediction)
        x_values = np.arange(prediction_offset, prediction_offset + len(prediction))
        ax.plot(x_values, prediction, color=p_colors[i], linestyle=p_line_styles[i],linewidth=p_linewidth[i])
    
    plt.grid(False)

    if style["delta_t"] is None:
        delta_t = 1
    else:
        delta_t = style["delta_t"]

    y_label = rf"${style['signal_label'].strip('$')}(t+{prediction_distance*delta_t})$"


    plt.ylabel(y_label)
    ax.set_yticks(style["y_ticks"])

    y_max = max(style["y_ticks"])
    y_min = min(style["y_ticks"])
    ax.set_ylim((y_min,y_max))
    plt.xlabel(r"$t$")

    x_ticks_delta_t = []
    for tick in style["x_ticks"]:
        x = tick*delta_t
        if x%1==0:
            x_ticks_delta_t.append(int(x))
        else:
            x_ticks_delta_t.append(x)
    ax.set_xticks(style["x_ticks"])
    ax.set_xticklabels(x_ticks_delta_t)
    x_max = max(style["x_ticks"])
    x_min = min(style["x_ticks"])
    ax.set_xlim((x_min,x_max))

    
    ax.xaxis.label.set_fontsize(style["font_size"])
    ax.yaxis.label.set_fontsize(style["font_size"])

    ax.tick_params(axis='both', which='major',labelsize=style["font_size"],direction='in',pad=15,width=style["axes_linewidth"],size=style["font_size"]/3,top=True)

    return ax


C_PROF_STYLE = {
    "dpi": 200,
    "plot_ratio": (1, 1),
    "fig_width": 8,
    "grid": False,
    "font_size": 18,
    "x_ticks": None,
    "y_ticks": [0, 0.5, 1],
    "axes_linewidth": 1,
    "truncated_to": 100,
    "cc_dp": 1,
    "show_consistent_capacity": True,
    "gamma2_colors": None,
    "gamma2_linestyles": "solid",
    "gamma2_linewidths": 2,
    "capacity_labels": None,
}

def plot_consistency_profile(consistent_capacity, consistency_profile, **kwargs):
    style = {**C_PROF_STYLE, **kwargs}

    plt.rcParams.update({"font.size": style["font_size"], "axes.linewidth": style["axes_linewidth"]})

    if not isinstance(consistency_profile, list):
        consistency_profile = [consistency_profile]

    if not isinstance(consistent_capacity, list):
        consistent_capacity = [consistent_capacity]

    num_profiles = len(consistency_profile)

    if len(consistent_capacity) != num_profiles:
        raise ValueError("consistent_capacity must have the same number of values as consistency_profile.")

    truncated_to = style["truncated_to"]
    ratio_width, ratio_height = style["plot_ratio"]
    fig_height = style["fig_width"] * ratio_height / ratio_width
    fig, ax = plt.subplots(figsize=(style["fig_width"], fig_height), dpi=style["dpi"])

    p_colors = style["gamma2_colors"]
    if isinstance(p_colors, list):
        if len(p_colors) != num_profiles:
            raise ValueError("gamma2_colors must have the same number of options as consistency_profile.")
    else:
        cmap = plt.get_cmap("tab10")
        p_colors = [cmap(i % cmap.N) for i in range(num_profiles)]

    p_line_styles = style["gamma2_linestyles"]
    if isinstance(p_line_styles, list):
        if len(p_line_styles) != num_profiles:
            raise ValueError("gamma2_linestyles must have the same number of options as consistency_profile.")
    else:
        p_line_styles = [p_line_styles for _ in range(num_profiles)]

    p_linewidths = style["gamma2_linewidths"]
    if isinstance(p_linewidths, list):
        if len(p_linewidths) != num_profiles:
            raise ValueError("gamma2_linewidths must have the same number of options as consistency_profile.")
    else:
        p_linewidths = [p_linewidths for _ in range(num_profiles)]

    capacity_labels = style["capacity_labels"]
    if capacity_labels is not None and len(capacity_labels) != num_profiles:
        raise ValueError("capacity_labels must have the same number of options as consistency_profile.")

    for i, profile in enumerate(consistency_profile):
        gamma2_k_ranked = np.sort(np.asarray(profile))[::-1]
        ax.plot(gamma2_k_ranked[:truncated_to + 1], color=p_colors[i], linestyle=p_line_styles[i], linewidth=p_linewidths[i])

        if style["show_consistent_capacity"]:
            capacity = np.round(consistent_capacity[i], decimals=style["cc_dp"])
            middle_text = rf"$\Theta={capacity}$" if capacity_labels is None else rf"{capacity_labels[i]} $\Theta={capacity}$"
            ax.text(truncated_to / 2, 0.5 - i * 0.08, middle_text, color=p_colors[i], fontweight="bold", horizontalalignment="center")

    ax.set_xlim(0, truncated_to)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, truncated_to // 2, truncated_to] if style["x_ticks"] is None else style["x_ticks"])
    ax.set_yticks(style["y_ticks"])
    ax.tick_params(axis="both", which="major", labelsize=style["font_size"], direction="in", length=6, width=style["axes_linewidth"], top=True, right=True)
    ax.set_xlabel(r"$k$", fontsize=style["font_size"])
    ax.set_ylabel(r"$\gamma^{2}_{k}$", rotation=0, labelpad=20, fontsize=style["font_size"])
    ax.grid(style["grid"])
    fig.tight_layout()

    return ax