import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

from .SimParams import SimParams

# --- Global Visual Constants ---
DARK_MODE = False
ZOOM = 0.1
BOID_SIZE = 50*ZOOM
PRED_SIZE = 100*ZOOM
ARROW_SCALE = 20/(0.5*ZOOM)
ARROW_WIDTH = 0.05 * ZOOM

class BoidVisualizerOLD:
    def __init__(self, datasets, overlay=False):
        self.datasets = datasets
        self.overlay = overlay
        self.num_sims = len(datasets)
        
        # Initial geometry state
        self.is_torus = datasets[0].get('coord_type') == 'torus'
        self.torus_toggle = datasets[0].get('coord_type') == 'torus'
        self.time_steps = len(datasets[0]['positions'])
        
        self.playing = False
        self.frame = 0
        self.speed = 33
        
        self.fig = plt.figure(figsize=(10, 8) if not overlay else (8, 9))
        self.axs = []
        self.artists = []
        
        self._init_layout()
        self._init_ui()
        self.draw_frame(0)
        
        self.ani = FuncAnimation(
            self.fig, 
            self.tick, 
            interval=self.speed, 
            cache_frame_data=False,
            save_count=self.time_steps
        )
        self.fig._ani = self.ani 

        if DARK_MODE:
            plt.style.use('dark_background')

    def _init_layout(self):
        """Clears the figure and re-initializes axes to prevent interaction artifacts."""
        self.fig.clf() # Clear the entire figure to reset the backend state
        self.axs = []
        self.artists = []
        
        num_cols = 1 if self.overlay else self.num_sims
        cmap = plt.get_cmap("tab10")
        
        for i in range(self.num_sims):
            if self.overlay and i > 0:
                ax = self.axs[0]
            else:
                proj = '3d' if self.torus_toggle else None
                ax = self.fig.add_subplot(1, num_cols, i + 1, projection=proj)
                if self.torus_toggle:
                    ax.set_box_aspect([1, 1, 1])
                    ax.axis('off')
                else:
                    ax.set_aspect("equal", adjustable="box")
                self.axs.append(ax)
            
            color = cmap(i % cmap.N)
            self.artists.append(self._create_artist(ax, self.datasets[i], color))
        
        # Because clf() wipes the figure, we must re-init the UI widgets
        self._init_ui()

    def _create_artist(self, ax, data, color):
        # Below is a crude way of determining which version of the simulation was saved in the npz
        sim_params = data.get('parameters',SimParams.DEFAULTS)
        self.show_lorenz = sim_params.get('PREDATOR')
        self.show_lorenz = True if self.show_lorenz is None else False

        sim_width = data.get('sim_width', sim_params.get('SIM_WIDTH'))#this ensures compatibility with older versions
        coord_type = sim_params.get('COORD_SYSTEM')

        config_title = data.get('config_title', ["Simulation"])[0]
        spawn_bounds = data.get('spawn_bounds', data.get('bounds', [-1.0, 1.0]))
        
        ax.set_title(config_title)
        
        if self.torus_toggle and self.is_torus:
            # 3D Torus Projection View
            R, r = 10, 4
            u, v = np.mgrid[0:2*np.pi:25j, 0:2*np.pi:20j]
            xw = (R + r * np.cos(v)) * np.cos(u)
            yw = (R + r * np.cos(v)) * np.sin(u)
            zw = r * np.sin(v)
            ax.plot_wireframe(xw, yw, zw, color="gray", alpha=0.1, linewidth=0.5)
            
            limit = R + r + 2
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
            
            boids = ax.scatter([], [], [], s=BOID_SIZE, color=color, alpha=0.8)
            pred = ax.scatter([], [], [], s=PRED_SIZE, color='red', marker='X',visible = self.show_lorenz)
            return {"type": "3d", "boids": boids, "pred": pred, "L": sim_width}
        else:
            # 2D Flat View (Limits determined by provided logic)
            spawn_bounds = np.array(spawn_bounds).flatten()
            if coord_type == 'torus':
                left, right = 0.0, float(sim_width)
            else:
                left, right = float(spawn_bounds[0])/ZOOM, float(spawn_bounds[1])/ZOOM

            ax.set_xlim(left, right)
            ax.set_ylim(left, right)
            
            p0 = data['positions'][0]
            v0 = data['velocities'][0]
            boids = ax.scatter(p0[:, 0], p0[:, 1], s=BOID_SIZE, color=color)
            pred = ax.scatter([], [], s=PRED_SIZE, color='red',visible=self.show_lorenz)
            quiver = ax.quiver(p0[:, 0], p0[:, 1], v0[:, 0], v0[:, 1], 
                                color=color, alpha=0.4, scale=ARROW_SCALE, width=ARROW_WIDTH)
            return {"type": "2d", "boids": boids, "pred": pred, "quiver": quiver}

    def _init_ui(self):
        self.fig.subplots_adjust(bottom=0.25)
        
        ax_time = self.fig.add_axes([0.25, 0.12, 0.55, 0.03])
        self.slider_time = Slider(ax_time, "Time ", 0, self.time_steps - 1, valinit=0, valstep=1)
        self.slider_time.on_changed(self._on_slider_manual)
        
        ax_play = self.fig.add_axes([0.1, 0.11, 0.1, 0.05])
        self.btn_play = Button(ax_play, "Play")
        self.btn_play.on_clicked(self._toggle_playback)
        
        if self.is_torus:
            ax_toggle = self.fig.add_axes([0.1, 0.04, 0.15, 0.05])
            self.btn_toggle = Button(ax_toggle, "Toggle 2D/3D")
            self.btn_toggle.on_clicked(self._toggle_geometry)

    def _toggle_geometry(self, event):
        #self.torus_toggle = not self.torus_toggle
        self.torus_toggle = not self.torus_toggle

        self._init_layout()

        self.slider_time.set_val(self.frame)
        self.draw_frame(self.frame)
        self.fig.canvas.draw_idle()

    def _on_slider_manual(self, val):
        if not self.playing:
            self.draw_frame(val)

    def _toggle_playback(self, event):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def _to_torus_3d(self, pos, L, R=10, r=4):
        phi = 2 * np.pi * (pos[:, 0] / L)
        theta = 2 * np.pi * (pos[:, 1] / L)
        return (R + r * np.cos(theta)) * np.cos(phi), (R + r * np.cos(theta)) * np.sin(phi), r * np.sin(theta)

    def draw_frame(self, i):
        self.frame = int(i)
        for idx, art in enumerate(self.artists):
            data = self.datasets[idx]
            p, pr = data['positions'][self.frame], data['predator_positions'][self.frame]
            if art['type'] == '3d':
                bx, by, bz = self._to_torus_3d(p, art['L'])
                art['boids']._offsets3d = (bx, by, bz)
                px, py, pz = self._to_torus_3d(pr.reshape(-1, 2), art['L'])
                art['pred']._offsets3d = (px, py, pz)
            else:
                v = data['velocities'][self.frame]
                art['boids'].set_offsets(p); art['pred'].set_offsets(pr)
                art['quiver'].set_offsets(p); art['quiver'].set_UVC(v[:, 0], v[:, 1])
        self.fig.canvas.draw_idle()

    def tick(self, _):
        if self.playing:
            self.frame = (self.frame + 1) % self.time_steps
            self.slider_time.eventson = False
            self.slider_time.set_val(self.frame)
            self.slider_time.eventson = True
            self.draw_frame(self.frame)


class BoidVisualizer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_sims = len(datasets)
        self.time_steps = len(datasets[0]['positions'])
        
        # Initial state logic
        self.config = datasets[0].get('config',[{}]).item()
        self.is_torus_sim = self.config.get('COORD_SYSTEM','flat') == 'torus'
        self.torus_view = self.is_torus_sim  # Current toggle state
        print(self.is_torus_sim)
        
        self.playing = False
        self.frame = 0
        self.speed = 33
        
        self.fig = None
        self.ani = None
        self.artists = []
        self.overlay = False

    def _setup_layout(self):
        """Internal helper to build/rebuild the UI and Axes."""
        self.fig.clf()
        self.artists = []
        self.axs = []  # Initialize/Reset the axes list
        
        num_cols = 1 if self.overlay else self.num_sims
        cmap = plt.get_cmap("tab10")

        for i in range(self.num_sims):
            if self.overlay and i > 0:
                # Use the first axis created for all subsequent datasets
                ax = self.axs[0]
            else:
                proj = '3d' if self.torus_view else None
                ax = self.fig.add_subplot(1, num_cols, i + 1, projection=proj)
                
                if self.torus_view:
                    ax.set_box_aspect([1, 1, 1])
                    ax.axis('off')
                else:
                    ax.set_aspect("equal", adjustable="box")
                
                self.axs.append(ax) # Add to list so overlay can find it
            
            color = cmap(i % cmap.N)
            self.artists.append(self._create_artist(ax, self.datasets[i], color, self.torus_view))
        
        self._init_ui()
        self.draw_frame(self.frame)

    def _create_artist(self, ax, data, color, torus_view):
        sim_config = self.config
        show_pred = sim_config.get('PREDATOR', True)
        sim_width = data.get('sim_width', sim_config.get('SIM_WIDTH'))
        
        # Determine simulation title
        title_list = data.get('config_title', ["Simulation"])
        ax.set_title(title_list[0] if isinstance(title_list, list) else title_list)
        
        # Get initial data for stable initialization
        p0 = data['positions'][0]
        pr0 = data['predator_positions'][0]
        
        if torus_view:
            # 3D Torus Geometry
            R, r = 10, 4
            u, v_grid = np.mgrid[0:2*np.pi:25j, 0:2*np.pi:20j]
            xw = (R + r * np.cos(v_grid)) * np.cos(u)
            yw = (R + r * np.cos(v_grid)) * np.sin(u)
            zw = r * np.sin(v_grid)
            ax.plot_wireframe(xw, yw, zw, color="gray", alpha=0.1, linewidth=0.5)
            
            limit = R + r + 2
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
            
            boids = ax.scatter([], [], [], s=BOID_SIZE, color=color, alpha=0.8)
            pred = ax.scatter([], [], [], s=PRED_SIZE, color='red', marker='X', visible=show_pred)
            return {"type": "3d", "boids": boids, "pred": pred, "L": sim_width}
        else:
            # 2D View
            spawn_bounds = np.array(data.get('spawn_bounds', [-1.0, 1.0])).flatten()
            left, right = (0.0, float(sim_width)) if self.is_torus_sim else (float(spawn_bounds[0])/ZOOM, float(spawn_bounds[1])/ZOOM)
            
            ax.set_xlim(left, right); ax.set_ylim(left, right)
            
            v0 = data['velocities'][0]
            boids = ax.scatter(p0[:, 0], p0[:, 1], s=BOID_SIZE, color=color)
            pred = ax.scatter(pr0[0], pr0[1], s=PRED_SIZE, color='red', visible=show_pred)
            
            # Quiver initialized with p0/v0 to avoid the "Size 0" ValueError
            quiver = ax.quiver(p0[:, 0], p0[:, 1], v0[:, 0], v0[:, 1], 
                               color=color, alpha=0.4, scale=ARROW_SCALE, width=ARROW_WIDTH)
            
            return {"type": "2d", "boids": boids, "pred": pred, "quiver": quiver}
    def _init_ui(self):
        self.fig.subplots_adjust(bottom=0.25)
        
        # Slider
        ax_time = self.fig.add_axes([0.25, 0.12, 0.55, 0.03])
        self.slider_time = Slider(ax_time, "Time ", 0, self.time_steps - 1, valinit=self.frame, valstep=1)
        self.slider_time.on_changed(lambda val: self.draw_frame(int(val)) if not self.playing else None)
        
        # Play Button
        ax_play = self.fig.add_axes([0.1, 0.11, 0.1, 0.05])
        self.btn_play = Button(ax_play, "Pause" if self.playing else "Play")
        self.btn_play.on_clicked(self._toggle_playback)
        
        # Toggle View Button
        if self.is_torus_sim:
            ax_toggle = self.fig.add_axes([0.1, 0.04, 0.15, 0.05])
            self.btn_toggle = Button(ax_toggle, "Toggle 2D/3D")
            self.btn_toggle.on_clicked(self._handle_view_toggle)

    def _handle_view_toggle(self, event):
        self.torus_view = not self.torus_view
        self._setup_layout()
        self.fig.canvas.draw_idle()

    def draw_frame(self, frame_idx):
        self.frame = int(frame_idx)
        for art in self.artists:
            data = self.datasets[self.artists.index(art)]
            p = data['positions'][self.frame]
            pr = data['predator_positions'][self.frame]
            
            if art['type'] == '3d':
                bx, by, bz = self._to_torus_3d(p, art['L'])
                art['boids']._offsets3d = (bx, by, bz)
                px, py, pz = self._to_torus_3d(pr.reshape(-1, 2), art['L'])
                art['pred']._offsets3d = (px, py, pz)
            else:
                v = data['velocities'][self.frame]
                art['boids'].set_offsets(p)
                art['pred'].set_offsets(pr)
                art['quiver'].set_offsets(p)
                art['quiver'].set_UVC(v[:, 0], v[:, 1])

    def get_animation(self, overlay=False):
        self.overlay = overlay
        if DARK_MODE: plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 8) if not overlay else (8, 9))
        
        self._setup_layout()
        
        self.ani = FuncAnimation(self.fig, self._tick, interval=self.speed, cache_frame_data=False)
        return self.ani

    def get_frames(self, frame_indices, torus_view=None):
        """Film strip view."""
        t_view = torus_view if torus_view is not None else self.is_torus_sim
        num_f = len(frame_indices)
        fig, axes = plt.subplots(self.num_sims, num_f, figsize=(4*num_f, 4*self.num_sims),
                                 subplot_kw={'projection': '3d'} if t_view else None)
        # (Logic for drawing static frames remains the same as previous)
        return fig

    def _to_torus_3d(self, pos, L, R=10, r=4):
        phi, theta = 2 * np.pi * (pos[:, 0] / L), 2 * np.pi * (pos[:, 1] / L)
        return (R + r * np.cos(theta)) * np.cos(phi), (R + r * np.cos(theta)) * np.sin(phi), r * np.sin(theta)

    def _toggle_playback(self, event):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def _tick(self, _):
        if self.playing:
            self.frame = (self.frame + 1) % self.time_steps
            self.slider_time.eventson = False
            self.slider_time.set_val(self.frame)
            self.slider_time.eventson = True
            self.draw_frame(self.frame)