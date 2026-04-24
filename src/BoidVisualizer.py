import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

DARK_MODE = False
ZOOM = 0.1
BOID_SIZE = 50*ZOOM
PRED_SIZE = 100*ZOOM
ARROW_SCALE = 20/(0.5*ZOOM)
ARROW_WIDTH = 0.05 * ZOOM


class BoidVisualiser:

    def __init__(self, datasets, predator_trail=10, boid_trail=0):
        self.datasets = datasets
        self.num_sims = len(datasets)
        self.simulation_steps = len(datasets[0]['positions'])
        
        self.predator_trail_len = predator_trail
        self.boid_trail_len = boid_trail
        
        config = datasets[0].get('config', np.array([{}]) )
        if isinstance(config,np.ndarray):
            self.config = config.item()
        else:
            self.config = config

        print()

        self.is_torus_sim = self.config.get('coord_system','flat') == 'torus'
        self.torus_view = self.is_torus_sim 
        
        self.playing = False
        self.frame = 0
        self.speed = 33
        
        self.fig = None
        self.ani = None
        self.artists = []
        self.overlay = False

    def _setup_layout(self):
        self.fig.clf()
        self.artists = []
        self.axs = []
        
        num_cols = 1 if self.overlay else self.num_sims
        cmap = plt.get_cmap("tab10")

        for i in range(self.num_sims):
            if self.overlay and i > 0:
                ax = self.axs[0]
            else:
                proj = '3d' if self.torus_view else None
                ax = self.fig.add_subplot(1, num_cols, i + 1, projection=proj)
                
                if self.torus_view:
                    ax.set_box_aspect([1, 1, 1])
                    ax.axis('off')
                else:
                    ax.set_aspect("equal", adjustable="box")
                
                self.axs.append(ax)
            
            color = cmap(i % cmap.N)
            self.artists.append(self._create_artist(ax, self.datasets[i], color, self.torus_view))
        
        self._init_ui()
        self.draw_frame(self.frame)

    def _create_artist(self, ax, data, color, torus_view):
        sim_config = self.config
        show_pred = sim_config.get('predator', True)
        sim_width = data.get('sim_width', sim_config.get('sim_width'))
        
        p0 = data['positions'][0]
        pr0 = data['predator_positions'][0]
        
        # Predator trail (Red)
        if not torus_view:
            pred_line, = ax.plot([], [], color='red', alpha=0.5, linewidth=1.5, zorder=5)
        else:
            pred_line, = ax.plot([], [], [], color='red', alpha=0.5, linewidth=1.5)

        # Boid trails
        boid_lines = []
        if self.boid_trail_len > 0:
            for _ in range(len(p0)):
                if not torus_view:
                    ln, = ax.plot([], [], color=color, alpha=0.2, linewidth=0.5)
                else:
                    ln, = ax.plot([], [], [], color=color, alpha=0.2, linewidth=0.5)
                boid_lines.append(ln)

        if torus_view:
            # Torus Mesh Logic
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
            
            return {"type": "3d", "boids": boids, "pred": pred, "pred_line": pred_line, "boid_lines": boid_lines, "L": sim_width}
        else:
            spawn_bounds = np.array(data.get('spawn_bounds', [-1.0, 1.0])).flatten()
            left, right = (0.0, float(sim_width)) if self.is_torus_sim else (float(spawn_bounds[0])/ZOOM, float(spawn_bounds[1])/ZOOM)
            ax.set_xlim(left, right); ax.set_ylim(left, right)
            
            v0 = data['velocities'][0]
            boids = ax.scatter(p0[:, 0], p0[:, 1], s=BOID_SIZE, color=color)
            pred = ax.scatter(pr0[0], pr0[1], s=PRED_SIZE, color='red', visible=show_pred)
            quiver = ax.quiver(p0[:, 0], p0[:, 1], v0[:, 0], v0[:, 1], color=color, alpha=0.4, scale=ARROW_SCALE, width=ARROW_WIDTH)
            
            return {"type": "2d", "boids": boids, "pred": pred, "quiver": quiver, "pred_line": pred_line, "boid_lines": boid_lines}
        
    def _init_ui(self):
        self.fig.subplots_adjust(bottom=0.25)
        
        # Slider
        ax_time = self.fig.add_axes([0.25, 0.12, 0.55, 0.03])
        self.slider_time = Slider(ax_time, "Time ", 0, self.simulation_steps - 1, valinit=self.frame, valstep=1)
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
        for i, art in enumerate(self.artists):
            data = self.datasets[i]
            
            p = data['positions'][self.frame]
            pr = data['predator_positions'][self.frame]
            
            start_p = max(0, self.frame - self.predator_trail_len)
            pr_history = data['predator_positions'][start_p : self.frame + 1]
            
            start_b = max(0, self.frame - self.boid_trail_len)
            b_history = data['positions'][start_b : self.frame + 1]

            if art['type'] == '3d':
                # Predator Position & Trail
                bx, by, bz = self._to_torus_3d(p, art['L'])
                art['boids']._offsets3d = (bx, by, bz)
                
                px, py, pz = self._to_torus_3d(pr.reshape(-1, 2), art['L'])
                art['pred']._offsets3d = (px, py, pz)
                
                # Update Predator Line
                lx, ly, lz = self._to_torus_3d(pr_history.reshape(-1, 2), art['L'])
                art['pred_line'].set_data(lx, ly)
                art['pred_line'].set_3d_properties(lz)
                
                # Update Boid Lines (if enabled)
                for b_idx, ln in enumerate(art['boid_lines']):
                    hx, hy, hz = self._to_torus_3d(b_history[:, b_idx, :], art['L'])
                    ln.set_data(hx, hy)
                    ln.set_3d_properties(hz)
            else:
                # 2D Updates
                v = data['velocities'][self.frame]
                art['boids'].set_offsets(p)
                art['pred'].set_offsets(pr)
                art['quiver'].set_offsets(p)
                art['quiver'].set_UVC(v[:, 0], v[:, 1])
                
                # Update Predator Line
                art['pred_line'].set_data(pr_history[:, 0], pr_history[:, 1])
                
                # Update Boid Lines (if enabled)
                for b_idx, ln in enumerate(art['boid_lines']):
                    ln.set_data(b_history[:, b_idx, 0], b_history[:, b_idx, 1])

    def get_animation(self, overlay=False):
        self.overlay = overlay
        if DARK_MODE: plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 8) if not overlay else (8, 9))
        
        self._setup_layout()
        
        self.ani = FuncAnimation(self.fig, self._tick, interval=self.speed, cache_frame_data=False)
        return self.ani

    def get_frames(self, frame_indexes, max_columns, overlay=True, trails=0):
        """
        Displays specified frames in a 'roll of film' grid.
        trails: int, number of previous steps to draw.
        """
        frame_indexes = list(frame_indexes)
        num_frames = len(frame_indexes)
        num_rows = (num_frames + max_columns - 1) // max_columns
        proj = '3d' if self.torus_view else None
        
        fig_film, axs_film = plt.subplots(
            num_rows, max_columns, 
            figsize=(max_columns * 3, num_rows * 3),
            subplot_kw={'projection': proj},
            constrained_layout=True
        )
        
        axs_flat = np.atleast_1d(axs_film).flatten()
        original_artists = self.artists
        cmap = plt.get_cmap("tab10")

        for idx, frame_idx in enumerate(frame_indexes):
            ax = axs_flat[idx]
            ax.set_title(f"t={frame_idx}")
            current_frame_artists = []
            sims_to_draw = self.num_sims if overlay else 1
            
            for i in range(sims_to_draw):
                color = cmap(i % cmap.N)
                data = self.datasets[i]
                
                if trails > 0:
                    start_trail = max(0, frame_idx - trails)
                    history = data['positions'][start_trail : frame_idx + 1]
                    
                    if len(history) > 1:
                        for b_idx in range(history.shape[1]):
                            boid_history = history[:, b_idx, :]
                            
                            if self.torus_view:
                                tx, ty, tz = self._to_torus_3d(boid_history, data.get('sim_width', self.config.get('sim_width')))
                                ax.plot(tx, ty, tz, color=color, alpha=0.3, linewidth=0.8)
                            else:
                                ax.plot(boid_history[:, 0], boid_history[:, 1], color=color, alpha=0.3, linewidth=0.8)

                # Use existing logic for the main boid bodies
                artist = self._create_artist(ax, data, color, self.torus_view)
                current_frame_artists.append(artist)

            self.artists = current_frame_artists
            self.draw_frame(frame_idx)
            
            if not self.torus_view:
                ax.set_xticks([]); ax.set_yticks([])

        for j in range(len(frame_indexes), len(axs_flat)):
            axs_flat[j].axis('off')

        self.artists = original_artists
        return fig_film
    
    def _to_torus_3d(self, pos, L, R=10, r=4):
        phi, theta = 2 * np.pi * (pos[:, 0] / L), 2 * np.pi * (pos[:, 1] / L)
        return (R + r * np.cos(theta)) * np.cos(phi), (R + r * np.cos(theta)) * np.sin(phi), r * np.sin(theta)

    def _toggle_playback(self, event):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def _tick(self, _):
        if self.playing:
            self.frame = (self.frame + 1) % self.simulation_steps
            self.slider_time.eventson = False
            self.slider_time.set_val(self.frame)
            self.slider_time.eventson = True
            self.draw_frame(self.frame)