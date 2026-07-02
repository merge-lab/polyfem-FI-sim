import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd

import newton
import warp as wp
from pxr import Usd

import newton_builders
import utils_FI

dict_poke_idx_real2sim = {
    1: 2,
    2: 6,
    3: 7,
    4: 1,
    5: 5,
    6: 8,
    7: 3,
    8: 4,
    9: 9
}

class ThighpadPokeTestRealistic:
    def __init__(self, viewer, verbose=False):
        self.sim_time = 0.0
        self.sim_params = newton_builders.SimParams()
        self.sim_params.fps = 30

        self.sim_start_time = 0.0

        self.gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -self.sim_params.g), dtype=wp.vec3)

        self.verbose = verbose
        self.viewer  = viewer

        self.args = self.parse_args()

        # Load real poke trajectories before building the model (init_q depends on them)
        self._load_trajectories()

        # Starting joint position — taken from the first data point of poke 1's trajectory
        # self.init_q   = float(self.poke_trajs[0]["position_m"].iloc[0])
        self.init_q = -0.085
        self.current_q = self.init_q

        self.i_current_poke = 0

        init_qs = self.init_q * wp.ones(9, dtype=wp.float32)
        self.model = self.create_model(init_qs)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([-0.15, 0.0, 0.03]), -5, 0.0)
        self.viewer._cam_speed = 0.1

        self.solver_vbd = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.sim_params.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=self.sim_params.particle_self_contact_radius,
            particle_self_contact_margin=self.sim_params.particle_self_contact_margin,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.sim_params.rigid_contact_k_start,
            rigid_avbd_beta=self.sim_params.rigid_avbd_beta,
        )

        self.solver_rigid = None
        if not self.args["no_poke"]:
            self.solver_rigid = newton.solvers.SolverMuJoCo(self.model)
            # self.solver_rigid = newton.solvers.SolverFeatherstone(
            #     self.model,
            #     update_mass_matrix_interval=self.sim_params.sim_substeps
            # )

        self.state_now  = self.model.state()
        self.state_next = self.model.state()
        self.control    = self.model.control()

        self.control.joint_target_pos = init_qs

        self.contacts = self.model.contacts()

        self.graph = None
        self.capture()

        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")

        self.vol_initial   = -1
        self.log_sim_times: list[float] = []
        self.log_volumes:   list[float] = []
        self.log_pressures: list[float] = []
        self.log_pokes:     list[int]   = []

    # ── trajectory loading ────────────────────────────────────────────────────

    def _load_trajectories(self):
        data_dir         = Path("../Data/Gilbert/Thigh_pad/pokes/adjusted")
        self.poke_trajs = []
        poke_end_times = []
        for i in range(1, 10):
            df = pd.read_csv(data_dir / f"adjusted_newton_poke_{i}_1.csv")
            self.poke_trajs.append(df)
            start_time_i = df.iloc[-1]["time_s"]
            poke_end_times.append(start_time_i)

        self.poke_end_times = np.array(poke_end_times)

    # ── arg parsing ───────────────────────────────────────────────────────────

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-poke", action="store_true", help="Skip loading the poke fixture")
        parser.add_argument("--debug_particles", action="store_true", help="Show debug particles")
        parser.add_argument("--name", default="", type=str, help="Prefix to filename")
        return vars(parser.parse_args())

    # ── scene creation ────────────────────────────────────────────────────────

    def create_model(self, init_qs=None):
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = self.sim_params.particle_radius

        self.pad_start_particle_idx = len(self.scene.particle_q)
        self._fixed_particle_ids, dict_surf_select = newton_builders.create_pad(
            "../Assets/Thigh-pad/tets_fine/",
            self.scene,
            pos      = wp.vec3(0.0, 0.0, 0.0),
            rot      = wp.quat_identity(),
            rho      = self.sim_params.material_rho,
            k_mu     = self.sim_params.material_k_mu,
            k_lambda = self.sim_params.material_k_lambda,
            k_damp   = self.sim_params.material_k_damp,
        )
        id_channel = 2
        self.ids_channel = dict_surf_select[id_channel]

        if not self.args["no_poke"]:
            self.create_poker(self.scene, init_qs)

        self.scene.add_ground_plane()

        replicator = newton.ModelBuilder()
        replicator.replicate(self.scene, 1, spacing=(0.2, 0.2, 0.0))

        replicator.color()
        model = replicator.finalize()

        newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        model.soft_contact_ke = self.sim_params.soft_contact_ke
        model.soft_contact_kd = self.sim_params.soft_contact_kd
        model.soft_contact_mu = self.sim_params.soft_contact_mu

        self.setup_debug_viz(model)

        from newton._src.geometry.flags import ShapeFlags
        shape_flags_np = model.shape_flags.numpy()
        shape_labels = model.shape_label if hasattr(model, "shape_label") else [str(i) for i in range(len(shape_flags_np))]
        print(f"\n=== Shape flags diagnostic ({len(shape_flags_np)} shapes) ===")
        for i, (flags, label) in enumerate(zip(shape_flags_np, shape_labels)):
            visible           = bool(flags & ShapeFlags.VISIBLE)
            collide_shapes    = bool(flags & ShapeFlags.COLLIDE_SHAPES)
            collide_particles = bool(flags & ShapeFlags.COLLIDE_PARTICLES)
            print(f"  Shape {i} [{label}]: VISIBLE={visible}, COLLIDE_SHAPES={collide_shapes}, COLLIDE_PARTICLES={collide_particles}")
        print("=== End shape flags ===\n")

        return model

    def create_poker(self, scene, init_qs):
        builder_poke_fixture = newton.ModelBuilder()
        path_poker = "./usd/poke_fixture_FLATTENED.usd"
        builder_poke_fixture.add_usd(
            path_poker,
            xform=wp.transform((0.025117, -0.020982, 0.1), wp.quat_identity()),
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder_poke_fixture.joint_q = init_qs.list()
        scene.add_world(builder_poke_fixture)

    def setup_debug_viz(self, model):
        debug_radius = self.sim_params.particle_radius
        colors_np = np.tile([0.6, 0.6, 0.6], (model.particle_count, 1)).astype(np.float32)
        for idx in self._fixed_particle_ids:
            colors_np[idx] = [0.0, 0.0, 1.0]
        self.particle_debug_colors = wp.array(colors_np, dtype=wp.vec3)
        self.particle_debug_radii  = wp.full(model.particle_count, debug_radius, dtype=wp.float32)

    # ── CUDA graph capture ────────────────────────────────────────────────────

    def capture(self):
        assert wp.get_device().is_cuda, "No device available or device isn't CUDA"
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    # ── simulation step ───────────────────────────────────────────────────────

    def simulate(self):
        self.solver_vbd.rebuild_bvh(self.state_now)
        for _ in range(self.sim_params.sim_substeps):
            self.state_now.clear_forces()
            self.state_next.clear_forces()
            self.viewer.apply_forces(self.state_now)

            if not self.args["no_poke"]:
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)
                self.model.shape_contact_pair_count = 0

                self.solver_rigid.step(self.state_now, self.state_next, self.control, None, self.sim_params.sim_dt)

                self.state_now.particle_f.zero_()
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            self.model.collide(self.state_now, self.contacts)
            self.solver_vbd.step(self.state_now, self.state_next, self.control, self.contacts, self.sim_params.sim_dt)

            self.state_now, self.state_next = self.state_next, self.state_now
            self.sim_time += self.sim_params.sim_dt

    def step(self):
        if not self.args["no_poke"]:
            self._control_poker_from_trajectory()

        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.sim_params.frame_dt
        else:
            self.simulate()

        self._log_states()

    # ── trajectory-based poke controller ─────────────────────────────────────

    def _control_poker_from_trajectory(self):
        # All 9 pokes finished
        if self.i_current_poke > 8 and self.sim_time > self.poke_end_times[-1]:
            self.terminate()
            return

        # Advance poke index if the current window has expired
        if self.i_current_poke < 8 and self.sim_time > self.poke_end_times[self.i_current_poke]:
            self.i_current_poke += 1

        # Interpolate target z from the active poke's trajectory
        traj = self.poke_trajs[self.i_current_poke]
        target_q = float(np.interp(
            self.sim_time,
            traj["time_s"].values,
            traj["position_m"].values,
            left=float(traj["position_m"].iloc[0]),
            right=float(traj["position_m"].iloc[-1]),
        ))

        controller_dt = self.sim_params.frame_dt / self.sim_params.sim_substeps
        delta_q = target_q - self.current_q
        dq = delta_q / self.sim_params.frame_dt
        self.current_q = target_q

        # Active joint follows trajectory; all others stay at rest position
        # target_qs  = np.full(9, self.init_q, dtype=np.float32)
        # target_qds = np.zeros(9, dtype=np.float32)

        i_joint = dict_poke_idx_real2sim[self.i_current_poke + 1] - 1
        control_mask = np.zeros(9, dtype=np.float32)
        control_mask[i_joint] = 1
        control_mask = wp.array(control_mask)


        # target_qs[i_joint]  = target_q
        # target_qds[i_joint] = dq / self.sim_params.frame_dt

        target_qs = target_q * control_mask.view(wp.float32)
        target_qds = dq * control_mask.view(wp.float32)

        wp.copy(self.state_now.joint_q,  wp.array(target_qs,  dtype=wp.float32))
        wp.copy(self.state_now.joint_qd, wp.array(target_qds, dtype=wp.float32))

        print(f"Target q: {target_q}")

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.sim_start_time = time.time()
        while self.viewer.is_running():
            if not self.viewer.is_paused():
                self.step()
            self.render()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_now)
        self.viewer.log_contacts(self.contacts, self.state_now)
        if self.args["debug_particles"]:
            self.viewer.log_points(
                name="/debug/fixed_particles",
                points=self.state_now.particle_q,
                radii=self.particle_debug_radii,
                colors=self.particle_debug_colors,
            )
        self.viewer.end_frame()

    # ── logging ───────────────────────────────────────────────────────────────

    def _log_states(self):
        tube_diameter = 0.0254 / 16
        tube_length   = 576.22 / 1e3
        tube_vol      = (wp.PI * (tube_diameter / 2) ** 2) * tube_length

        t_first_poke_start = self.poke_trajs[0].loc[0, "time_s"]
        if self.sim_time >= t_first_poke_start:
            idxs           = self.pad_start_particle_idx + self.ids_channel
            verts          = self.state_now.particle_q[self.pad_start_particle_idx:]
            channel_volume = utils_FI.compute_volume_mesh(verts, idxs)
            self.log_volumes.append(channel_volume)

            if self.vol_initial == -1:
                self.vol_initial = channel_volume

            p_now = 1 * (self.vol_initial + tube_vol) / (channel_volume + tube_vol)
            self.log_pressures.append(p_now)
        else:
            self.log_volumes.append(0.0)
            self.log_pressures.append(0.0)

        self.log_sim_times.append(self.sim_time)
        self.log_pokes.append(self.i_current_poke)

    # ── GUI ───────────────────────────────────────────────────────────────────

    def gui(self, ui):
        ui.text(f"Wall time: {time.time() - self.sim_start_time:.1f}s")
        ui.text(f"Sim time:  {self.sim_time:.3f}s")
        ui.text(f"Active poke: {self.i_current_poke + 1} / 9")
        ui.text(f"Latest volume [cm^3]: {self.log_volumes[-1] * 100**3:.4f}" if self.log_volumes else "")
        ui.separator()

        w = 250
        graph_size = ui.ImVec2(-1, 80)

        def padded(data):
            arr = np.array(data[-w:], dtype=np.float32)
            if len(arr) < w:
                arr = np.pad(arr, (w - len(arr), 0), mode="edge")
            return arr

        ui.text("Channel volume")
        ui.plot_lines("##iters", padded(self.log_volumes), graph_size=graph_size)

    # ── termination ───────────────────────────────────────────────────────────

    def terminate(self):
        print(
            f"End of poke test reached at sim time {self.sim_time:.3f}s, "
            f"elapsed wall time {time.time() - self.sim_start_time:.2f}s. Terminating sim"
        )
        self.viewer.close()

        now_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        df_out = pd.DataFrame({
            "sim_times_s":   self.log_sim_times,
            "volumes_m3":    self.log_volumes,
            "pressures_atm": self.log_pressures,
            "i_poke":        self.log_pokes,
        })
        df_out.to_csv(f"./logs/{self.args['name']}sim-outputs-realistic_{now_str}.csv")


if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL(headless=False)
    sim    = ThighpadPokeTestRealistic(viewer, verbose=False)
    sim.run()
    viewer.close()
