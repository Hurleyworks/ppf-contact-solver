// File: crates/ppf-cts-capi/src/context.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `CtsContext` owns the scene state across the whole embedding lifecycle:
// params/material/geometry/pins are staged before `initialize`, then the
// GPU-backed `Backend`/`DataSet`/`ParamSet` are held for the per-step loop.
// The lifecycle mirrors `backend::Backend::run` minus all file IO: stage
// inputs -> `initialize` (one GPU allocation) -> `step` repeatedly (each
// rebuilds the pin constraint and advances) -> `read_positions`.

use nalgebra::Matrix3xX;

use ppf_cts_solver::backend::Backend;
use ppf_cts_solver::builder::{self, make_collision_mesh};
use ppf_cts_solver::data::{
    CollisionMesh, Constraint, DataSet, FaceParam, FaceProp, FixPair, ParamSet, PullPair, Vec3f,
    PDRD_BODY_ROW_LEN, ROTATION_LOCK_ALLOW_ONLY, TRANSLATION_LOCK_AXIS,
};
use ppf_cts_solver::triutils;
use ppf_cts_solver::{CVec, SimArgs};

use crate::error::{set_last_error, CtsStatus};
use crate::ingest;
use crate::params::{CtsColliderMaterial, CtsGlobalParams, CtsMaterial, CtsModel};

/// Geometry as staged by `set_geometry_ex`, in the caller's indexing. The
/// `MeshSet` is built at `initialize`, after animated-collider shells are
/// spliced in (`ingest::insert_shells`).
struct StagedGeometry {
    verts: Vec<f32>,
    faces: Vec<u32>,
    tets: Vec<u32>,
    rods: Vec<u32>,
    shell_face_count: usize,
}

/// An animated collider: a pin shell in the solved namespace, every vertex an
/// exact kinematic pin driven by TRS keyframes (the frontend's
/// `_setup_pin_shell` + `PinOperation::TransformKeyframes`).
struct AnimatedCollider {
    /// Local-space vertices, `3*n`.
    local: Vec<f32>,
    /// Triangles in local indices.
    faces: Vec<u32>,
    times: Vec<f64>,
    translations: Vec<[f64; 3]>,
    /// `[w, x, y, z]`, normalized.
    quaternions: Vec<[f64; 4]>,
    scales: Vec<[f64; 3]>,
    /// Segment interpolation codes (all 0 = linear) and Bezier handles
    /// (unused for linear, but the shared evaluator indexes them).
    interps: Vec<u8>,
    handles: Vec<[f64; 4]>,
    material: CtsColliderMaterial,
    /// Global index of its first vertex, set at `initialize`.
    first: usize,
}

impl AnimatedCollider {
    fn vertex_count(&self) -> usize {
        self.local.len() / 3
    }

    /// World position of local vertex `j` at sim time `time`:
    /// `R(q) * (S .* local) + T`, linear T/S, slerp q, held outside the keys.
    /// Same evaluator the solver's `TransformKeyframes` pin uses.
    fn pose(&self, j: usize, time: f64) -> Vec3f {
        let l = [
            f64::from(self.local[3 * j]),
            f64::from(self.local[3 * j + 1]),
            f64::from(self.local[3 * j + 2]),
        ];
        let p = ppf_cts_core::datamodel::quat::transform_keyframes_step(
            l,
            &self.times,
            &self.translations,
            &self.quaternions,
            &self.scales,
            &self.interps,
            &self.handles,
            [0.0; 3],
            time,
        );
        Vec3f::new(p[0] as f32, p[1] as f32, p[2] as f32)
    }

    fn posed_verts(&self, time: f64) -> Vec<f32> {
        (0..self.vertex_count())
            .flat_map(|j| {
                let p = self.pose(j, time);
                [p[0], p[1], p[2]]
            })
            .collect()
    }

    /// The pin-shell material: the frontend sets density, Young's modulus,
    /// Poisson's ratio and bend to 0; contact gap and friction come from the
    /// collider row; contact offset 0 as for a static collider.
    fn shell_material(&self) -> CtsMaterial {
        CtsMaterial {
            model: CtsModel::BaraffWitkin,
            young_modulus: 0.0,
            poisson_ratio: 0.0,
            density: 0.0,
            bend: 0.0,
            friction: self.material.friction,
            strainlimit: 0.0,
            contact_gap: self.material.contact_gap,
            contact_offset: 0.0,
        }
    }
}

/// Static collider geometry kept in CPU form. `CollisionMesh` is not
/// `Clone` and is consumed by `Constraint`, so the per-step constraint
/// rebuild reconstructs it each step from these pieces (see
/// [`CtsContext::build_constraint`]).
struct Collider {
    vert: Matrix3xX<f32>,
    face: Matrix3xX<usize>,
    face_props: Vec<FaceProp>,
    face_params: Vec<FaceParam>,
}

/// The opaque handle behind the C API. All staging setters are valid
/// before `initialize`; after it the GPU state (`backend`/`dataset`/
/// `param`) is live and `step`/`read_positions` operate on it.
pub struct CtsContext {
    sim_args: SimArgs,
    /// Per-element material table (at least one row after `set_geometry_ex`).
    materials: Vec<CtsMaterial>,
    /// Per-face/tet/rod material-row selector. Empty => uniform (row 0 for
    /// every element of that kind).
    face_mat_idx: Vec<u32>,
    tet_mat_idx: Vec<u32>,
    rod_mat_idx: Vec<u32>,
    /// Staged before init; the `MeshSet` is built and moved into `backend`
    /// on `initialize`.
    geometry: Option<StagedGeometry>,
    /// Caller pin indices; remapped to the merged vertex order at
    /// `initialize` (only tet-interior indices move).
    pin_indices: Vec<u32>,
    pin_stiffness: Vec<f32>,
    /// Per-pin pull weight, aligned with `pin_indices`: `> 0` => a soft
    /// `PullPair` of that weight toward the pin's position, else an exact
    /// `FixPair` (as scene.rs `make_constraint`). Empty => every pin is fixed.
    pin_pull: Vec<f32>,
    /// Per-pin target: `Some` => kinematic (driven to the target),
    /// `None` => fixed at its rest position. Aligned with `pin_indices`.
    pin_targets: Vec<Option<Vec3f>>,
    /// Where each kinematic pin starts its current frame segment (`None` =>
    /// its rest position). A pin moves linearly from here to its target over
    /// `pin_segment` in sim time, so the frame's motion is spread over that
    /// frame's steps instead of jumping on the first one.
    pin_from: Vec<Option<Vec3f>>,
    /// Sim-time span `(t0, t1)` of the current pin segment; `t1 <= t0` means
    /// the target holds at once.
    pin_segment: (f64, f64),
    /// Gravity pushed by `set_gravity` after initialize: `(from, to)`, blended
    /// across `gravity_segment` like a pin and written into `param.gravity`
    /// at each step start (scene.rs `update_param` samples its gravity
    /// schedule there too). `None` => `param.gravity` as built.
    gravity_blend: Option<(Vec3f, Vec3f)>,
    gravity_segment: (f64, f64),
    collider: Option<Collider>,
    animated_colliders: Vec<AnimatedCollider>,
    /// Initial velocity (m/s), `3*n` in the caller's staged vertex order;
    /// empty => every vertex starts at rest. Remapped to the merged order at
    /// `initialize` (shell vertices start at rest).
    velocities: Vec<f32>,
    /// Granular (SAND) scalar material, `Some` once `set_sand` is called.
    /// Threaded into `Props.sand` at `initialize`; grains are loose
    /// vertices (no incident element) in the staged geometry.
    sand: Option<builder::SandParams>,
    /// PDRD (rigid-body) scene inputs, staged verbatim and handed to
    /// `builder::build` as a `PdrdSceneData` at `initialize`. Empty vecs
    /// mean "no PDRD bodies" and the build falls through to the parity
    /// path. Layout mirrors `scene.rs`'s `pdrd_*` fields:
    /// - `pdrd_body_rows`: `PDRD_BODY_ROW_LEN` (23) f32 per body.
    /// - `pdrd_vert_index`: per-global-vertex 1-based body id (0 = not PDRD).
    /// - `pdrd_vert_list`: flat global vertex indices, body-major.
    /// - `pdrd_rest_centered`: 3 f32 per `pdrd_vert_list` entry.
    pdrd_body_rows: Vec<f32>,
    pdrd_vert_index: Vec<u32>,
    pdrd_vert_list: Vec<u32>,
    pdrd_rest_centered: Vec<f32>,
    backend: Option<Backend>,
    dataset: Option<DataSet>,
    param: Option<ParamSet>,
    surface_vert_count: u32,
    initialized: bool,
}

impl CtsContext {
    pub fn new() -> Self {
        Self {
            sim_args: CtsGlobalParams::default().to_sim_args(),
            materials: vec![CtsMaterial::default()],
            face_mat_idx: Vec::new(),
            tet_mat_idx: Vec::new(),
            rod_mat_idx: Vec::new(),
            geometry: None,
            pin_indices: Vec::new(),
            pin_stiffness: Vec::new(),
            pin_pull: Vec::new(),
            pin_targets: Vec::new(),
            pin_from: Vec::new(),
            pin_segment: (0.0, 0.0),
            gravity_blend: None,
            gravity_segment: (0.0, 0.0),
            collider: None,
            animated_colliders: Vec::new(),
            velocities: Vec::new(),
            sand: None,
            pdrd_body_rows: Vec::new(),
            pdrd_vert_index: Vec::new(),
            pdrd_vert_list: Vec::new(),
            pdrd_rest_centered: Vec::new(),
            backend: None,
            dataset: None,
            param: None,
            surface_vert_count: 0,
            initialized: false,
        }
    }

    pub fn set_params(&mut self, params: &CtsGlobalParams) -> CtsStatus {
        if self.initialized {
            set_last_error("set_params called after initialize");
            return CtsStatus::BadState;
        }
        self.sim_args = params.to_sim_args();
        CtsStatus::Ok
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_geometry_ex(
        &mut self,
        verts: &[f32],
        faces: &[u32],
        tets: &[u32],
        rods: &[u32],
        shell_face_count: usize,
        materials: &[CtsMaterial],
        face_mat_idx: Option<&[u32]>,
        tet_mat_idx: Option<&[u32]>,
        rod_mat_idx: Option<&[u32]>,
    ) -> CtsStatus {
        if self.initialized {
            set_last_error("set_geometry called after initialize");
            return CtsStatus::BadState;
        }
        if materials.is_empty() {
            set_last_error("set_geometry_ex: material table is empty");
            return CtsStatus::BadArg;
        }
        let face_count = faces.len() / 3;
        let tet_count = tets.len() / 4;
        let rod_count = rods.len() / 2;
        let n_mat = materials.len();
        // Each provided index array must be one entry per element of its kind,
        // and every entry must select a valid material row.
        let check = |idx: Option<&[u32]>, count: usize, kind: &str| -> Result<(), ()> {
            if let Some(a) = idx {
                if a.len() != count {
                    set_last_error(format!(
                        "set_geometry_ex: {kind}_material_idx length != {kind} count"
                    ));
                    return Err(());
                }
                if a.iter().any(|&m| m as usize >= n_mat) {
                    set_last_error(format!(
                        "set_geometry_ex: {kind}_material_idx entry >= material_count"
                    ));
                    return Err(());
                }
            }
            Ok(())
        };
        if check(face_mat_idx, face_count, "face").is_err()
            || check(tet_mat_idx, tet_count, "tet").is_err()
            || check(rod_mat_idx, rod_count, "rod").is_err()
        {
            return CtsStatus::BadArg;
        }
        // Tets run only the ARAP/StVK/SNHk kernels (Baraff-Witkin on a tet hits
        // a kernel assert), and need the ranges scene.rs `make_props` asserts.
        if tet_count > 0 {
            let mut rows: Vec<usize> = match tet_mat_idx {
                Some(a) => a.iter().map(|&m| m as usize).collect(),
                None => vec![0],
            };
            rows.sort_unstable();
            rows.dedup();
            for row in rows {
                if let Err(msg) = check_tet_material(&materials[row]) {
                    set_last_error(format!("set_geometry_ex: tet material row {row}: {msg}"));
                    return CtsStatus::BadArg;
                }
            }
        }
        self.velocities.clear();
        self.materials = materials.to_vec();
        self.face_mat_idx = face_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.tet_mat_idx = tet_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.rod_mat_idx = rod_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.geometry = Some(StagedGeometry {
            verts: verts.to_vec(),
            faces: faces.to_vec(),
            tets: tets.to_vec(),
            rods: rods.to_vec(),
            shell_face_count: shell_face_count.min(face_count),
        });
        CtsStatus::Ok
    }

    /// Initial velocity per staged vertex (`3*n`, m/s), in `set_geometry_ex`
    /// vertex order. Must follow `set_geometry_ex`, which clears it.
    pub fn set_velocities(&mut self, velocities: &[f32]) -> CtsStatus {
        if self.initialized {
            set_last_error("set_velocities called after initialize");
            return CtsStatus::BadState;
        }
        let Some(geometry) = self.geometry.as_ref() else {
            set_last_error("set_velocities before set_geometry");
            return CtsStatus::BadState;
        };
        if velocities.len() != geometry.verts.len() {
            set_last_error("set_velocities: vertex_count != set_geometry vertex count");
            return CtsStatus::BadArg;
        }
        if !velocities.iter().all(|v| v.is_finite()) {
            set_last_error("set_velocities: non-finite component");
            return CtsStatus::BadArg;
        }
        self.velocities = velocities.to_vec();
        CtsStatus::Ok
    }

    /// Stage one animated collider (see `ppf_cts_add_animated_collider`).
    /// `times` strictly ascending; `rotations` are `[w, x, y, z]` and are
    /// normalized here. Array lengths are the caller's (lib.rs) job.
    #[allow(clippy::too_many_arguments)]
    pub fn add_animated_collider(
        &mut self,
        local_verts: &[f32],
        faces: &[u32],
        times: &[f64],
        translations: &[f64],
        rotations: &[f64],
        scales: &[f64],
        material: CtsColliderMaterial,
    ) -> CtsStatus {
        if self.initialized {
            set_last_error("add_animated_collider called after initialize");
            return CtsStatus::BadState;
        }
        let n_vert = local_verts.len() / 3;
        let n_key = times.len();
        if n_vert == 0 || faces.is_empty() || n_key == 0 {
            set_last_error("add_animated_collider: empty vertices, faces or keys");
            return CtsStatus::BadArg;
        }
        if translations.len() != 3 * n_key || rotations.len() != 4 * n_key || scales.len() != 3 * n_key {
            set_last_error("add_animated_collider: key array length mismatch");
            return CtsStatus::BadArg;
        }
        if faces.iter().any(|&f| f as usize >= n_vert) {
            set_last_error("add_animated_collider: face index >= vertex_count");
            return CtsStatus::BadArg;
        }
        if !times.iter().all(|t| t.is_finite()) || times.windows(2).any(|w| w[1] <= w[0]) {
            set_last_error("add_animated_collider: key times must be finite and strictly ascending");
            return CtsStatus::BadArg;
        }
        if !(material.contact_gap > 0.0) || material.friction < 0.0 {
            set_last_error("add_animated_collider: contact_gap must be positive, friction non-negative");
            return CtsStatus::BadArg;
        }
        let mut quaternions = Vec::with_capacity(n_key);
        for q in rotations.chunks_exact(4) {
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if !(norm > 0.0) || !norm.is_finite() {
                set_last_error("add_animated_collider: zero or non-finite rotation quaternion");
                return CtsStatus::BadArg;
            }
            quaternions.push([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]);
        }
        let vec3 = |a: &[f64]| -> Vec<[f64; 3]> { a.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
        let segments = n_key - 1;
        self.animated_colliders.push(AnimatedCollider {
            local: local_verts.to_vec(),
            faces: faces.to_vec(),
            times: times.to_vec(),
            translations: vec3(translations),
            quaternions,
            scales: vec3(scales),
            interps: vec![0; segments],
            handles: vec![[0.0; 4]; segments],
            material,
            first: 0,
        });
        CtsStatus::Ok
    }

    pub fn set_collider(
        &mut self,
        verts: &[f32],
        faces: &[u32],
        materials: &[CtsColliderMaterial],
        face_mat_idx: Option<&[u32]>,
    ) -> CtsStatus {
        if self.initialized {
            set_last_error("set_collider called after initialize");
            return CtsStatus::BadState;
        }
        if materials.is_empty() {
            set_last_error("set_collider_ex: material table is empty");
            return CtsStatus::BadArg;
        }
        let face_count = faces.len() / 3;
        if let Some(a) = face_mat_idx {
            if a.len() != face_count {
                set_last_error("set_collider_ex: face_material_idx length != face count");
                return CtsStatus::BadArg;
            }
            if a.iter().any(|&m| m as usize >= materials.len()) {
                set_last_error("set_collider_ex: face_material_idx entry >= material_count");
                return CtsStatus::BadArg;
            }
        }
        let (vert, face, face_props, face_params) =
            ingest::build_collider(verts, faces, materials, face_mat_idx);
        self.collider = Some(Collider {
            vert,
            face,
            face_props,
            face_params,
        });
        CtsStatus::Ok
    }

    /// Stage SAND granular material. Every grain (a loose vertex with no
    /// incident element in the staged geometry) shares this mass / contact
    /// radius / gap / friction. Validation mirrors `scene::sand_params`.
    pub fn set_sand(
        &mut self,
        particle_mass: f32,
        grain_radius: f32,
        contact_gap: f32,
        friction: f32,
    ) -> CtsStatus {
        if self.initialized {
            set_last_error("set_sand called after initialize");
            return CtsStatus::BadState;
        }
        if !(particle_mass > 0.0) {
            set_last_error("set_sand: particle_mass must be positive");
            return CtsStatus::BadArg;
        }
        if grain_radius < 0.0 {
            set_last_error("set_sand: grain_radius must be non-negative");
            return CtsStatus::BadArg;
        }
        if !(contact_gap > 0.0) {
            set_last_error("set_sand: contact_gap must be positive");
            return CtsStatus::BadArg;
        }
        if friction < 0.0 {
            set_last_error("set_sand: friction must be non-negative");
            return CtsStatus::BadArg;
        }
        self.sand = Some(builder::SandParams {
            particle_mass,
            grain_radius,
            contact_gap,
            friction,
        });
        CtsStatus::Ok
    }

    /// Stage PDRD (rigid-body) inputs verbatim. The four arrays mirror
    /// `builder::PdrdSceneData` (and `scene.rs`'s precomputed `pdrd_*`
    /// files): `body_rows` is `PDRD_BODY_ROW_LEN` floats per body,
    /// `vert_index` is per-global-vertex (1-based body id, 0 = not PDRD),
    /// `vert_list` is flat global indices body-major, `rest_centered` is
    /// 3 floats per `vert_list` entry. The per-body rows are precomputed by
    /// the caller (the addon/plugin), exactly as the Python frontend packs
    /// `pdrd_body.bin`. Length cross-checks against the geometry are
    /// deferred to `initialize`.
    pub fn set_rigidbodies(
        &mut self,
        body_rows: &[f32],
        vert_index: &[u32],
        vert_list: &[u32],
        rest_centered: &[f32],
    ) -> CtsStatus {
        if self.initialized {
            set_last_error("set_rigidbodies called after initialize");
            return CtsStatus::BadState;
        }
        if body_rows.len() % PDRD_BODY_ROW_LEN != 0 {
            set_last_error("set_rigidbodies: body_rows length not a multiple of PDRD_BODY_ROW_LEN");
            return CtsStatus::BadArg;
        }
        if rest_centered.len() != 3 * vert_list.len() {
            set_last_error("set_rigidbodies: rest_centered length != 3 * vert_list length");
            return CtsStatus::BadArg;
        }
        self.pdrd_body_rows = body_rows.to_vec();
        self.pdrd_vert_index = vert_index.to_vec();
        self.pdrd_vert_list = vert_list.to_vec();
        self.pdrd_rest_centered = rest_centered.to_vec();
        CtsStatus::Ok
    }

    pub fn set_pins(&mut self, indices: &[u32], stiffness: Option<&[f32]>) -> CtsStatus {
        if self.initialized {
            set_last_error("set_pins called after initialize");
            return CtsStatus::BadState;
        }
        self.pin_indices = indices.to_vec();
        self.pin_stiffness = match stiffness {
            Some(s) if s.len() == indices.len() => s.to_vec(),
            Some(_) => {
                set_last_error("set_pins: stiffness length != index count");
                return CtsStatus::BadArg;
            }
            None => vec![1.0; indices.len()],
        };
        // Default every pin to fixed at rest until a target arrives.
        self.pin_pull = Vec::new();
        self.pin_targets = vec![None; indices.len()];
        self.pin_from = vec![None; indices.len()];
        self.pin_segment = (0.0, 0.0);
        CtsStatus::Ok
    }

    /// Per-pin pull weights, one per pin in `set_pins` order. A weight `> 0`
    /// makes that pin a soft `PullPair` of that weight toward its (moving)
    /// target; `<= 0` keeps it an exact `FixPair`. The solver assigns each
    /// vertex its fix/pull slot when it builds the scene, so this must come
    /// before `initialize` (and after `set_pins`, which resets it).
    pub fn set_pin_pull(&mut self, weights: &[f32]) -> CtsStatus {
        if self.initialized {
            set_last_error("set_pin_pull called after initialize");
            return CtsStatus::BadState;
        }
        if weights.len() != self.pin_indices.len() {
            set_last_error("set_pin_pull: weight count != pin count");
            return CtsStatus::BadArg;
        }
        self.pin_pull = weights.to_vec();
        CtsStatus::Ok
    }

    /// Push the targets for the NEXT frame boundary. After `initialize`, each
    /// pin moves linearly in sim time from where it is now to its new target,
    /// reaching it at the next multiple of `1/fps` (the `fps` of
    /// `set_params`), and holds there after. The frame's motion is spread over
    /// that frame's steps, and every step hands the solver the exact travel
    /// it covers (`step_delta`). Before `initialize` the targets apply at once
    /// (the pins' starting pose, as before ABI 8).
    pub fn update_pin_targets(&mut self, positions: &[f32]) -> CtsStatus {
        let count = self.pin_indices.len();
        if positions.len() != 3 * count {
            set_last_error("update_pin_targets: positions length != 3 * pin count");
            return CtsStatus::BadArg;
        }
        if self.initialized {
            let backend = self.backend.as_ref().unwrap();
            let now = backend.state.time;
            let from: Vec<Option<Vec3f>> = (0..count)
                .map(|i| Some(self.pin_position(i, now, &backend.mesh.vertex)))
                .collect();
            self.pin_from = from;
            self.pin_segment = (now, next_frame_time(now, self.sim_args.fps));
        } else {
            self.pin_segment = (0.0, 0.0);
        }
        for i in 0..count {
            let target = Vec3f::new(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
            self.pin_targets[i] = Some(target);
            if !self.initialized {
                self.pin_from[i] = Some(target);
            }
        }
        CtsStatus::Ok
    }

    /// Set gravity for the NEXT frame boundary. After `initialize` it blends
    /// linearly in sim time from its current value to `gravity`, reached at
    /// the next multiple of `1/fps`, as `update_pin_targets` does; the
    /// solver reads it at each step start. Before `initialize` it replaces
    /// the `set_params` gravity at once.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> CtsStatus {
        if !gravity.iter().all(|g| g.is_finite()) {
            set_last_error("set_gravity: non-finite component");
            return CtsStatus::BadArg;
        }
        let target = Vec3f::new(gravity[0], gravity[1], gravity[2]);
        if !self.initialized {
            self.sim_args.gravity = gravity;
            return CtsStatus::Ok;
        }
        let now = self.backend.as_ref().unwrap().state.time;
        let from = self.gravity_at(now);
        self.gravity_blend = Some((from, target));
        self.gravity_segment = (now, next_frame_time(now, self.sim_args.fps));
        CtsStatus::Ok
    }

    /// Gravity at sim time `time` (after initialize).
    fn gravity_at(&self, time: f64) -> Vec3f {
        match self.gravity_blend {
            Some((from, to)) => pin_blend(from, to, self.gravity_segment, time),
            None => self.param.as_ref().unwrap().gravity,
        }
    }

    /// Build the pin constraint for the step from `prev_time` to `time` (the
    /// driver's `make_constraint(state.time + dt, state.time)`). Caller pins
    /// take their (interpolated) target or rest position; every animated
    /// collider vertex takes its keyframed pose. `step_delta` is the pose
    /// difference between the two times, which the solver walks back when CCD
    /// truncates the step. The static collider's `CollisionMesh` is rebuilt
    /// here each call because it cannot be cloned.
    ///
    /// Since ABI 6 every pin is an exact Dirichlet condition (the solver
    /// eliminates the DOF), so the per-pin stiffness staged by `set_pins` is
    /// accepted but no longer used. A pin with a pull weight `> 0`
    /// (`set_pin_pull`) is a soft `PullPair` toward the same position
    /// instead, as scene.rs `make_constraint` does.
    fn build_constraint(&self, rest_vertex: &Matrix3xX<f32>, time: f64, prev_time: f64) -> Constraint {
        let ghat = self.sim_args.constraint_ghat;
        let mut fix = Vec::with_capacity(self.pin_indices.len());
        let mut pull = Vec::new();
        for (i, &idx) in self.pin_indices.iter().enumerate() {
            let kinematic = self.pin_targets[i].is_some();
            let position = self.pin_position(i, time, rest_vertex);
            let weight = self.pin_pull.get(i).copied().unwrap_or(0.0);
            if weight > 0.0 {
                pull.push(PullPair {
                    position,
                    index: idx,
                    weight,
                    allow_intersection: false,
                });
                continue;
            }
            let step_delta = if kinematic {
                position - self.pin_position(i, prev_time, rest_vertex)
            } else {
                Vec3f::zeros()
            };
            fix.push(FixPair {
                position,
                step_delta,
                ghat,
                index: idx,
                kinematic,
                allow_intersection: false,
            });
        }
        for c in &self.animated_colliders {
            for j in 0..c.vertex_count() {
                let position = c.pose(j, time);
                fix.push(FixPair {
                    position,
                    step_delta: position - c.pose(j, prev_time),
                    ghat,
                    index: (c.first + j) as u32,
                    kinematic: true,
                    allow_intersection: false,
                });
            }
        }
        let mesh = match &self.collider {
            Some(c) => make_collision_mesh(&c.vert, &c.face, &c.face_props, &c.face_params),
            None => CollisionMesh::new(),
        };
        Constraint {
            fix: CVec::from(&fix[..]),
            pull: CVec::from(&pull[..]),
            torque_groups: CVec::new(),
            torque_vertices: CVec::new(),
            sphere: CVec::new(),
            floor: CVec::new(),
            stitch: CVec::new(),
            mesh,
        }
    }

    /// Where caller pin `i` is at sim time `time`: its rest position with no
    /// target, else the linear blend from `pin_from` to the target across
    /// `pin_segment`. `rest_vertex` is indexed by `pin_indices` (merged order
    /// after `initialize`).
    fn pin_position(&self, i: usize, time: f64, rest_vertex: &Matrix3xX<f32>) -> Vec3f {
        let rest = || rest_vertex.column(self.pin_indices[i] as usize).into_owned();
        let Some(target) = self.pin_targets[i] else {
            return rest();
        };
        let from = self.pin_from[i].unwrap_or_else(rest);
        pin_blend(from, target, self.pin_segment, time)
    }

    pub fn initialize(&mut self) -> CtsStatus {
        if self.initialized {
            set_last_error("initialize called twice");
            return CtsStatus::BadState;
        }
        // A build that links no GPU backend library would run the whole solve
        // on the CPU host renderings while every call still succeeds, about
        // 170x slower. Refuse it here so the host sees it at once.
        if ppf_cts_solver::linked_backend_name().is_none() {
            set_last_error(
                "this ppf_cts_capi build links no GPU backend library (abi_backend_linked                  not set); the solve would run on the CPU",
            );
            return CtsStatus::NoCudaDevice;
        }
        let Some(geometry) = self.geometry.as_ref() else {
            set_last_error("initialize before set_geometry");
            return CtsStatus::BadState;
        };
        let staged_vertex_count = geometry.verts.len() / 3;
        if !self.pdrd_vert_index.is_empty() && self.pdrd_vert_index.len() != staged_vertex_count {
            set_last_error("set_rigidbodies: vert_index length != vertex count");
            return CtsStatus::BadArg;
        }
        if self.pin_indices.iter().any(|&p| p as usize >= staged_vertex_count) {
            set_last_error("set_pins: pin index >= vertex count");
            return CtsStatus::BadArg;
        }

        // Splice the animated-collider pin shells in after the caller's
        // contact vertices / shell faces, each with its own material row.
        let first_shell_row = self.materials.len() as u32;
        let posed: Vec<Vec<f32>> =
            self.animated_colliders.iter().map(|c| c.posed_verts(0.0)).collect();
        let shells: Vec<ingest::ShellInsert> = self
            .animated_colliders
            .iter()
            .zip(&posed)
            .enumerate()
            .map(|(k, (c, verts))| ingest::ShellInsert {
                verts,
                faces: &c.faces,
                material_row: first_shell_row + k as u32,
            })
            .collect();
        let merged = ingest::insert_shells(
            &geometry.verts,
            &geometry.faces,
            &geometry.tets,
            &geometry.rods,
            geometry.shell_face_count,
            (!self.face_mat_idx.is_empty()).then_some(self.face_mat_idx.as_slice()),
            &shells,
        );
        drop(shells);
        self.geometry = None;
        let shell_materials: Vec<CtsMaterial> =
            self.animated_colliders.iter().map(|c| c.shell_material()).collect();
        self.materials.extend(shell_materials);
        if let Some(idx) = &merged.face_mat_idx {
            self.face_mat_idx = idx.clone();
        }
        for (c, &first) in self.animated_colliders.iter_mut().zip(&merged.shell_first) {
            c.first = first;
        }
        for p in self.pin_indices.iter_mut() {
            *p = merged.remap(*p);
        }
        if !self.pdrd_vert_index.is_empty() {
            let insert = merged.insert;
            self.pdrd_vert_index
                .splice(insert..insert, std::iter::repeat(0).take(merged.shell_vertex_count));
        }
        let remapped: Vec<u32> = self.pdrd_vert_list.iter().map(|&v| merged.remap(v)).collect();
        self.pdrd_vert_list = remapped;
        // `read_positions` returns the caller's contact vertices only; the
        // shells sit right after them, so that is the first `insert` verts.
        let cloth_surface_vert_count = merged.insert as u32;

        // Initial velocity in the merged order (scene.rs `vel.bin`); `build`
        // seeds the previous positions as `x - dt * v`.
        let mut velocity = Matrix3xX::<f32>::zeros(merged.verts.len() / 3);
        for (i, v) in self.velocities.chunks_exact(3).enumerate() {
            let col = merged.remap(i as u32) as usize;
            velocity.set_column(col, &Vec3f::new(v[0], v[1], v[2]));
        }

        let meshset = ingest::build_meshset(
            &merged.verts,
            &merged.faces,
            &merged.tets,
            &merged.rods,
            merged.shell_face_count,
            merged.collider_vertex_mask,
        );

        let mm = &meshset.mesh.mesh;
        let face_area = triutils::face_areas(&meshset.vertex, &mm.face);
        let tet_volumes = triutils::tet_volumes(&meshset.vertex, &mm.tet);
        let surface_vert_count = cloth_surface_vert_count;

        let nv = meshset.vertex.ncols();

        // Tag PDRD faces/tets (every vertex in a PDRD body) with
        // `Model::Pdrd` so their elastic energy is skipped and their
        // element mass is zeroed — the per-body PDRD pass in `build`
        // supplies volumetric mass instead. Mirrors `scene.rs`.
        let face_mat_idx = (!self.face_mat_idx.is_empty()).then_some(self.face_mat_idx.as_slice());
        let tet_mat_idx = (!self.tet_mat_idx.is_empty()).then_some(self.tet_mat_idx.as_slice());
        let rod_mat_idx = (!self.rod_mat_idx.is_empty()).then_some(self.rod_mat_idx.as_slice());
        let mut props = ingest::build_props(
            &meshset,
            &self.materials,
            face_mat_idx,
            tet_mat_idx,
            rod_mat_idx,
            &face_area,
            &tet_volumes,
            &self.pdrd_vert_index,
        );
        // Grain mass / contact params for the SAND pass in `build`.
        props.sand = self.sand;
        let constraint = self.build_constraint(&meshset.vertex, 0.0, 0.0);

        let pdrd = builder::PdrdSceneData {
            body_rows: &self.pdrd_body_rows,
            vert_index: &self.pdrd_vert_index,
            vert_list: &self.pdrd_vert_list,
            rest_centered: &self.pdrd_rest_centered,
        };
        // No aggregate locks: one displacement group, both axes zero (a zero
        // axis under the default modes disables the lock), every vertex in it.
        // This is exactly what `scene.rs` builds when no lock files exist.
        let lock_axes = [Vec3f::zeros()];
        let translation_modes = [TRANSLATION_LOCK_AXIS];
        let rotation_modes = [ROTATION_LOCK_ALLOW_ONLY];
        let vert_dmap = vec![0u32; nv];
        let lock_data = builder::LockSceneData {
            translation_axes: &lock_axes,
            translation_modes: &translation_modes,
            rotation_axes: &lock_axes,
            rotation_modes: &rotation_modes,
            vert_dmap: &vert_dmap,
        };
        // No Cosserat rods.
        let cosserat = builder::CosseratSceneData { edge: &[], tilt: &[] };
        let dataset = builder::build(
            &self.sim_args,
            &meshset,
            &velocity,
            &mut props,
            constraint,
            pdrd,
            lock_data,
            cosserat,
        );

        // The CUDA backend retains the *pointer* to this `ParamSet`: main.cu
        // stores `param = _param` at `initialize` and dereferences it on
        // every `advance` (reading `dt`/`playback`, writing back `time`/
        // `prev_dt`). It must therefore live at a stable address for the
        // whole session. Move it into `self.param` *before* `initialize` and
        // hand the backend a reference to that final location. Passing `&`
        // to a local that is then moved into `self` would leave the backend
        // dereferencing freed stack — observed as dt=0/playback=0 -> NaN in
        // the first solve. (The driver binary's `run` avoids this only
        // because `param` lives on its stack for the entire step loop.)
        self.param = Some(builder::make_param(&self.sim_args));
        let param = self.param.as_ref().unwrap();

        // `build` borrowed `meshset`; those borrows are done, so move it
        // into the backend now (it owns the rest mesh for `read_positions`
        // and per-step rest-pin lookups).
        // The driver keeps the *pointer* to the `DataSet` too (its SCENE
        // slot), so the same rule applies: move it into `self.dataset` first
        // and pass the final location, never the local.
        self.dataset = Some(dataset);
        let dataset = self.dataset.as_ref().unwrap();
        let mut backend = Backend::new(meshset);
        if !backend.initialize_in_memory(dataset, param) {
            self.dataset = None;
            set_last_error("backend initialize_in_memory failed");
            return CtsStatus::InitFailed;
        }

        self.surface_vert_count = surface_vert_count;
        self.backend = Some(backend);
        self.initialized = true;
        CtsStatus::Ok
    }

    pub fn step(&mut self, out_sim_time: Option<&mut f64>) -> CtsStatus {
        if !self.initialized {
            set_last_error("step before initialize");
            return CtsStatus::BadState;
        }
        // Build the constraint while holding only an immutable borrow of
        // the backend (for the rest vertices), then drop it before the
        // mutable step borrow below.
        if self.gravity_blend.is_some() {
            let gravity = self.gravity_at(self.backend.as_ref().unwrap().state.time);
            self.param.as_mut().unwrap().gravity = gravity;
        }
        let constraint = {
            let backend = self.backend.as_ref().unwrap();
            let param = self.param.as_ref().unwrap();
            let start = backend.state.time;
            self.build_constraint(&backend.mesh.vertex, start + f64::from(param.dt), start)
        };
        let backend = self.backend.as_mut().unwrap();
        let dataset = self.dataset.as_ref().unwrap();
        let param = self.param.as_ref().unwrap();
        let result = backend.step_in_memory(dataset, param, &constraint);
        // A GPU FAILURE IS RECORDED, NOT EXITED ON (soft mode, enabled by
        // `ppf_cts_create`). The device is lost, so this context is done; the
        // next `initialize` of a new context reopens it.
        if ppf_cts_solver::driver::fatal_code() != 0 {
            self.initialized = false;
            set_last_error(device_fault_message());
            return CtsStatus::DeviceFault;
        }
        if !result.success() {
            set_last_error("solver step failed (CCD/PCG/intersection)");
            return CtsStatus::StepFailed;
        }
        let sim_time = backend.state.time;
        if let Some(slot) = out_sim_time {
            *slot = sim_time;
        }
        CtsStatus::Ok
    }

    pub fn read_positions(&self, out: &mut [f32]) -> CtsStatus {
        if !self.initialized {
            set_last_error("read_positions before initialize");
            return CtsStatus::BadState;
        }
        let n = self.surface_vert_count as usize;
        if out.len() < 3 * n {
            set_last_error("read_positions: output buffer too small");
            return CtsStatus::BadArg;
        }
        let backend = self.backend.as_ref().unwrap();
        for (i, col) in backend.state.curr_vertex.columns(0, n).column_iter().enumerate() {
            out[3 * i] = col[0];
            out[3 * i + 1] = col[1];
            out[3 * i + 2] = col[2];
        }
        CtsStatus::Ok
    }

    pub fn surface_vertex_count(&self) -> u32 {
        self.surface_vert_count
    }

    pub fn intersection_count(&self) -> u32 {
        match &self.backend {
            Some(b) => b.intersection_count(),
            None => 0,
        }
    }
}

/// A material row a tet may use: a solid model and the ranges scene.rs
/// `make_props` asserts for tets.
fn check_tet_material(m: &CtsMaterial) -> Result<(), &'static str> {
    if !matches!(m.model, CtsModel::Arap | CtsModel::StVK | CtsModel::SNHk) {
        return Err("model must be ARAP, StVK or SNHk (Baraff-Witkin is shell-only)");
    }
    if !(m.young_modulus > 0.0) || !m.young_modulus.is_finite() {
        return Err("young_modulus must be positive");
    }
    if !(m.poisson_ratio > 0.0 && m.poisson_ratio < 0.5) {
        return Err("poisson_ratio must be in (0, 0.5)");
    }
    if !(m.density > 0.0) || !m.density.is_finite() {
        return Err("density must be positive");
    }
    Ok(())
}

/// The first multiple of `1/fps` strictly after `now` (a hair of slack so a
/// step that lands a rounding error short of a frame counts as on it).
fn next_frame_time(now: f64, fps: f64) -> f64 {
    if !(fps > 0.0) {
        return now;
    }
    ((now * fps + 1e-6).floor() + 1.0) / fps
}

/// Linear blend from `from` to `target` across the sim-time span `segment`,
/// clamped to its ends.
fn pin_blend(from: Vec3f, target: Vec3f, segment: (f64, f64), time: f64) -> Vec3f {
    let (t0, t1) = segment;
    if t1 <= t0 || time >= t1 {
        return target;
    }
    if time <= t0 {
        return from;
    }
    let s = ((time - t0) / (t1 - t0)) as f32;
    from + (target - from) * s
}

/// The driver's recorded fatal as a one-line message. The detail is wrapped
/// in the driver's and the seam's context; when it carries the CUDA
/// backend's own message ("CUDA out of memory: ..."), that part is what a user
/// can act on, so it is returned alone.
fn device_fault_message() -> String {
    let raw = ppf_cts_solver::driver::fatal_detail();
    let detail = if raw.is_null() {
        String::new()
    } else {
        // Safety: the driver hands out a NUL-terminated, process-lived buffer.
        unsafe { std::ffi::CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    };
    match detail.find("CUDA ") {
        Some(at) => detail[at..].to_string(),
        None if detail.is_empty() => "GPU device fault during the step".to_string(),
        None => detail,
    }
}

impl Default for CtsContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pull_weight_splits_pins_into_pull_and_fix() {
        let mut ctx = CtsContext::new();
        assert!(matches!(ctx.set_pins(&[0, 1], None), CtsStatus::Ok));
        assert!(matches!(ctx.set_pin_pull(&[0.5]), CtsStatus::BadArg));
        assert!(matches!(ctx.set_pin_pull(&[0.0, 0.5]), CtsStatus::Ok));
        let targets = [1.0f32, 0.0, 0.0, 2.0, 3.0, 4.0];
        assert!(matches!(ctx.update_pin_targets(&targets), CtsStatus::Ok));
        let rest = Matrix3xX::<f32>::zeros(2);
        let c = ctx.build_constraint(&rest, 0.1, 0.0);
        let fix: Vec<FixPair> = c.fix.iter().copied().collect();
        let pull: Vec<PullPair> = c.pull.iter().copied().collect();
        assert_eq!(fix.len(), 1);
        assert_eq!(pull.len(), 1);
        assert_eq!(fix[0].index, 0);
        let p = pull[0];
        assert_eq!(p.index, 1);
        assert_eq!(p.weight, 0.5);
        assert!((p.position - Vec3f::new(2.0, 3.0, 4.0)).norm() < 1e-6);
    }

    #[test]
    fn keyframed_collider_moves_rotates_and_holds() {
        let mut ctx = CtsContext::new();
        // One triangle; key 0 at rest, key 1 at t=1 s moved +2 in x and
        // turned 90 degrees about +z.
        let local = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let half = std::f64::consts::FRAC_1_SQRT_2;
        let status = ctx.add_animated_collider(
            &local,
            &[0, 1, 2],
            &[0.0, 1.0],
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, half, 0.0, 0.0, half],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            CtsColliderMaterial::default(),
        );
        assert!(matches!(status, CtsStatus::Ok));
        let c = &ctx.animated_colliders[0];
        let at = |t: f64| c.pose(0, t);
        assert!((at(0.0) - Vec3f::new(1.0, 0.0, 0.0)).norm() < 1e-6);
        // Halfway: translated +1, rotated 45 degrees.
        let mid = at(0.5);
        assert!((mid - Vec3f::new(1.0 + half as f32, half as f32, 0.0)).norm() < 1e-5, "{mid:?}");
        // End and beyond: +2 x, (1,0,0) turned to (0,1,0).
        assert!((at(1.0) - Vec3f::new(2.0, 1.0, 0.0)).norm() < 1e-5);
        assert!((at(5.0) - Vec3f::new(2.0, 1.0, 0.0)).norm() < 1e-5);
        assert_eq!(c.shell_material().density, 0.0);
    }

    #[test]
    fn animated_collider_rejects_bad_keys() {
        let mut ctx = CtsContext::new();
        let local = [0.0f32; 9];
        let status = ctx.add_animated_collider(
            &local,
            &[0, 1, 2],
            &[1.0, 1.0],
            &[0.0; 6],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[1.0; 6],
            CtsColliderMaterial::default(),
        );
        assert!(matches!(status, CtsStatus::BadArg));
    }

    #[test]
    fn pin_moves_across_the_frame_not_on_the_first_step() {
        let from = Vec3f::new(0.0, 0.0, 0.0);
        let to = Vec3f::new(1.0, 0.0, 0.0);
        // Frame span 0.1..0.2 s: a quarter in, a quarter of the way.
        let p = pin_blend(from, to, (0.1, 0.2), 0.125);
        assert!((p[0] - 0.25).abs() < 1e-6);
        assert_eq!(pin_blend(from, to, (0.1, 0.2), 0.3), to);
        assert_eq!(pin_blend(from, to, (0.0, 0.0), 0.0), to);
        // Next boundary after 0.05 s at 24 fps is frame 2; a time a rounding
        // error short of frame 1 still counts as frame 1.
        assert!((next_frame_time(0.05, 24.0) - 2.0 / 24.0).abs() < 1e-12);
        assert!((next_frame_time(1.0 / 24.0 - 1e-12, 24.0) - 2.0 / 24.0).abs() < 1e-12);
    }

    /// One tet (corner at the origin, edges 0.1 m) with its four boundary
    /// faces as contact-only faces (`shell_face_count` 0).
    const TET_VERTS: [f32; 12] = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1];
    const TET_FACES: [u32; 12] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
    const TET: [u32; 4] = [0, 1, 2, 3];

    fn last_error_text() -> String {
        let ptr = crate::error::last_error_ptr();
        unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    fn solid(model: CtsModel) -> CtsMaterial {
        CtsMaterial {
            model,
            young_modulus: 500.0,
            poisson_ratio: 0.35,
            density: 100.0,
            ..CtsMaterial::default()
        }
    }

    fn stage_tet(ctx: &mut CtsContext, mats: &[CtsMaterial], tet_idx: Option<&[u32]>) -> CtsStatus {
        ctx.set_geometry_ex(&TET_VERTS, &TET_FACES, &TET, &[], 0, mats, None, tet_idx, None)
    }

    #[test]
    fn tet_rows_must_be_solid_materials() {
        let mut ctx = CtsContext::new();
        // Baraff-Witkin (the default row) on a tet is rejected.
        assert!(matches!(stage_tet(&mut ctx, &[CtsMaterial::default()], None), CtsStatus::BadArg));
        assert!(last_error_text().contains("Baraff-Witkin"));
        // A BW row is fine for faces when the tet selects a solid row.
        let mats = [CtsMaterial::default(), solid(CtsModel::Arap)];
        assert!(matches!(stage_tet(&mut ctx, &mats, Some(&[1])), CtsStatus::Ok));
        for model in [CtsModel::Arap, CtsModel::StVK, CtsModel::SNHk] {
            assert!(matches!(stage_tet(&mut ctx, &[solid(model)], None), CtsStatus::Ok));
        }
        let bad = |f: fn(&mut CtsMaterial)| {
            let mut m = solid(CtsModel::Arap);
            f(&mut m);
            m
        };
        for m in [
            bad(|m| m.young_modulus = 0.0),
            bad(|m| m.poisson_ratio = 0.5),
            bad(|m| m.poisson_ratio = 0.0),
            bad(|m| m.density = -1.0),
        ] {
            assert!(matches!(stage_tet(&mut CtsContext::new(), &[m], None), CtsStatus::BadArg));
        }
    }

    #[test]
    fn velocities_need_geometry_and_matching_count() {
        let mut ctx = CtsContext::new();
        let v = [0.0f32; 12];
        assert!(matches!(ctx.set_velocities(&v), CtsStatus::BadState));
        assert!(matches!(stage_tet(&mut ctx, &[solid(CtsModel::Arap)], None), CtsStatus::Ok));
        assert!(matches!(ctx.set_velocities(&v[..9]), CtsStatus::BadArg));
        assert!(matches!(ctx.set_velocities(&v), CtsStatus::Ok));
        assert_eq!(ctx.velocities.len(), 12);
        // Restaging geometry clears them.
        assert!(matches!(stage_tet(&mut ctx, &[solid(CtsModel::Arap)], None), CtsStatus::Ok));
        assert!(ctx.velocities.is_empty());
    }

    /// One ARAP tet, no gravity, no air: moves by v * t. Needs a GPU backend
    /// (`--features ppf-cts-solver/cuda`); skipped on a build without one.
    #[test]
    fn tet_with_initial_velocity_moves_by_v_t() {
        if ppf_cts_solver::linked_backend_name().is_none() {
            eprintln!("skipped: no GPU backend linked");
            return;
        }
        let mut ctx = CtsContext::new();
        let params = CtsGlobalParams {
            gravity: [0.0; 3],
            air_density: 0.0,
            air_friction: 0.0,
            ..CtsGlobalParams::default()
        };
        assert!(matches!(ctx.set_params(&params), CtsStatus::Ok));
        assert!(matches!(stage_tet(&mut ctx, &[solid(CtsModel::Arap)], None), CtsStatus::Ok));
        let v = [1.0f32, 0.0, -0.5];
        let vel: Vec<f32> = (0..4).flat_map(|_| v).collect();
        assert!(matches!(ctx.set_velocities(&vel), CtsStatus::Ok));
        assert!(matches!(ctx.initialize(), CtsStatus::Ok), "{}", last_error_text());
        let mut time = 0.0;
        for _ in 0..20 {
            assert!(matches!(ctx.step(Some(&mut time)), CtsStatus::Ok));
        }
        let mut out = [0.0f32; 12];
        assert!(matches!(ctx.read_positions(&mut out), CtsStatus::Ok));
        assert!(time > 0.0);
        for i in 0..4 {
            for a in 0..3 {
                let moved = out[3 * i + a] - TET_VERTS[3 * i + a];
                let expect = v[a] * time as f32;
                assert!((moved - expect).abs() < 1e-3 + 0.02 * expect.abs(), "vertex {i} axis {a}: moved {moved}, expected {expect}");
            }
        }
    }

    #[test]
    fn gravity_set_before_init_and_blends_across_the_frame() {
        let mut ctx = CtsContext::new();
        assert!(matches!(ctx.set_gravity([0.0, f32::NAN, 0.0]), CtsStatus::BadArg));
        assert!(matches!(ctx.set_gravity([0.0, 9.8, 0.0]), CtsStatus::Ok));
        assert_eq!(ctx.sim_args.gravity, [0.0, 9.8, 0.0]);
        // After initialize: flip from -9.8 to +9.8 over the frame 0.1..0.2 s.
        ctx.gravity_blend = Some((Vec3f::new(0.0, -9.8, 0.0), Vec3f::new(0.0, 9.8, 0.0)));
        ctx.gravity_segment = (0.1, 0.2);
        assert!(ctx.gravity_at(0.15)[1].abs() < 1e-5);
        assert_eq!(ctx.gravity_at(0.05)[1], -9.8);
        assert_eq!(ctx.gravity_at(0.3)[1], 9.8);
    }
}
