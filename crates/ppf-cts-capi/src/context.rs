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

use ppf_cts_solver::backend::{Backend, MeshSet};
use ppf_cts_solver::builder::{self, make_collision_mesh};
use ppf_cts_solver::data::{
    CollisionMesh, Constraint, DataSet, FaceParam, FaceProp, FixPair, ParamSet, Vec3f,
    PDRD_BODY_ROW_LEN, ROTATION_LOCK_ALLOW_ONLY, TRANSLATION_LOCK_AXIS,
};
use ppf_cts_solver::triutils;
use ppf_cts_solver::{CVec, SimArgs};

use crate::error::{set_last_error, CtsStatus};
use crate::ingest;
use crate::params::{CtsColliderMaterial, CtsGlobalParams, CtsMaterial};

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
    /// Staged before init; moved into `backend` on `initialize`.
    meshset: Option<MeshSet>,
    pin_indices: Vec<u32>,
    pin_stiffness: Vec<f32>,
    /// Per-pin target: `Some` => kinematic (driven to the target),
    /// `None` => soft-fixed at its rest position. Aligned with
    /// `pin_indices`.
    pin_targets: Vec<Option<Vec3f>>,
    /// Where each pin was at the START of the coming step (the target the
    /// previous step reached, or the rest position before the first step).
    /// `FixPair::step_delta` is `position - start`, which the solver needs to
    /// truncate a kinematic pin's travel when CCD shortens the step. Empty
    /// before `initialize`.
    pin_step_start: Vec<Vec3f>,
    collider: Option<Collider>,
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
            meshset: None,
            pin_indices: Vec::new(),
            pin_stiffness: Vec::new(),
            pin_targets: Vec::new(),
            pin_step_start: Vec::new(),
            collider: None,
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
        self.materials = materials.to_vec();
        self.face_mat_idx = face_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.tet_mat_idx = tet_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.rod_mat_idx = rod_mat_idx.map(|a| a.to_vec()).unwrap_or_default();
        self.meshset = Some(ingest::build_meshset(verts, faces, tets, rods, shell_face_count));
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
        // Default every pin to soft-fixed at rest until a target arrives.
        self.pin_targets = vec![None; indices.len()];
        CtsStatus::Ok
    }

    pub fn update_pin_targets(&mut self, positions: &[f32]) -> CtsStatus {
        let count = self.pin_indices.len();
        if positions.len() != 3 * count {
            set_last_error("update_pin_targets: positions length != 3 * pin count");
            return CtsStatus::BadArg;
        }
        for i in 0..count {
            self.pin_targets[i] = Some(Vec3f::new(
                positions[3 * i],
                positions[3 * i + 1],
                positions[3 * i + 2],
            ));
        }
        CtsStatus::Ok
    }

    /// Build the per-step pin constraint. `rest_vertex` supplies the rest
    /// position of soft-fixed (non-kinematic) pins; kinematic pins use
    /// their pushed target. The collider's `CollisionMesh` is rebuilt
    /// here each call because it cannot be cloned.
    ///
    /// Since ABI 6 every pin is an exact Dirichlet condition (the solver
    /// eliminates the DOF), so the per-pin stiffness staged by `set_pins` is
    /// accepted but no longer used.
    fn build_constraint(&self, rest_vertex: &Matrix3xX<f32>) -> Constraint {
        let ghat = self.sim_args.constraint_ghat;
        let mut fix = Vec::with_capacity(self.pin_indices.len());
        for (i, &idx) in self.pin_indices.iter().enumerate() {
            let position = self.pin_position(i, rest_vertex);
            let step_delta = match self.pin_step_start.get(i) {
                Some(start) => position - start,
                None => Vec3f::zeros(),
            };
            fix.push(FixPair {
                position,
                step_delta,
                ghat,
                index: idx,
                kinematic: self.pin_targets[i].is_some(),
                allow_intersection: false,
            });
        }
        let mesh = match &self.collider {
            Some(c) => make_collision_mesh(&c.vert, &c.face, &c.face_props, &c.face_params),
            None => CollisionMesh::new(),
        };
        Constraint {
            fix: CVec::from(&fix[..]),
            pull: CVec::new(),
            torque_groups: CVec::new(),
            torque_vertices: CVec::new(),
            sphere: CVec::new(),
            floor: CVec::new(),
            stitch: CVec::new(),
            mesh,
        }
    }

    /// The position pin `i` is driven to this step: its pushed target, or
    /// its rest position when no target has been pushed.
    fn pin_position(&self, i: usize, rest_vertex: &Matrix3xX<f32>) -> Vec3f {
        match self.pin_targets[i] {
            Some(target) => target,
            None => rest_vertex.column(self.pin_indices[i] as usize).into_owned(),
        }
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
        let meshset = match self.meshset.take() {
            Some(m) => m,
            None => {
                set_last_error("initialize before set_geometry");
                return CtsStatus::BadState;
            }
        };

        let mm = &meshset.mesh.mesh;
        let face_area = triutils::face_areas(&meshset.vertex, &mm.face);
        let tet_volumes = triutils::tet_volumes(&meshset.vertex, &mm.tet);
        let surface_vert_count = mm.surface_vert_count as u32;

        let nv = meshset.vertex.ncols();

        // PDRD vert_index, when present, must be one entry per global
        // vertex; reject a mismatch here with a clean error rather than
        // letting `build` panic across the FFI boundary.
        if !self.pdrd_vert_index.is_empty() && self.pdrd_vert_index.len() != nv {
            set_last_error("set_rigidbodies: vert_index length != vertex count");
            return CtsStatus::BadArg;
        }

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
        let constraint = self.build_constraint(&meshset.vertex);
        self.pin_step_start = self
            .pin_indices
            .iter()
            .map(|&idx| meshset.vertex.column(idx as usize).into_owned())
            .collect();

        let velocity = Matrix3xX::<f32>::zeros(nv);
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
        let constraint = {
            let backend = self.backend.as_ref().unwrap();
            self.build_constraint(&backend.mesh.vertex)
        };
        let backend = self.backend.as_mut().unwrap();
        let dataset = self.dataset.as_ref().unwrap();
        let param = self.param.as_ref().unwrap();
        let result = backend.step_in_memory(dataset, param, &constraint);
        if !result.success() {
            set_last_error("solver step failed (CCD/PCG/intersection)");
            return CtsStatus::StepFailed;
        }
        // The targets this step reached are where the next step starts.
        let rest = &backend.mesh.vertex;
        let reached: Vec<Vec3f> = (0..self.pin_indices.len())
            .map(|i| match self.pin_targets[i] {
                Some(target) => target,
                None => rest.column(self.pin_indices[i] as usize).into_owned(),
            })
            .collect();
        let sim_time = backend.state.time;
        self.pin_step_start = reached;
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

impl Default for CtsContext {
    fn default() -> Self {
        Self::new()
    }
}
