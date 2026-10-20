// File: crates/ppf-cts-capi/src/params.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// repr(C) parameter/material structs and their mapping onto the driver's
// `SimArgs` and per-element params. `SimArgs` carries no `Default`, so the
// documented defaults are hardcoded here (matching the frontend's
// `app_param` / `object_param`). The strings `SimArgs` uses for `barrier`
// and `friction_mode` are exposed to C as small enums.

use ppf_cts_solver::data::Model;
use ppf_cts_solver::SimArgs;

/// Elastic model selector. Mirrors `data::Model` (and `ppf_cts_model` in
/// the header). Kept as an explicit `u32` enum for a stable C ABI.
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum CtsModel {
    Arap = 0,
    StVK = 1,
    BaraffWitkin = 2,
    SNHk = 3,
}

impl CtsModel {
    pub fn to_model(self) -> Model {
        match self {
            CtsModel::Arap => Model::Arap,
            CtsModel::StVK => Model::StVK,
            CtsModel::BaraffWitkin => Model::BaraffWitkin,
            CtsModel::SNHk => Model::SNHk,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum CtsBarrier {
    Cubic = 0,
    Quad = 1,
    Log = 2,
}

impl CtsBarrier {
    fn as_str(self) -> &'static str {
        match self {
            CtsBarrier::Cubic => "cubic",
            CtsBarrier::Quad => "quad",
            CtsBarrier::Log => "log",
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum CtsFrictionMode {
    Min = 0,
    Max = 1,
    Mean = 2,
}

impl CtsFrictionMode {
    fn as_str(self) -> &'static str {
        match self {
            CtsFrictionMode::Min => "min",
            CtsFrictionMode::Max => "max",
            CtsFrictionMode::Mean => "mean",
        }
    }
}

/// Global solver parameters. Field-for-field mappable to `SimArgs`, minus
/// the file/output-only knobs (`keep_states`, `frames`, `auto_save`, ...)
/// which the in-memory embedding fills with inert defaults. Booleans are
/// `i32` (0/1) for a clean C ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtsGlobalParams {
    pub gravity: [f32; 3],
    pub wind: [f32; 3],
    pub dt: f32,
    pub fps: f64,
    pub air_density: f32,
    pub air_friction: f32,
    pub isotropic_air_friction: f32,
    pub min_newton_steps: u32,
    pub target_toi: f32,
    pub line_search_max_t: f32,
    pub constraint_ghat: f32,
    pub constraint_tol: f32,
    pub cg_max_iter: u32,
    pub cg_tol: f32,
    pub ccd_eps: f32,
    /// Ignored since ABI 6: the neutral-driver solver has no such knob. Kept
    /// so the struct layout is unchanged from ABI 5.
    pub ccd_reduction: f32,
    /// Ignored since ABI 6 (see `ccd_reduction`).
    pub ccd_max_iter: u32,
    pub max_dx: f32,
    pub eiganalysis_eps: f32,
    pub friction_eps: f32,
    pub csrmat_max_nnz: u32,
    pub playback: f32,
    /// Ignored: stitch stiffness is per `ppf_cts_set_stitches` call. Kept so
    /// the struct layout is unchanged.
    pub stitch_stiffness: f32,
    pub stitch_length_factor: f32,
    pub fix_xz: f32,
    pub barrier: CtsBarrier,
    pub friction_mode: CtsFrictionMode,
    pub disable_contact: i32,
    pub inactive_momentum: i32,
    pub include_face_mass: i32,
}

impl Default for CtsGlobalParams {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.8, 0.0],
            wind: [0.0, 0.0, 0.0],
            dt: 1e-3,
            fps: 60.0,
            air_density: 1e-3,
            air_friction: 0.2,
            isotropic_air_friction: 0.0,
            min_newton_steps: 0,
            target_toi: 0.25,
            line_search_max_t: 1.25,
            constraint_ghat: 1e-3,
            constraint_tol: 0.01,
            cg_max_iter: 10000,
            cg_tol: 1e-3,
            ccd_eps: 1e-7,
            ccd_reduction: 0.01,
            ccd_max_iter: 4096,
            max_dx: 1.0,
            eiganalysis_eps: 1e-2,
            friction_eps: 1e-5,
            csrmat_max_nnz: 10_000_000,
            playback: 1.0,
            stitch_stiffness: 1.0,
            stitch_length_factor: 10.0,
            fix_xz: 0.0,
            barrier: CtsBarrier::Cubic,
            friction_mode: CtsFrictionMode::Min,
            disable_contact: 0,
            inactive_momentum: 0,
            include_face_mass: 0,
        }
    }
}

impl CtsGlobalParams {
    /// Build the driver's `SimArgs` from these params. The file/output
    /// knobs the in-memory path never uses get inert values: `keep_states`
    /// / `keep_verts` / `auto_save` = 0, `frames` = 300, `fake_crash_frame`
    /// = -1 (disabled).
    pub fn to_sim_args(&self) -> SimArgs {
        SimArgs {
            disable_contact: self.disable_contact != 0,
            keep_states: 0,
            keep_verts: 0,
            dt: self.dt,
            inactive_momentum: self.inactive_momentum != 0,
            playback: self.playback,
            min_newton_steps: self.min_newton_steps,
            target_toi: self.target_toi,
            air_friction: self.air_friction,
            line_search_max_t: self.line_search_max_t,
            constraint_ghat: self.constraint_ghat,
            constraint_tol: self.constraint_tol,
            fps: self.fps,
            cg_max_iter: self.cg_max_iter,
            cg_tol: self.cg_tol,
            ccd_eps: self.ccd_eps,
            max_dx: self.max_dx,
            eiganalysis_eps: self.eiganalysis_eps,
            friction_eps: self.friction_eps,
            csrmat_max_nnz: self.csrmat_max_nnz,
            frames: 300,
            auto_save: 0,
            save_state_on_finish: false,
            checkpoints: String::new(),
            barrier: self.barrier.as_str().to_string(),
            friction_mode: self.friction_mode.as_str().to_string(),
            stitch_length_factor: self.stitch_length_factor,
            air_density: self.air_density,
            isotropic_air_friction: self.isotropic_air_friction,
            gravity: self.gravity,
            wind: self.wind,
            include_face_mass: self.include_face_mass != 0,
            fix_xz: self.fix_xz,
            fake_crash_frame: -1,
            precond: "block-jacobi".to_string(),
            schwarz_levels: 2,
            world_scaling: 1.0,
            // The driver's own default (args.rs `default_max_newton_steps`).
            max_newton_steps: 2048,
        }
    }
}

/// One material/group. Young's modulus + Poisson ratio + density mirror the
/// frontend; the capi derives Lame mu/lambda via `builder::convert_prop`.
/// For v1 a single material is applied uniformly to every face/tet/rod (see
/// DESIGN.md §5 Q2 — one material per group, one group).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtsMaterial {
    pub model: CtsModel,
    pub young_modulus: f32,
    pub poisson_ratio: f32,
    pub density: f32,
    pub bend: f32,
    pub friction: f32,
    pub strainlimit: f32,
    pub contact_gap: f32,
    // Geometric contact thickness (element inflation radius). Zero for shells
    // (a closed surface already bounds a volume); 1D rods need a positive value
    // to have any collision radius -- see noodle.ipynb's `contact-offset`.
    pub contact_offset: f32,
    // Rod rest-length factor ("length-factor"): each rod edge's rest length is
    // its drawn length times this. Below 1.0 pre-tensions the rod (yarn.ipynb
    // uses 0.85). Must be positive. Rods only; mass uses the drawn length.
    pub length_factor: f32,
    // Stretch plasticity ("plasticity", "plasticity-threshold"): shell faces
    // and tets. The rest shape creeps toward the current shape at this rate
    // (per second, alpha = 1 - exp(-rate * dt) per step) wherever a principal
    // stretch S leaves the dead zone |S - 1| > threshold (dimensionless).
    // 0 = off. Both must be non-negative. Rods ignore them.
    pub plasticity: f32,
    pub plasticity_threshold: f32,
    // Bend plasticity ("bend-plasticity", "bend-plasticity-threshold"): shell
    // hinges and interior rod joints. The rest bend angle creeps toward the
    // current angle at this rate (per second) once it is more than the
    // threshold (radians) away. 0 = off. Both must be non-negative. Tets
    // ignore them.
    pub bend_plasticity: f32,
    pub bend_plasticity_threshold: f32,
}

impl Default for CtsMaterial {
    fn default() -> Self {
        // Shell (Baraff-Witkin) defaults from `object_param`.
        Self {
            model: CtsModel::BaraffWitkin,
            young_modulus: 1000.0,
            poisson_ratio: 0.35,
            density: 1.0,
            bend: 10.0,
            friction: 0.0,
            strainlimit: 0.0,
            contact_gap: 1e-3,
            contact_offset: 0.0,
            length_factor: 1.0,
            plasticity: 0.0,
            plasticity_threshold: 0.0,
            bend_plasticity: 0.0,
            bend_plasticity_threshold: 0.0,
        }
    }
}

/// One collider material row. A collider is a passive contact surface, so it
/// consumes only the two contact fields — `contact_gap` (ghat) and
/// `friction`. Kept lean and separate from `CtsMaterial` (which carries a
/// dozen elastic fields a collider never uses).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtsColliderMaterial {
    pub contact_gap: f32,
    pub friction: f32,
}

/// One SAND material row for `ppf_cts_set_sand_ex` (one Sand object's
/// grains). Same units and rules as the `ppf_cts_set_sand` scalars:
/// `particle_mass > 0` (kg per grain), `grain_radius >= 0` (contact sphere
/// radius), `contact_gap > 0` (barrier gap), `friction >= 0`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtsSandMaterial {
    pub particle_mass: f32,
    pub grain_radius: f32,
    pub contact_gap: f32,
    pub friction: f32,
}

/// Force-field type. The values are the solver's `force_field::FieldType`;
/// the types the C API does not take (magnetic, texture, fluid) are absent.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CtsFieldType {
    Force = 0,
    /// Moving air: acts only through the air drag, so it needs
    /// `CtsGlobalParams.air_density > 0`.
    Wind = 1,
    Vortex = 2,
    Turbulence = 3,
    Drag = 5,
    Harmonic = 6,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CtsFieldShape {
    Point = 0,
    Plane = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CtsFieldFalloff {
    Sphere = 0,
    Tube = 1,
    Cone = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CtsFieldZDirection {
    Both = 0,
    Positive = 1,
    Negative = 2,
}

/// One force field's fixed settings (see `ppf_cts_add_force_field`).
/// `flags` is a mask of the `CTS_FIELD_*` bits.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtsForceField {
    pub field_type: CtsFieldType,
    pub shape: CtsFieldShape,
    pub falloff: CtsFieldFalloff,
    pub z_direction: CtsFieldZDirection,
    pub flags: u32,
    pub seed: u32,
}

impl Default for CtsColliderMaterial {
    fn default() -> Self {
        Self {
            contact_gap: 1e-3,
            friction: 0.5,
        }
    }
}
