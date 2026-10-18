// File: crates/ppf-cts-capi/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// C-ABI surface for embedding the CUDA contact solver (the LightWave
// plugin path). Every entry point is a thin `extern "C"` shim: it
// null-checks its pointers, wraps the body in `error::guard`/`guard_value`
// so a Rust panic is reported as `PPF_CTS_ERR_PANIC` rather than unwinding
// across FFI, and delegates the real work to `CtsContext`. The opaque
// handle is a `Box<CtsContext>` round-tripped through raw pointers.
//
// The CUDA backend's `print_rust` log callback is *not* defined here; it
// lives in `ppf-cts-solver`'s `lib.rs` and is linked in transitively, so
// the binary and this staticlib each provide it exactly once.

mod context;
mod error;
mod ingest;
mod params;

use std::ffi::c_char;

use context::CtsContext;
use error::{guard, guard_value, last_error_ptr, CtsStatus};
use params::{CtsColliderMaterial, CtsGlobalParams, CtsMaterial, CtsModel};
use ppf_cts_solver::data::PDRD_BODY_ROW_LEN;

/// Bumped whenever the C ABI (struct layouts or function signatures)
/// changes. The caller checks this against the value its header was built
/// against before doing anything else.
///
/// 6: ported onto the neutral-driver solver (kernels-only CUDA library,
///    Rust Newton driver). Adds `ppf_cts_backend_name`. `ppf_cts_initialize`
///    now returns `PPF_CTS_ERR_NO_CUDA_DEVICE` when the build links no GPU
///    backend library (the solve would otherwise run silently on the CPU host
///    renderings). Struct layouts are unchanged from 5.
/// 7: adds `ppf_cts_set_verbose`. The solver's per-step console transcript is
///    now OFF by default on the C API path. Struct layouts are unchanged.
/// 8: adds `ppf_cts_add_animated_collider` (keyframed collider as a pin
///    shell). `ppf_cts_update_pin_targets` after initialize now moves each pin
///    linearly over the steps up to the next frame boundary instead of
///    jumping on the first step. Struct layouts are unchanged.
/// 9: adds `ppf_cts_set_pin_pull` (per-pin soft pull weight: `> 0` makes the
///    pin a `PullPair` toward its target instead of an exact `FixPair`).
///    Struct layouts are unchanged.
/// 10: adds `ppf_cts_set_gravity` (gravity changeable between frames, blended
///    to the next frame boundary like pin targets). Struct layouts are
///    unchanged.
/// 11: adds `ppf_cts_set_velocities` (initial vertex velocities, m/s).
///    `ppf_cts_set_geometry_ex` now rejects an unknown material model, and a
///    tet whose material row is not a solid row (ARAP/StVK/SNHk, E > 0,
///    0 < nu < 0.5, density > 0). Struct layouts are unchanged.
/// 12: a CUDA error during a step (out of memory, a failed launch, a lost
///    context) no longer ends the process: `ppf_cts_step` returns the new
///    `CTS_STATUS_DEVICE_FAULT` (8) with the CUDA message in
///    [`ppf_cts_last_error`]. Destroy the context; a new one re-opens the
///    device. Struct layouts are unchanged.
const ABI_VERSION: u32 = 12;

/// Current C ABI version. Callers compare against their compiled header.
#[no_mangle]
pub extern "C" fn ppf_cts_abi_version() -> u32 {
    ABI_VERSION
}

/// Thread-local last-error message (valid until the next call on this
/// thread). Never null.
#[no_mangle]
pub extern "C" fn ppf_cts_last_error() -> *const c_char {
    last_error_ptr()
}

/// Name of the backend library the solve dispatches through, as reported by
/// the loaded library itself (e.g. `"cuda"`), or `"host"` when this build
/// links no GPU backend and would run the solve on the CPU. Static string,
/// never null.
#[no_mangle]
pub extern "C" fn ppf_cts_backend_name() -> *const c_char {
    static NAME: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let name = ppf_cts_solver::linked_backend_name().unwrap_or("host");
        std::ffi::CString::new(name).unwrap_or_default()
    })
    .as_ptr()
}

/// Whether the solver prints its per-step console transcript. Off by default
/// for the C API: the host owns the console. Process-global, like the solver.
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn apply_verbose() {
    use std::sync::atomic::Ordering;
    ppf_cts_solver::QUIET_TRANSCRIPT.store(!VERBOSE.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Turn the solver's per-step console transcript (`====== advance ======`,
/// `* time = ...`, newton/phase lines) on (nonzero) or off (0). Default off.
/// Process-global; may be called at any time, before or after
/// [`ppf_cts_create`]. Errors are still reported through the status codes and
/// [`ppf_cts_last_error`], and the CUDA backend's error lines still print.
#[no_mangle]
pub extern "C" fn ppf_cts_set_verbose(verbose: i32) {
    VERBOSE.store(verbose != 0, std::sync::atomic::Ordering::Relaxed);
    apply_verbose();
}

/// Allocate a new solver context. Returns null only on allocation panic.
/// Also puts the solver in soft-failure mode (process-global): a GPU error
/// during a step is returned as `CTS_STATUS_DEVICE_FAULT` instead of
/// ending the host process.
#[no_mangle]
pub extern "C" fn ppf_cts_create() -> *mut CtsContext {
    apply_verbose();
    ppf_cts_solver::driver::set_soft_fatal(true);
    guard_value(std::ptr::null_mut(), || {
        Box::into_raw(Box::new(CtsContext::new()))
    })
}

/// Destroy a context created by [`ppf_cts_create`]. Null is a no-op.
#[no_mangle]
pub extern "C" fn ppf_cts_destroy(ctx: *mut CtsContext) {
    if ctx.is_null() {
        return;
    }
    let _ = guard(|| {
        // SAFETY: `ctx` came from `Box::into_raw` in `ppf_cts_create` and
        // is not used again after this call (caller contract).
        drop(unsafe { Box::from_raw(ctx) });
        CtsStatus::Ok
    });
}

/// Fill `out` with the documented default global parameters.
#[no_mangle]
pub extern "C" fn ppf_cts_default_params(out: *mut CtsGlobalParams) -> CtsStatus {
    guard(|| {
        let Some(out) = (unsafe { out.as_mut() }) else {
            return CtsStatus::Null;
        };
        *out = CtsGlobalParams::default();
        CtsStatus::Ok
    })
}

/// Fill `out` with the documented default material.
#[no_mangle]
pub extern "C" fn ppf_cts_default_material(out: *mut CtsMaterial) -> CtsStatus {
    guard(|| {
        let Some(out) = (unsafe { out.as_mut() }) else {
            return CtsStatus::Null;
        };
        *out = CtsMaterial::default();
        CtsStatus::Ok
    })
}

#[no_mangle]
pub extern "C" fn ppf_cts_set_params(
    ctx: *mut CtsContext,
    params: *const CtsGlobalParams,
) -> CtsStatus {
    guard(|| {
        let (Some(ctx), Some(params)) = (unsafe { ctx.as_mut() }, unsafe { params.as_ref() }) else {
            return CtsStatus::Null;
        };
        ctx.set_params(params)
    })
}

/// Set the simulated geometry. Vertices are a shared pool (`3*vertex_count`
/// floats); faces/tets/rods are flat index arrays (`3*`/`4*`/`2*` their
/// counts). Any element array may be null when its count is 0.
///
/// `shell_face_count` is the number of *leading* faces treated as cloth
/// shells (membrane + bending elasticity); the remaining faces are
/// **solid contact-only** surfaces with no shell energy — the boundary
/// triangles of tet bodies belong here so the solid participates in
/// contact without behaving like fabric. Order faces shell-first. Pass
/// `shell_face_count == face_count` for pure cloth (the pre-ABI-2
/// behaviour) or `0` for pure fracture/solid bodies. Clamped to
/// `face_count`.
/// Uniform-material geometry staging. A thin wrapper over
/// [`ppf_cts_set_geometry_ex`] with a one-row material table and no
/// per-element index arrays (every element -> material row 0), preserving
/// the pre-ABI-4 single-material behaviour byte-for-byte.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ppf_cts_set_geometry(
    ctx: *mut CtsContext,
    vertices: *const f32,
    vertex_count: u32,
    faces: *const u32,
    face_count: u32,
    shell_face_count: u32,
    tets: *const u32,
    tet_count: u32,
    rods: *const u32,
    rod_count: u32,
    material: *const CtsMaterial,
) -> CtsStatus {
    ppf_cts_set_geometry_ex(
        ctx,
        vertices,
        vertex_count,
        faces,
        face_count,
        shell_face_count,
        tets,
        tet_count,
        rods,
        rod_count,
        material,
        1,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
    )
}

/// Per-element-material geometry staging. Same geometry arguments as
/// [`ppf_cts_set_geometry`], plus a material table (`materials` /
/// `material_count`, at least one row) and three optional per-element index
/// arrays selecting a material row per face/tet/rod. A null index array
/// means "uniform, every element uses row 0"; a non-null array must hold
/// exactly `face_count` / `tet_count` / `rod_count` entries, each
/// `< material_count`.
///
/// Since ABI 11 every row's `model` must be a known `CtsModel` value, and
/// every row a tet uses must be a solid row: `model` ARAP, StVK or SNHk
/// (Baraff-Witkin is shell-only), `young_modulus > 0`,
/// `0 < poisson_ratio < 0.5` and `density > 0`. Otherwise this returns
/// `PPF_CTS_ERR_BAD_ARG` and [`ppf_cts_last_error`] names the row. Staging
/// geometry clears any velocities set by [`ppf_cts_set_velocities`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ppf_cts_set_geometry_ex(
    ctx: *mut CtsContext,
    vertices: *const f32,
    vertex_count: u32,
    faces: *const u32,
    face_count: u32,
    shell_face_count: u32,
    tets: *const u32,
    tet_count: u32,
    rods: *const u32,
    rod_count: u32,
    materials: *const CtsMaterial,
    material_count: u32,
    face_material_idx: *const u32,
    tet_material_idx: *const u32,
    rod_material_idx: *const u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if vertices.is_null() || vertex_count == 0 {
            error::set_last_error("set_geometry: vertices null or empty");
            return CtsStatus::BadArg;
        }
        if materials.is_null() || material_count == 0 {
            error::set_last_error("set_geometry_ex: materials null or empty");
            return CtsStatus::BadArg;
        }
        // `model` is the struct's first field (a repr(u32) enum). Read it raw
        // before forming `&[CtsMaterial]`: a value outside the enum would be
        // undefined behaviour once typed.
        for row in 0..material_count as usize {
            let model = unsafe { std::ptr::read(materials.add(row).cast::<u32>()) };
            if model > CtsModel::SNHk as u32 {
                error::set_last_error(format!(
                    "set_geometry_ex: material row {row}: unknown model {model}"
                ));
                return CtsStatus::BadArg;
            }
        }
        if shell_face_count > face_count {
            error::set_last_error("set_geometry: shell_face_count exceeds face_count");
            return CtsStatus::BadArg;
        }
        let verts = unsafe { slice_f32(vertices, 3 * vertex_count as usize) };
        let faces = unsafe { slice_u32(faces, 3 * face_count as usize) };
        let tets = unsafe { slice_u32(tets, 4 * tet_count as usize) };
        let rods = unsafe { slice_u32(rods, 2 * rod_count as usize) };
        let materials = unsafe { slice_mat(materials, material_count as usize) };
        let face_mat_idx = if face_material_idx.is_null() {
            None
        } else {
            Some(unsafe { slice_u32(face_material_idx, face_count as usize) })
        };
        let tet_mat_idx = if tet_material_idx.is_null() {
            None
        } else {
            Some(unsafe { slice_u32(tet_material_idx, tet_count as usize) })
        };
        let rod_mat_idx = if rod_material_idx.is_null() {
            None
        } else {
            Some(unsafe { slice_u32(rod_material_idx, rod_count as usize) })
        };
        ctx.set_geometry_ex(
            verts,
            faces,
            tets,
            rods,
            shell_face_count as usize,
            materials,
            face_mat_idx,
            tet_mat_idx,
            rod_mat_idx,
        )
    })
}

/// Uniform-friction collider staging. A thin wrapper over
/// [`ppf_cts_set_collider_ex`] with a one-row collider-material table and no
/// per-face index array (every collider face shares `contact_gap` /
/// `friction`), preserving the pre-ABI-4 behaviour.
#[no_mangle]
pub extern "C" fn ppf_cts_set_collider(
    ctx: *mut CtsContext,
    vertices: *const f32,
    vertex_count: u32,
    faces: *const u32,
    face_count: u32,
    contact_gap: f32,
    friction: f32,
) -> CtsStatus {
    let mat = CtsColliderMaterial {
        contact_gap,
        friction,
    };
    ppf_cts_set_collider_ex(
        ctx,
        vertices,
        vertex_count,
        faces,
        face_count,
        &mat,
        1,
        std::ptr::null(),
    )
}

/// Per-face-material collider staging. Same collider geometry as
/// [`ppf_cts_set_collider`], plus a collider-material table (`materials` /
/// `material_count`) and an optional per-face index array. A null index
/// array means "uniform, every face uses row 0"; a non-null array must hold
/// exactly `face_count` entries, each `< material_count`. One call stages
/// every merged collider face; the per-face index separates distinct
/// colliders.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ppf_cts_set_collider_ex(
    ctx: *mut CtsContext,
    vertices: *const f32,
    vertex_count: u32,
    faces: *const u32,
    face_count: u32,
    materials: *const CtsColliderMaterial,
    material_count: u32,
    face_material_idx: *const u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if vertices.is_null() || faces.is_null() || vertex_count == 0 || face_count == 0 {
            error::set_last_error("set_collider: null/empty geometry");
            return CtsStatus::BadArg;
        }
        if materials.is_null() || material_count == 0 {
            error::set_last_error("set_collider_ex: materials null or empty");
            return CtsStatus::BadArg;
        }
        let verts = unsafe { slice_f32(vertices, 3 * vertex_count as usize) };
        let faces = unsafe { slice_u32(faces, 3 * face_count as usize) };
        let materials = unsafe { slice_collider_mat(materials, material_count as usize) };
        let face_mat_idx = if face_material_idx.is_null() {
            None
        } else {
            Some(unsafe { slice_u32(face_material_idx, face_count as usize) })
        };
        ctx.set_collider(verts, faces, materials, face_mat_idx)
    })
}

/// Add one ANIMATED collider body (call once per body, before
/// [`ppf_cts_initialize`]; any order relative to the other staging calls).
/// Static colliders still go through [`ppf_cts_set_collider_ex`].
///
/// - `local_vertices`: `3*vertex_count` floats, the body's mesh in its own
///   local space.
/// - `faces`: `3*face_count` triangle indices into `local_vertices`.
/// - Keyframes, `key_count >= 1`: `key_times` (seconds of sim time, strictly
///   ascending; sim time 0 is the state staged by `set_geometry`),
///   `key_translations` (3 per key), `key_rotations` (4 per key, quaternion
///   in `w, x, y, z` order, normalized here), `key_scales` (3 per key).
///   World position = `R(q) * (S .* local) + T`; translation/scale
///   interpolate linearly, rotation by slerp; the pose holds before the first
///   and after the last key.
/// - `material`: contact gap (> 0) and friction of the body's surface.
///
/// Internally this is the Python frontend's animated-static "pin shell": a
/// shell with density, Young's modulus, Poisson's ratio and bend all 0,
/// flagged as a STATIC collider (no self-contact, no contact with other
/// colliders, no bending), every vertex an exact kinematic pin driven by the
/// keyframes, evaluated at each step's end time.
///
/// Vertex indexing: the shell vertices are inserted right after the
/// caller's contact (surface) vertices, before any tet-interior vertex, and
/// shell faces right after the caller's shell faces. Every index the caller
/// stages (faces, rods, pins, PDRD) is in the caller's own numbering and is
/// remapped internally; the caller's contact vertices keep their indices, so
/// [`ppf_cts_surface_vertex_count`] / [`ppf_cts_read_positions`] report
/// exactly the caller's contact vertices and never the collider shells.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ppf_cts_add_animated_collider(
    ctx: *mut CtsContext,
    local_vertices: *const f32,
    vertex_count: u32,
    faces: *const u32,
    face_count: u32,
    key_times: *const f64,
    key_translations: *const f64,
    key_rotations: *const f64,
    key_scales: *const f64,
    key_count: u32,
    material: *const CtsColliderMaterial,
) -> CtsStatus {
    guard(|| {
        let (Some(ctx), Some(material)) = (unsafe { ctx.as_mut() }, unsafe { material.as_ref() })
        else {
            return CtsStatus::Null;
        };
        if local_vertices.is_null()
            || faces.is_null()
            || key_times.is_null()
            || key_translations.is_null()
            || key_rotations.is_null()
            || key_scales.is_null()
        {
            error::set_last_error("add_animated_collider: null array");
            return CtsStatus::BadArg;
        }
        let n_key = key_count as usize;
        let verts = unsafe { slice_f32(local_vertices, 3 * vertex_count as usize) };
        let faces = unsafe { slice_u32(faces, 3 * face_count as usize) };
        let times = unsafe { slice_f64(key_times, n_key) };
        let translations = unsafe { slice_f64(key_translations, 3 * n_key) };
        let rotations = unsafe { slice_f64(key_rotations, 4 * n_key) };
        let scales = unsafe { slice_f64(key_scales, 3 * n_key) };
        ctx.add_animated_collider(verts, faces, times, translations, rotations, scales, *material)
    })
}

/// Mark pinnable vertices once. `stiffness` may be null (per-pin force
/// scale defaults to 1.0); otherwise it must hold `count` floats.
#[no_mangle]
pub extern "C" fn ppf_cts_set_pins(
    ctx: *mut CtsContext,
    vertex_indices: *const u32,
    count: u32,
    stiffness: *const f32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if vertex_indices.is_null() && count != 0 {
            error::set_last_error("set_pins: null indices with non-zero count");
            return CtsStatus::BadArg;
        }
        let indices = unsafe { slice_u32(vertex_indices, count as usize) };
        let stiffness = if stiffness.is_null() {
            None
        } else {
            Some(unsafe { slice_f32(stiffness, count as usize) })
        };
        ctx.set_pins(indices, stiffness)
    })
}

/// Per-pin pull weights (`count` floats, one per pin in the `set_pins`
/// index order; `count` must equal the pin count). A weight `> 0` makes that
/// pin a soft pull of that weight toward its target (which still follows
/// [`ppf_cts_update_pin_targets`]); `<= 0` keeps it an exact pin. Call after
/// [`ppf_cts_set_pins`] (which resets every pin to exact) and before
/// [`ppf_cts_initialize`].
#[no_mangle]
pub extern "C" fn ppf_cts_set_pin_pull(
    ctx: *mut CtsContext,
    weights: *const f32,
    count: u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if weights.is_null() && count != 0 {
            error::set_last_error("set_pin_pull: null weights with non-zero count");
            return CtsStatus::BadArg;
        }
        let weights = unsafe { slice_f32(weights, count as usize) };
        ctx.set_pin_pull(weights)
    })
}

/// Push per-frame pin target positions (`3*count` floats), aligned with
/// the `set_pins` index order. Pins are exact (Dirichlet) constraints.
///
/// After [`ppf_cts_initialize`] the targets are for the NEXT frame boundary
/// (the next multiple of `1/fps`, `fps` from [`ppf_cts_set_params`]): each
/// pin moves linearly in sim time from where it is now to its target over
/// the steps up to that boundary, then holds. Call it once per frame, before
/// stepping that frame. Before initialize the targets apply at once (the
/// pins' starting pose).
#[no_mangle]
pub extern "C" fn ppf_cts_update_pin_targets(
    ctx: *mut CtsContext,
    positions: *const f32,
    count: u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if positions.is_null() && count != 0 {
            error::set_last_error("update_pin_targets: null positions");
            return CtsStatus::BadArg;
        }
        let positions = unsafe { slice_f32(positions, 3 * count as usize) };
        ctx.update_pin_targets(positions)
    })
}

/// Set gravity `(x, y, z)` (m/s^2). After [`ppf_cts_initialize`] it is the
/// value for the NEXT frame boundary (the next multiple of `1/fps`): gravity
/// moves linearly in sim time from its current value to it over the steps up
/// to that boundary, then holds, as [`ppf_cts_update_pin_targets`] does. Call
/// it between frames, before stepping that frame. Before initialize it
/// replaces the `gravity` given to [`ppf_cts_set_params`] at once.
#[no_mangle]
pub extern "C" fn ppf_cts_set_gravity(ctx: *mut CtsContext, x: f32, y: f32, z: f32) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        ctx.set_gravity([x, y, z])
    })
}

/// Set initial vertex velocities (m/s): `3*vertex_count` floats, xyz per
/// vertex in the same vertex order as [`ppf_cts_set_geometry_ex`].
/// `vertex_count` must equal that call's vertex count. Call after
/// [`ppf_cts_set_geometry_ex`] (which clears them) and before
/// [`ppf_cts_initialize`]; unset velocities are zero. The solver seeds its
/// previous positions as `x - dt * v`, as the frontend's `vel.bin` does.
#[no_mangle]
pub extern "C" fn ppf_cts_set_velocities(
    ctx: *mut CtsContext,
    velocities: *const f32,
    vertex_count: u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        if velocities.is_null() || vertex_count == 0 {
            error::set_last_error("set_velocities: velocities null or empty");
            return CtsStatus::BadArg;
        }
        let velocities = unsafe { slice_f32(velocities, 3 * vertex_count as usize) };
        ctx.set_velocities(velocities)
    })
}

/// Stage SAND granular material. Every grain is a loose vertex (no
/// incident face/tet) in the geometry staged via [`ppf_cts_set_geometry`];
/// all grains share this `particle_mass`, `grain_radius` (contact sphere
/// radius), `contact_gap`, and `friction`. Must be called before
/// [`ppf_cts_initialize`]. Threaded into `Props.sand`.
#[no_mangle]
pub extern "C" fn ppf_cts_set_sand(
    ctx: *mut CtsContext,
    particle_mass: f32,
    grain_radius: f32,
    contact_gap: f32,
    friction: f32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        ctx.set_sand(particle_mass, grain_radius, contact_gap, friction)
    })
}

/// Stage PDRD (position-based rigid-body) inputs. The caller precomputes
/// the per-body rows exactly as the Python frontend packs `pdrd_body.bin`:
/// - `body_rows`: `PDRD_BODY_ROW_LEN` (23) floats per body, `body_count`
///   bodies (`23 * body_count` floats total).
/// - `vert_index`: one entry per *global* vertex — a 1-based body id, or 0
///   for a vertex that is not part of any PDRD body. Length
///   (`vert_index_count`) must equal the staged vertex count, or null/0
///   for "no PDRD bodies".
/// - `vert_list`: flat global vertex indices, body-major
///   (`vert_list_count` entries).
/// - `rest_centered`: 3 floats per `vert_list` entry
///   (`rest_centered_count == 3 * vert_list_count`).
///
/// Any array may be null when its count is 0. Faces/tets fully inside a
/// PDRD body are tagged `Model::Pdrd` (elastic energy skipped, element
/// mass zeroed) at [`ppf_cts_initialize`]. Must be called before
/// initialize.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ppf_cts_set_rigidbodies(
    ctx: *mut CtsContext,
    body_rows: *const f32,
    body_count: u32,
    vert_index: *const u32,
    vert_index_count: u32,
    vert_list: *const u32,
    vert_list_count: u32,
    rest_centered: *const f32,
    rest_centered_count: u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        let body_rows =
            unsafe { slice_f32(body_rows, PDRD_BODY_ROW_LEN * body_count as usize) };
        let vert_index = unsafe { slice_u32(vert_index, vert_index_count as usize) };
        let vert_list = unsafe { slice_u32(vert_list, vert_list_count as usize) };
        let rest_centered = unsafe { slice_f32(rest_centered, rest_centered_count as usize) };
        ctx.set_rigidbodies(body_rows, vert_index, vert_list, rest_centered)
    })
}

/// Allocate GPU state and initialize the backend. Geometry/params/pins
/// must be staged first.
#[no_mangle]
pub extern "C" fn ppf_cts_initialize(ctx: *mut CtsContext) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        ctx.initialize()
    })
}

/// Advance one solver step. `out_sim_time` (nullable) receives the
/// post-step simulation time.
#[no_mangle]
pub extern "C" fn ppf_cts_step(ctx: *mut CtsContext, out_sim_time: *mut f64) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_mut() }) else {
            return CtsStatus::Null;
        };
        ctx.step(unsafe { out_sim_time.as_mut() })
    })
}

/// Copy current surface vertex positions into `out` (`3*vertex_count`
/// floats). `vertex_count` must be at least `surface_vertex_count`. These are
/// the caller's contact vertices `0..surface_vertex_count` in the caller's
/// numbering; animated collider shells are never included.
#[no_mangle]
pub extern "C" fn ppf_cts_read_positions(
    ctx: *const CtsContext,
    out: *mut f32,
    vertex_count: u32,
) -> CtsStatus {
    guard(|| {
        let Some(ctx) = (unsafe { ctx.as_ref() }) else {
            return CtsStatus::Null;
        };
        if out.is_null() {
            error::set_last_error("read_positions: null output");
            return CtsStatus::BadArg;
        }
        let out = unsafe { slice_f32_mut(out, 3 * vertex_count as usize) };
        ctx.read_positions(out)
    })
}

/// Number of surface vertices (the count `read_positions` writes): the
/// caller's staged vertices minus tet-interior ones, excluding animated
/// collider shells.
#[no_mangle]
pub extern "C" fn ppf_cts_surface_vertex_count(ctx: *const CtsContext) -> u32 {
    guard_value(0, || match unsafe { ctx.as_ref() } {
        Some(ctx) => ctx.surface_vertex_count(),
        None => 0,
    })
}

/// Current intersection-record count (0 when penetration-free). A
/// regression-harness hook.
#[no_mangle]
pub extern "C" fn ppf_cts_intersection_count(ctx: *mut CtsContext) -> u32 {
    guard_value(0, || match unsafe { ctx.as_mut() } {
        Some(ctx) => ctx.intersection_count(),
        None => 0,
    })
}

// --- pointer/length -> slice helpers --------------------------------------
// Each returns an empty slice for a null pointer or zero length, so the
// callee never sees a dangling pointer. SAFETY in each: the caller
// guarantees `ptr` points to at least `len` initialized elements.

unsafe fn slice_f32<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_f32_mut<'a>(ptr: *mut f32, len: usize) -> &'a mut [f32] {
    if ptr.is_null() || len == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

unsafe fn slice_f64<'a>(ptr: *const f64, len: usize) -> &'a [f64] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_u32<'a>(ptr: *const u32, len: usize) -> &'a [u32] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_mat<'a>(ptr: *const CtsMaterial, len: usize) -> &'a [CtsMaterial] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn slice_collider_mat<'a>(
    ptr: *const CtsColliderMaterial,
    len: usize,
) -> &'a [CtsColliderMaterial] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}
