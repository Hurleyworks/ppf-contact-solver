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
use params::{CtsColliderMaterial, CtsGlobalParams, CtsMaterial};
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
const ABI_VERSION: u32 = 6;

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

/// Allocate a new solver context. Returns null only on allocation panic.
#[no_mangle]
pub extern "C" fn ppf_cts_create() -> *mut CtsContext {
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

/// Push per-frame pin target positions (`3*count` floats), aligned with
/// the `set_pins` index order.
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
/// floats). `vertex_count` must be at least `surface_vertex_count`.
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

/// Number of surface vertices (the count `read_positions` writes).
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
