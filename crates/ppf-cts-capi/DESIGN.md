# ppf-cts-capi — Crate Layout & C API Surface (Proposal)

Status: **PROPOSAL — awaiting confirmation. No Rust code written yet.**
Target: embed ppf-contact-solver into a LightWave 3D plugin (NVIDIA/CUDA-only).

This document proposes (1) how the new `ppf-cts-capi` crate is structured and
linked, and (2) the concrete `extern "C"` header surface it exposes. It follows
the approved plan (`is-it-feasible-to-squishy-marble.md`).

---

## 0. Build environment (confirmed on this machine)

| Component | Value |
|---|---|
| Rust | cargo/rustc 1.96.0, `stable-x86_64-pc-windows-msvc` |
| CUDA | 12.9 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| MSVC | VS 2022 Community 14.44 |
| `libsimbackend_cuda.dll` + `.lib` | **built** (nvcc + Eigen 3.4.0) at `crates/ppf-cts-solver/src/cpp/build/lib/` |
| Linkage proof | `ppf-contact-solver.exe` links + loads the DLL at runtime (Phase 0 ✅) |
| C-ABI proof | `spike/spike.exe` links `ppf_cts_capi.lib` + `libsimbackend_cuda.lib` + `cudart.lib` and solves a pinned cloth on GPU, penetration-free (Phase 0 console spike ✅, see `spike/`) |

Eigen note: the project requires the **full Eigen 3.4.0** (has the `Core/arch/GPU`
headers nvcc needs). The LightWave third-party Eigen (`eigen34`, a stripped 3.4.90)
is CPU-only and will not compile the CUDA backend.

---

## 1. Prerequisite refactor: expose the driver as a library

The reusable driver logic lives in the **binary** crate `ppf-cts-solver` as
private modules (`mod backend; mod builder; mod mesh; mod data; mod cvec;
mod cvecvec; mod scene; mod args; mod triutils;` in `main.rs`). There is no
`lib.rs`, so nothing outside that binary can call `builder::build`,
`backend::Backend`, `mesh::Mesh::new`, etc.

**Change:** add `crates/ppf-cts-solver/src/lib.rs` that re-exports the modules:

```rust
// src/lib.rs  (new)
pub mod args;
pub mod backend;
pub mod builder;
pub mod cvec;
pub mod cvecvec;
pub mod data;
pub mod mesh;
pub mod scene;
pub mod triutils;
```

`main.rs` then becomes a thin binary that `use`s the lib (or keeps its current
`mod` form behind `#[path]` — but the clean route is `use ppf_cts_solver::...`).
The existing `[[bin]]` target and `build.rs` (CUDA link) are unchanged; a crate
can have both a `lib` and a `bin` target sharing one `build.rs`.

Two free-function helpers the binary keeps to itself today (`save`, `read`,
`read_gz`, `remove_*`, `setup`) stay in `main.rs` — the capi does not need the
file-based checkpoint/output machinery.

> Alternative considered: move the modules into a brand-new `ppf-cts-driver`
> library crate that both the binary and the capi depend on. Cleaner separation,
> but a larger diff and it churns the existing crate. **Recommended: the `lib.rs`
> route** — smallest change, keeps history legible, the binary is unaffected.

---

## 2. `ppf-cts-capi` crate layout

```
crates/ppf-cts-capi/
├── Cargo.toml          # crate-type = ["staticlib"]; deps: ppf-cts-solver (lib), ppf-cts-core
├── cbindgen.toml       # generates include/ppf_cts.h from the #[no_mangle] fns
├── build.rs            # (optional) runs cbindgen to emit the header on build
├── include/
│   └── ppf_cts.h       # generated C header the LightWave plugin #includes
└── src/
    ├── lib.rs          # extern "C" surface (thin); panic = abort guard; opaque handle
    ├── context.rs      # CtsContext: owns scene state across frames (the lifecycle)
    ├── ingest.rs       # in-memory scene ingestion: C arrays -> MeshSet/Props/SimArgs
    │                   #   (this is the one genuinely-new chunk — replaces scene.rs file IO)
    ├── params.rs       # repr(C) CtsGlobalParams / CtsMaterial / CtsPin <-> SimArgs/Props
    └── error.rs        # status codes; catch_unwind so a Rust panic never crosses FFI
```

### Cargo.toml essentials

```toml
[package]
name = "ppf-cts-capi"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
crate-type = ["staticlib"]   # -> ppf_cts_capi.lib (Windows), libppf_cts_capi.a (Unix)

[dependencies]
ppf-cts-solver = { path = "../ppf-cts-solver" }   # now a lib
ppf-cts-core   = { path = "../ppf-cts-core" }
nalgebra.workspace = true
libc = "0.2"

[build-dependencies]
cbindgen = "0.27"   # optional: auto-generate include/ppf_cts.h
```

### Linking model (matches the plan)

`ppf-cts-capi` is a **staticlib**. Its compiled `.lib` contains the Rust driver
**plus unresolved references** to the C++/CUDA FFI symbols (`initialize`,
`advance`, `fetch`, …). Those are resolved at the **final** link — the LightWave
plugin's premake link step adds:

- `ppf_cts_capi.lib` (this crate)
- `libsimbackend_cuda.lib` (the import lib for the CUDA DLL)
- `cudart.lib` (from `%CUDA_PATH%\lib\x64`)

So `ppf-cts-capi` itself needs **no `build.rs` CUDA linkage** — it defers, exactly
like a C++ static lib would. Ship `libsimbackend_cuda.dll` + the CUDA runtime DLLs
next to the `.p`. (The Phase-0 console-exe spike will link these three to prove the
final-link recipe before LightWave is involved.)

---

## 3. C API surface (`ppf_cts.h`)

Design goals: opaque handle, C-only types (no C++), explicit lifecycle mirroring
`backend::Backend::run`, externally driven one-step-at-a-time (LightWave owns the
frame loop), no panics cross the boundary (every fn wraps `catch_unwind`).

### 3.1 Handle, status, lifecycle

```c
typedef struct ppf_cts_context ppf_cts_context;   /* opaque */

typedef enum {
    PPF_CTS_OK                 = 0,
    PPF_CTS_ERR_NULL           = 1,
    PPF_CTS_ERR_NO_CUDA_DEVICE = 2,
    PPF_CTS_ERR_INIT_FAILED    = 3,
    PPF_CTS_ERR_BAD_STATE      = 4,   /* called out of lifecycle order   */
    PPF_CTS_ERR_BAD_ARG        = 5,
    PPF_CTS_ERR_STEP_FAILED    = 6,   /* CCD / PCG / intersection failure */
    PPF_CTS_ERR_PANIC          = 7,   /* a Rust panic was caught          */
} ppf_cts_status;

/* ABI/version guard — caller checks before anything else. */
uint32_t        ppf_cts_abi_version(void);
const char*     ppf_cts_last_error(void);          /* thread-local message */

ppf_cts_context* ppf_cts_create(void);
void             ppf_cts_destroy(ppf_cts_context*);
```

### 3.2 Global parameters (maps to `SimArgs` / `ParamSet`)

```c
typedef struct {
    float gravity[3];
    float dt;              /* solver substep target           */
    double fps;            /* frame cadence for time mapping  */
    float wind[3];
    float air_density;
    float air_friction;
    uint32_t min_newton_steps;
    float    target_toi;
    uint32_t cg_max_iter;
    float    cg_tol;
    int      disable_contact;   /* bool as int for C ABI      */
    /* ... remaining SimArgs fields, all with documented defaults ... */
} ppf_cts_global_params;

void          ppf_cts_default_params(ppf_cts_global_params* out);  /* fills defaults */
ppf_cts_status ppf_cts_set_params(ppf_cts_context*, const ppf_cts_global_params*);
```

### 3.3 Geometry ingestion (replaces `scene.rs` file loading)

Vertices are a shared global pool (rod/face/tet index into it, exactly as
`Mesh::new` expects). Materials use Young's modulus / Poisson ratio + density;
the capi calls `builder::convert_prop` to derive Lamé μ/λ, matching the frontend.

```c
typedef enum { PPF_CTS_MODEL_ARAP=0, PPF_CTS_MODEL_STVK=1,
               PPF_CTS_MODEL_BARAFFWITKIN=2, PPF_CTS_MODEL_SNHK=3 } ppf_cts_model;

typedef struct {
    ppf_cts_model model;
    float young_modulus;
    float poisson_ratio;
    float density;
    float bend;            /* shells/rods                       */
    float friction;
    float strainlimit;     /* 0 = disabled                      */
    float contact_gap;     /* "ghat"                            */
    /* ... per-element-type extras, documented ... */
} ppf_cts_material;

/* All counts are element counts; arrays are flat & caller-owned (copied in).
 * vertices: 3*vertex_count floats. faces: 3*face_count uint32.
 * tets: 4*tet_count uint32. rods: 2*rod_count uint32. Any may be 0/NULL.
 * shell_face_count: leading faces that are cloth shells; the rest are
 * solid contact-only surfaces (tet boundaries). Order faces shell-first.
 * Pass face_count for pure cloth, 0 for pure fracture/solid. */
ppf_cts_status ppf_cts_set_geometry(
    ppf_cts_context*,
    const float*    vertices,     uint32_t vertex_count,
    const uint32_t* faces,        uint32_t face_count,
                                  uint32_t shell_face_count,
    const uint32_t* tets,         uint32_t tet_count,
    const uint32_t* rods,         uint32_t rod_count,
    const ppf_cts_material* material);

/* Static collider mesh (maps to Constraint.mesh / CollisionMesh). */
ppf_cts_status ppf_cts_set_collider(
    ppf_cts_context*,
    const float*    vertices, uint32_t vertex_count,
    const uint32_t* faces,    uint32_t face_count,
    float contact_gap, float friction);
```

### 3.4 Pins / constraints (maps to `FixPair`, per-frame `make_constraint`)

```c
/* Mark vertices as pinnable once (the fix set / fixed_index_table). */
ppf_cts_status ppf_cts_set_pins(
    ppf_cts_context*,
    const uint32_t* vertex_indices, uint32_t count,
    const float* stiffness /* nullable: per-pin, else 1.0 */);

/* Per-frame: push pin target positions (driven by LightWave motion).
 * positions: 3*count floats, aligned with the set_pins index order. */
ppf_cts_status ppf_cts_update_pin_targets(
    ppf_cts_context*,
    const float* positions, uint32_t count);
```

### 3.5 Step + readback (maps to `initialize` then the `advance`/`fetch` loop)

```c
/* Build DataSet/ParamSet and call backend initialize(). Geometry/params/pins
 * must be set first. Allocates GPU state; returns NO_CUDA_DEVICE if absent. */
ppf_cts_status ppf_cts_initialize(ppf_cts_context*);

/* One solver step. Internally: update_constraint -> advance -> fetch.
 * out_sim_time (nullable) receives the post-step sim time. */
ppf_cts_status ppf_cts_step(ppf_cts_context*, double* out_sim_time);

/* Copy current surface vertex positions out. out: 3*vertex_count floats.
 * vertex_count must equal surface_vert_count (queryable below). */
ppf_cts_status ppf_cts_read_positions(
    ppf_cts_context*, float* out, uint32_t vertex_count);

uint32_t ppf_cts_surface_vertex_count(const ppf_cts_context*);

/* Penetration-free assertion hook for the regression harness. */
uint32_t ppf_cts_intersection_count(ppf_cts_context*);
```

### 3.6 Typical caller sequence (LightWave handler)

```c
ppf_cts_context* ctx = ppf_cts_create();
ppf_cts_global_params p; ppf_cts_default_params(&p); p.gravity[1] = -9.8f;
ppf_cts_set_params(ctx, &p);
ppf_cts_set_geometry(ctx, verts, nv, faces, nf, nf, 0,0, 0,0, &mat);
ppf_cts_set_collider(ctx, cverts, cnv, cfaces, cnf, 1e-3f, 0.5f);
ppf_cts_set_pins(ctx, pin_idx, npins, NULL);
if (ppf_cts_initialize(ctx) != PPF_CTS_OK) { /* surface ppf_cts_last_error */ }
for (int frame = 0; frame < nframes; ++frame) {
    ppf_cts_update_pin_targets(ctx, pin_pos_for_frame, npins);
    double t;
    if (ppf_cts_step(ctx, &t) != PPF_CTS_OK) break;   /* solve failed */
    ppf_cts_read_positions(ctx, out_verts, nv);       /* write back to LW geo */
}
ppf_cts_destroy(ctx);
```

---

## 4. Internal mapping (capi → existing Rust)

| C function | Underlying Rust (existing unless noted) |
|---|---|
| `set_params` | build `SimArgs` (`params.rs`, **new**) → `builder::make_param` |
| `set_geometry` | `mesh::Mesh::new` + `MeshSet`; materials via `builder::convert_prop`; **new** `ingest.rs` assembles `Props` (replaces `scene::make_props`) |
| `set_collider` | `builder::make_collision_mesh` |
| `set_pins` / `update_pin_targets` | build `Constraint.fix` (`FixPair`) — replaces `scene::make_constraint` keyframe eval |
| `initialize` | `builder::build` → `backend::Backend::new` → `initialize()` FFI |
| `step` | `update_constraint` → `advance` → `fetch` (lifted from `Backend::run` loop body, minus file IO) |
| `read_positions` | read `state.curr_vertex` (already populated by `fetch_state`) |
| `intersection_count` | `fetch_intersection_records` |

The **only substantial new Rust** is `ingest.rs` + `params.rs`: turning C arrays
into `MeshSet`/`Props`/`SimArgs`/`Constraint` without the TOML+bin file path.
Everything from `builder::build` downward is reused unchanged.

---

## 5. Open questions for confirmation

1. **lib.rs route vs. new `ppf-cts-driver` crate** — recommend lib.rs (§1). OK?
2. **Material grouping** — the solver supports per-element material variation via
   dedup tables. For v1, is **one material per object/group** (cloth, soft body,
   rod each one material) sufficient, or do you need per-face/per-tet materials
   from LightWave surfaces/part tags on day one?
3. **cbindgen** for header generation (keeps `.h` in sync with Rust) vs. a
   hand-written `ppf_cts.h`. Recommend cbindgen.
4. **Phase-0 spike scope**: build `ppf-cts-capi` + a console exe that links all
   three libs and runs `create → set_geometry(small cloth) → initialize → step×N
   → read_positions`, asserting zero intersections. This needs the `ingest.rs`
   minimal path, i.e. it pulls a slice of Phase 1 forward. Acceptable, or keep
   Phase 0 to the bare linkage proof we already have and start `ingest.rs` as
   Phase 1?
