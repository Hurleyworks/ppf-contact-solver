# ppf-cts-capi port status (branch lw-capi, uncommitted)

Source: old-ppf-contact-solver a5fdad3. Target base: 6b30976. Nothing committed.

## State: port done, runs on the GPU (2026-10-17)
- `build-win-native\build-capi-local.bat nocuda` builds target\capi\release\ppf_cts_capi.lib (Release, +crt-static).
- `crates\ppf-cts-capi\spike\build.bat` links the spike against the CUDA DLL backend. It passes: ABI 6, backend "cuda", 441-vertex cloth falls from y=0.15 and rests about 0.0006 above the plane collider, 0 intersections in 30 steps. nvidia-smi lists spike.exe as a compute process.
- `spike\build.bat static` links the /MT static backend (build-mt\lib\libsimbackend_cuda.lib) plus cudart_static.lib into build-static\spike.exe (39.6 MB). Its only imports are system DLLs (kernel32, ntdll, bcryptprimitives). It passes from a folder with no DLLs and CUDA removed from PATH.
- `cargo build --release -p ppf-cts-solver` (normal target dir) builds the upstream binary after the lib/bin split.

## Files changed
- Cargo.toml: workspace member crates/ppf-cts-capi.
- crates/ppf-cts-capi/: copied from a5fdad3 and adapted to the new solver: ABI 6, `ppf_cts_backend_name()`, LockSceneData, CosseratSceneData, Dirichlet pins, new MeshSet/param fields, SimArgs (ccd_eps, max_newton_steps).
  - context.rs: the dataset is moved into `self.dataset` BEFORE `initialize_in_memory`. The driver keeps a pointer to the DataSet (its SCENE slot), just as it does for the ParamSet. Passing the local made the first step fail with "constraint.fix[0] names vertex 0 but the scene has 0 vertices".
  - spike/build.bat: adds the cargo registry folder that holds windows.0.52.0.lib (windows_x86_64_msvc) to LIBPATH, and the `static` option.
- crates/ppf-cts-solver: new src/lib.rs (modules made pub), main.rs uses the lib (unused imports removed), backend.rs adds initialize_in_memory/step_in_memory/intersection_count, probe_device and averaged_edge_param made pub.
- build-win-native: build-capi-local.bat, build-cuda-static-local.bat, scripts/derive_mt_jobs.py.

## ABI 7: quiet console (2026-10-17, uncommitted)
- All per-step console lines (`====== advance ======`, `* time = ...`, `------ newton step`, `> lbvh_build...`) go through one function, `ppf-cts-solver/src/driver/log.rs` `message_text`. The CUDA backend prints only errors and env-gated stats (stderr).
- ppf-cts-solver/src/lib.rs: `pub static QUIET_TRANSCRIPT: AtomicBool` (default false, so the binary is unchanged). log.rs `message_text` returns early when it is set. The `.out` stream files, `log::warn!`, `### FATAL`, CUDA error lines and panics are untouched.
- ppf-cts-capi/src/lib.rs: ABI 7, `ppf_cts_set_verbose(int32_t)`. A capi-side VERBOSE flag (default off) is applied by `ppf_cts_set_verbose` and by `ppf_cts_create`, so the C API path is quiet unless the host asks.
- Rebuilt with `build-capi-local.bat nocuda` (no CUDA change; the static CUDA lib is unchanged).

## ABI 8: animated colliders (2026-10-17, uncommitted)
- `ppf_cts_add_animated_collider(ctx, local_vertices, vertex_count, faces, face_count, key_times, key_translations, key_rotations, key_scales, key_count, material)`: one keyframed collider body (times in sim seconds, T/S linear, rotation `w,x,y,z` slerp; double arrays). Built like the frontend's `_setup_pin_shell`: a shell with density/Young/Poisson/bend 0, `collider_vertex_mask` set on its verts (FaceProp.collider via builder.rs), every vertex an exact kinematic FixPair posed by `ppf_cts_core::datamodel::quat::transform_keyframes_step` (the same evaluator as `PinOperation::TransformKeyframes`) at `state.time + dt`, with `step_delta` from the pose at `state.time`.
- context.rs: `set_geometry_ex` now stages raw arrays; the MeshSet is built at `initialize` after `ingest::insert_shells` splices the shells in after the caller's contact verts (before tet-interior verts) and after the caller's shell faces. Caller indices (pins, tets, PDRD) are remapped; `surface_vertex_count`/`read_positions` stay the caller's contact verts (shells excluded).
- `update_pin_targets` after initialize: pins blend linearly in sim time from their current position to the new target, reached at the next 1/fps boundary (was: whole frame delta on the first step). `step_delta` = pose(t+dt) - pose(t). Before initialize targets still apply at once. `pin_step_start` removed.
- Static collider path (`set_collider_ex`) unchanged. No CUDA change; rebuilt with `build-capi-local.bat nocuda`.
- Tests (CPU, `cargo test -p ppf-cts-capi --features ppf-cts-solver/cpu`): insert_shells layout, keyframed pose, bad keys, pin blend.

## Notes
- The solver driver keeps its state in globals (SCENE, SOLVER), so only one live context per process.
- The capi build passes `--features ppf-cts-solver/cuda` and links libsimbackend_cuda + cudart. For a static plugin, swap cudart.lib for cudart_static.lib and use the build-mt lib dir, as `spike\build.bat static` does.
