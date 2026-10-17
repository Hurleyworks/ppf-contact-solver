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

## Notes
- The solver driver keeps its state in globals (SCENE, SOLVER), so only one live context per process.
- The capi build passes `--features ppf-cts-solver/cuda` and links libsimbackend_cuda + cudart. For a static plugin, swap cudart.lib for cudart_static.lib and use the build-mt lib dir, as `spike\build.bat static` does.
