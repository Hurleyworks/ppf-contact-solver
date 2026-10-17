// File: crates/ppf-cts-capi/spike/spike.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Console spike for ppf-cts-capi (ABI 6, neutral-driver solver).
//
// Proves the final-link recipe end-to-end on a real GPU: a C++ host built
// /MT (the representative LightWave target) links
//   * ppf_cts_capi.lib          (the Rust C-ABI driver, staticlib, +crt-static)
//   * libsimbackend_cuda.lib    (the kernels-only CUDA backend library)
//   * cudart.lib                (CUDA runtime)
// plus the native system libs the Rust runtime needs (see build.bat).
//
// Scene: a 20x20-quad cloth sheet, two corners of one edge pinned, falling
// onto a static collider plane at y = 0. Lifecycle:
//   create -> default_params/default_material -> set_params -> set_geometry
//   -> set_collider -> set_pins -> initialize -> (update_pin_targets, step,
//   read_positions, intersection_count) x STEPS -> destroy
// and asserts the backend is the GPU and the solve stays penetration-free.

#include "ppf_cts.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t GRID = 20;            // quads per side
constexpr uint32_t SIDE = GRID + 1;      // vertices per side
constexpr float SPACING = 0.05f;         // 1m x 1m cloth
constexpr float HEIGHT = 0.15f;          // cloth starts this far above the plane
constexpr int STEPS = 30;
constexpr uint32_t EXPECTED_ABI = 6;

uint32_t vid(uint32_t i, uint32_t j) { return i * SIDE + j; }

} // namespace

int main() {
    std::printf("ppf-cts-capi spike\n");
    const uint32_t abi = ppf_cts_abi_version();
    const char *backend = ppf_cts_backend_name();
    std::printf("  abi version = %u (header expects %u)\n", abi, EXPECTED_ABI);
    std::printf("  backend     = %s\n", backend);
    if (abi != EXPECTED_ABI) {
        std::fprintf(stderr, "FAIL: ABI mismatch\n");
        return 1;
    }
    if (std::strcmp(backend, "host") == 0) {
        std::fprintf(stderr, "FAIL: no GPU backend linked\n");
        return 1;
    }

    CtsContext *ctx = ppf_cts_create();
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ppf_cts_create returned null\n");
        return 1;
    }

    auto fail = [&](const char *where, CtsStatus st) -> int {
        std::fprintf(stderr, "FAIL: %s -> status %d: %s\n", where,
                     static_cast<int>(st), ppf_cts_last_error());
        ppf_cts_destroy(ctx);
        return 1;
    };

    // --- Parameters -------------------------------------------------------
    CtsGlobalParams params;
    CtsStatus st = ppf_cts_default_params(&params);
    if (st != CTS_STATUS_OK) return fail("default_params", st);
    params.gravity[0] = 0.0f;
    params.gravity[1] = -9.8f;
    params.gravity[2] = 0.0f;
    params.dt = 0.01f;
    st = ppf_cts_set_params(ctx, &params);
    if (st != CTS_STATUS_OK) return fail("set_params", st);

    // --- Geometry: triangulated cloth grid in XZ at y = HEIGHT -------------
    std::vector<float> verts;
    verts.reserve(SIDE * SIDE * 3);
    for (uint32_t i = 0; i < SIDE; ++i) {
        for (uint32_t j = 0; j < SIDE; ++j) {
            verts.push_back(static_cast<float>(i) * SPACING);
            verts.push_back(HEIGHT);
            verts.push_back(static_cast<float>(j) * SPACING);
        }
    }
    const uint32_t vertex_count = SIDE * SIDE;

    std::vector<uint32_t> faces;
    faces.reserve(GRID * GRID * 6);
    for (uint32_t i = 0; i < GRID; ++i) {
        for (uint32_t j = 0; j < GRID; ++j) {
            const uint32_t v00 = vid(i, j);
            const uint32_t v10 = vid(i + 1, j);
            const uint32_t v01 = vid(i, j + 1);
            const uint32_t v11 = vid(i + 1, j + 1);
            faces.push_back(v00); faces.push_back(v11); faces.push_back(v10);
            faces.push_back(v00); faces.push_back(v01); faces.push_back(v11);
        }
    }
    const uint32_t face_count = GRID * GRID * 2;

    CtsMaterial mat;
    st = ppf_cts_default_material(&mat);
    if (st != CTS_STATUS_OK) return fail("default_material", st);

    st = ppf_cts_set_geometry(ctx, verts.data(), vertex_count,
                              faces.data(), face_count,
                              face_count, /* shell_face_count: all cloth */
                              nullptr, 0, nullptr, 0, &mat);
    if (st != CTS_STATUS_OK) return fail("set_geometry", st);
    std::printf("  geometry: %u verts, %u faces\n", vertex_count, face_count);

    // --- Static collider: a large plane at y = 0, normal +y ---------------
    const float plane_verts[12] = {
        -2.0f, 0.0f, -2.0f,
        -2.0f, 0.0f,  3.0f,
         3.0f, 0.0f,  3.0f,
         3.0f, 0.0f, -2.0f,
    };
    const uint32_t plane_faces[6] = {0, 1, 2, 0, 2, 3};
    st = ppf_cts_set_collider(ctx, plane_verts, 4, plane_faces, 2,
                              1e-3f /* contact_gap */, 0.5f /* friction */);
    if (st != CTS_STATUS_OK) return fail("set_collider", st);

    // --- Pins: the two corners of the first row ---------------------------
    const uint32_t pin_idx[2] = {vid(0, 0), vid(0, GRID)};
    st = ppf_cts_set_pins(ctx, pin_idx, 2, nullptr);
    if (st != CTS_STATUS_OK) return fail("set_pins", st);
    float pin_pos[6];
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < 3; ++k)
            pin_pos[p * 3 + k] = verts[pin_idx[p] * 3 + k];

    // --- Initialize (allocates GPU state) ---------------------------------
    st = ppf_cts_initialize(ctx);
    if (st != CTS_STATUS_OK) return fail("initialize", st);
    std::printf("  initialized; surface_vertex_count = %u\n",
                ppf_cts_surface_vertex_count(ctx));

    // --- Step loop --------------------------------------------------------
    std::vector<float> out(vertex_count * 3, 0.0f);
    int rc = 0;
    float final_min_y = 0.0f;
    for (int step = 0; step < STEPS; ++step) {
        st = ppf_cts_update_pin_targets(ctx, pin_pos, 2);
        if (st != CTS_STATUS_OK) { rc = fail("update_pin_targets", st); break; }

        double sim_time = 0.0;
        st = ppf_cts_step(ctx, &sim_time);
        if (st != CTS_STATUS_OK) { rc = fail("step", st); break; }

        st = ppf_cts_read_positions(ctx, out.data(), vertex_count);
        if (st != CTS_STATUS_OK) { rc = fail("read_positions", st); break; }

        const uint32_t isect = ppf_cts_intersection_count(ctx);
        float min_y = out[1], max_y = out[1];
        for (uint32_t v = 0; v < vertex_count; ++v) {
            const float y = out[v * 3 + 1];
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
        }
        final_min_y = min_y;
        if (step % 5 == 0 || step == STEPS - 1 || isect != 0)
            std::printf("  step %2d  t=%.3f  min_y=%+.5f  max_y=%+.5f  intersections=%u\n",
                        step, sim_time, min_y, max_y, isect);
        if (isect != 0) {
            std::fprintf(stderr, "FAIL: %u intersections at step %d\n", isect, step);
            rc = 1;
            break;
        }
    }

    ppf_cts_destroy(ctx);

    if (rc == 0) {
        if (final_min_y < -0.01f) {
            std::fprintf(stderr, "FAIL: cloth passed through the plane (min_y=%f)\n",
                         final_min_y);
            return 1;
        }
        std::printf("PASS: %d steps on backend '%s', penetration-free.\n", STEPS, backend);
    }
    return rc;
}
