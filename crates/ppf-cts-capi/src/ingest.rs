// File: crates/ppf-cts-capi/src/ingest.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// In-memory scene ingestion: flat C arrays -> `MeshSet` and `Props`. This
// replaces the TOML+bin file path of `scene.rs` for the embedding. The
// `Props` assembly mirrors `scene::make_props` followed by the shell-edge
// append from `main.rs` (lines 112-183): `make_props` only fills rod edges
// (0..rod_count), and the shell contact edges (rod_count..edge_count) must
// be appended with area-weighted ghat/offset/friction from their adjacent
// faces — without them cloth self-contact along shell edges is missing.

use std::collections::HashMap;

use nalgebra::{Matrix2xX, Matrix3xX, Matrix4xX};
use ppf_cts_solver::backend::MeshSet;
use ppf_cts_solver::builder::{convert_prop, Props};
use ppf_cts_solver::data::{
    EdgeParam, EdgeProp, FaceParam, FaceProp, Model, TetParam, TetProp,
};
use ppf_cts_solver::mesh::Mesh;

use crate::params::{CtsColliderMaterial, CtsMaterial};

/// Assemble a `MeshSet` from flat, caller-owned index/vertex arrays.
///
/// `verts` is `3*vertex_count` floats (xyz per vertex, the shared global
/// pool rod/face/tet index into). `faces`/`tets`/`rods` are flat element
/// index arrays.
///
/// `shell_face_count` is the number of *leading* faces treated as cloth
/// shells (membrane + bending elasticity); the remaining
/// `face_count - shell_face_count` faces are **solid contact-only**
/// surfaces — they carry no shell energy and exist purely so the tets'
/// boundary participates in contact (see `builder.rs` face/hinge typing
/// and the `shell_face_count`-bounded energy dispatch in `energy.cu`).
/// Callers must order faces shell-first. Pure-cloth callers pass
/// `shell_face_count == face_count`; pure fracture/solid callers pass `0`.
/// Values larger than `face_count` are clamped.
pub fn build_meshset(
    verts: &[f32],
    faces: &[u32],
    tets: &[u32],
    rods: &[u32],
    shell_face_count: usize,
) -> MeshSet {
    let vertex = Matrix3xX::<f32>::from_column_slice(verts);

    let face_idx: Vec<usize> = faces.iter().map(|&x| x as usize).collect();
    let face = Matrix3xX::<usize>::from_column_slice(&face_idx);
    let shell_face_count = shell_face_count.min(face.ncols());

    let tet_idx: Vec<usize> = tets.iter().map(|&x| x as usize).collect();
    let tet = if tet_idx.is_empty() {
        Matrix4xX::<usize>::zeros(0)
    } else {
        Matrix4xX::<usize>::from_column_slice(&tet_idx)
    };

    let rod_idx: Vec<usize> = rods.iter().map(|&x| x as usize).collect();
    let rod = if rod_idx.is_empty() {
        Matrix2xX::<usize>::zeros(0)
    } else {
        Matrix2xX::<usize>::from_column_slice(&rod_idx)
    };

    let n_vertices = vertex.ncols();
    let mesh = Mesh::new(rod, face, tet, shell_face_count, n_vertices);
    MeshSet {
        mesh,
        uv: None,
        vertex,
        rest_vertex: None,
        bend_rest_vertex: None,
        bend_rest_vertex_mask: Vec::new(),
        // The collider is passed as the constraint's CollisionMesh, not as a
        // STATIC object in the solved namespace, so no vertex is a collider.
        collider_vertex_mask: Vec::new(),
        // One object, one group, default intersection policy (allow
        // nothing), no pre-existing intersections linked.
        object_vertex_index: Vec::new(),
        group_vertex_index: Vec::new(),
        intersect_policy: Vec::new(),
        start_link: Vec::new(),
    }
}

fn dedup<K>(map: &mut HashMap<K, u32>, store: &mut Vec<K>, key: K) -> u32
where
    K: std::hash::Hash + Eq + Copy,
{
    *map.entry(key).or_insert_with(|| {
        let idx = store.len() as u32;
        store.push(key);
        idx
    })
}

/// Build `Props` from a per-element material table. Mirrors
/// `scene::make_props` (rod edges, faces, tets) and then appends the shell
/// contact edges exactly like `main.rs:112-183`. `mats` is the material
/// table (at least one row); the optional `*_mat_idx` arrays select a row
/// per element (None => every element of that kind uses row 0, which for a
/// one-row table is byte-identical to the old uniform path).
#[allow(clippy::too_many_arguments)]
pub fn build_props(
    mesh: &MeshSet,
    mats: &[CtsMaterial],
    face_mat_idx: Option<&[u32]>,
    tet_mat_idx: Option<&[u32]>,
    rod_mat_idx: Option<&[u32]>,
    face_area: &[f32],
    tet_volumes: &[f32],
    pdrd_vert_index: &[u32],
) -> Props {
    let mm = &mesh.mesh.mesh;
    let rod_count = mm.rod_count;

    // A face/tet whose every vertex carries a non-zero PDRD body id is a
    // rigid-body element: it gets `Model::Pdrd` (no elastic energy), zero
    // Lame parameters, and zero element mass (volumetric mass is added
    // per-body in `builder::build`). Mirrors `scene.rs` PDRD handling.
    // With no PDRD inputs this predicate is always false and every element
    // keeps the uniform material, leaving the parity path byte-identical.
    let is_pdrd = |verts: &[usize]| -> bool {
        !pdrd_vert_index.is_empty()
            && verts.iter().all(|&v| pdrd_vert_index[v] != 0)
    };

    // ----- Rod edges (0..rod_count) -----
    let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
    let mut edge_params: Vec<EdgeParam> = Vec::new();
    let mut edge: Vec<EdgeProp> = Vec::new();
    for i in 0..rod_count {
        let m = &mats[rod_mat_idx.map_or(0, |a| a[i] as usize)];
        let r = mm.edge.column(i);
        let length = (mesh.vertex.column(r[1]) - mesh.vertex.column(r[0])).norm();
        let param = EdgeParam {
            stiffness: m.young_modulus,
            bend: m.bend,
            ghat: m.contact_gap,
            offset: m.contact_offset,
            friction: m.friction,
            strainlimit: m.strainlimit,
            plasticity: 0.0,
            plasticity_threshold: 0.0,
            bend_rest_from_geometry: false,
            deform_damping: 0.0,
            bend_damping: 0.0,
            // Not a Cosserat rod: every cosserat_* field zero.
            ..Default::default()
        };
        let param_index = dedup(&mut edge_param_map, &mut edge_params, param);
        edge.push(EdgeProp {
            length,
            initial_length: length,
            mass: m.density * length,
            fixed: false,
            param_index,
            pin_allow_intersection: false,
        });
    }

    // ----- Faces -----
    let mut face_param_map: HashMap<FaceParam, u32> = HashMap::new();
    let mut face_params: Vec<FaceParam> = Vec::new();
    let face: Vec<FaceProp> = (0..mm.face.ncols())
        .map(|i| {
            let m = &mats[face_mat_idx.map_or(0, |a| a[i] as usize)];
            let (mu, lambda) = convert_prop(m.young_modulus, m.poisson_ratio);
            let model = m.model.to_model();
            let col = mm.face.column(i);
            let pdrd = is_pdrd(&[col[0], col[1], col[2]]);
            // PDRD faces carry zero elastic Lame params + `Model::Pdrd`
            // (skip-everything in energy.cu) and zero face mass (the body
            // pass adds volumetric mass). Non-PDRD faces use the material.
            let face_param = FaceParam {
                model: if pdrd { Model::Pdrd } else { model },
                mu: if pdrd { 0.0 } else { mu },
                lambda: if pdrd { 0.0 } else { lambda },
                friction: m.friction,
                ghat: m.contact_gap,
                offset: m.contact_offset,
                bend: m.bend,
                strainlimit: m.strainlimit,
                shrink_x: 1.0,
                shrink_y: 1.0,
                pressure: 0.0,
                plasticity: 0.0,
                plasticity_threshold: 0.0,
                bend_plasticity: 0.0,
                bend_plasticity_threshold: 0.0,
                bend_rest_from_geometry: false,
                deform_damping: 0.0,
                bend_damping: 0.0,
                // Isotropic bending.
                bend_warp: 0.0,
                bend_weft: 0.0,
            };
            let area = face_area[i];
            let param_index = dedup(&mut face_param_map, &mut face_params, face_param);
            FaceProp {
                area,
                mass: if pdrd { 0.0 } else { m.density * area },
                fixed: false,
                rest_excluded: false,
                // Not a STATIC collider in the solved namespace.
                collider: false,
                param_index,
                pin_allow_intersection: false,
            }
        })
        .collect();

    // ----- Tets -----
    let mut tet_param_map: HashMap<TetParam, u32> = HashMap::new();
    let mut tet_params: Vec<TetParam> = Vec::new();
    let tet: Vec<TetProp> = (0..mm.tet.ncols())
        .map(|i| {
            let m = &mats[tet_mat_idx.map_or(0, |a| a[i] as usize)];
            let (mu, lambda) = convert_prop(m.young_modulus, m.poisson_ratio);
            let model = m.model.to_model();
            let col = mm.tet.column(i);
            let pdrd = is_pdrd(&[col[0], col[1], col[2], col[3]]);
            let volume = tet_volumes[i];
            let param = TetParam {
                model: if pdrd { Model::Pdrd } else { model },
                mu: if pdrd { 0.0 } else { mu },
                lambda: if pdrd { 0.0 } else { lambda },
                shrink: 1.0,
                plasticity: 0.0,
                plasticity_threshold: 0.0,
                deform_damping: 0.0,
            };
            let param_index = dedup(&mut tet_param_map, &mut tet_params, param);
            TetProp {
                mass: if pdrd { 0.0 } else { m.density * volume },
                volume,
                fixed: false,
                rest_excluded: false,
                param_index,
            }
        })
        .collect();

    let mut props = Props {
        edge,
        face,
        tet,
        edge_params,
        face_params,
        tet_params,
        sand: None,
        animated_materials: false,
    };

    append_shell_edges(mesh, face_area, &mut props);
    props
}

/// Append contact-only props for the shell edges (rod_count..edge_count).
/// Each shell edge takes area-weighted ghat/offset/friction from its
/// adjacent faces; the elastic fields are zero (bending/stretch live on
/// hinges/faces, not these edges). Replicates `main.rs:112-183`.
fn append_shell_edges(mesh: &MeshSet, face_area: &[f32], props: &mut Props) {
    let mm = &mesh.mesh.mesh;
    let rod_count = mm.rod_count;
    let edge_count = mm.edge.ncols();

    // Seed the dedup map with the rod-edge params already in `props`.
    let mut edge_param_map: HashMap<EdgeParam, u32> = HashMap::new();
    for (i, param) in props.edge_params.iter().enumerate() {
        edge_param_map.insert(*param, i as u32);
    }

    for i in rod_count..edge_count {
        let r = mm.edge.column(i);
        let length = (mesh.vertex.column(r[1]) - mesh.vertex.column(r[0])).norm();
        let mut ghat_sum = 0.0;
        let mut offset_sum = 0.0;
        let mut friction_sum = 0.0;
        let mut area_sum = 0.0;
        for &j in mesh.mesh.neighbor.edge.face[i].iter() {
            let face_prop = &props.face[j];
            let face_param = &props.face_params[face_prop.param_index as usize];
            let area = face_area[j];
            ghat_sum += area * face_param.ghat;
            friction_sum += area * face_param.friction;
            offset_sum += area * face_param.offset;
            area_sum += area;
        }
        assert!(area_sum > 0.0, "shell edge {i} has no adjacent faces");
        let param = EdgeParam {
            stiffness: 0.0,
            bend: 0.0,
            ghat: ghat_sum / area_sum,
            offset: offset_sum / area_sum,
            friction: friction_sum / area_sum,
            strainlimit: 0.0,
            plasticity: 0.0,
            plasticity_threshold: 0.0,
            bend_rest_from_geometry: false,
            deform_damping: 0.0,
            bend_damping: 0.0,
            ..Default::default()
        };
        let param_index = dedup(&mut edge_param_map, &mut props.edge_params, param);
        props.edge.push(EdgeProp {
            length,
            initial_length: length,
            mass: 0.0,
            fixed: false,
            param_index,
            pin_allow_intersection: false,
        });
    }
}

/// Build the collider geometry/props for `make_collision_mesh`. One
/// `FaceParam` (only `ghat`/`offset`/`friction` matter for collision) is
/// built per collider-material row; each face selects its row via
/// `face_mat_idx` (None => every face uses row 0). Returns the pieces the
/// per-step constraint rebuild needs (`CollisionMesh` is not `Clone`, so the
/// mesh itself is rebuilt each step from these).
pub fn build_collider(
    verts: &[f32],
    faces: &[u32],
    mats: &[CtsColliderMaterial],
    face_mat_idx: Option<&[u32]>,
) -> (Matrix3xX<f32>, Matrix3xX<usize>, Vec<FaceProp>, Vec<FaceParam>) {
    let vertex = Matrix3xX::<f32>::from_column_slice(verts);
    let face_idx: Vec<usize> = faces.iter().map(|&x| x as usize).collect();
    let face = Matrix3xX::<usize>::from_column_slice(&face_idx);

    let face_params: Vec<FaceParam> = mats
        .iter()
        .map(|m| FaceParam {
            ghat: m.contact_gap,
            friction: m.friction,
            ..Default::default()
        })
        .collect();

    let face_props: Vec<FaceProp> = (0..face.ncols())
        .map(|i| {
            let area = ppf_cts_solver::triutils::area(&vertex, &face, i);
            FaceProp {
                area,
                mass: 0.0,
                fixed: false,
                rest_excluded: false,
                collider: false,
                param_index: face_mat_idx.map_or(0, |a| a[i]),
                pin_allow_intersection: false,
            }
        })
        .collect();

    (vertex, face, face_props, face_params)
}
