// File: lib.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Library face of the solver driver. The driver otherwise lives entirely
// inside the `ppf-contact-solver` binary, so nothing outside it could call
// `builder::build`, `backend::Backend`, `mesh::Mesh::new`, etc. The
// `ppf-cts-capi` crate (the LightWave embedding) needs exactly those entry
// points, so the modules hang off this `lib.rs` and main.rs is a binary on
// top of it.
//
// The modules reach back to the crate root with `super::` for a handful of
// names (the re-exports below) and for the file-IO helpers
// (`save`/`read`/`read_gz`/`remove_old_files`/`IO_BUF_CAPACITY`). Those
// helpers therefore live here, not in main.rs.
//
// `build.rs` cfgs (`neutral_driver`, `abi_backend_linked`, `cuda_backend`)
// apply to every target of this package, so the library sees the same
// driver and the same dispatch path as the binary, and its link directives
// (the backend library, cudart) reach any final artifact that links it,
// including the capi staticlib's `native-static-libs` list.

pub mod args;
pub mod backend;
pub mod builder;
pub mod cosserat;
// The neutral driver: the Newton loop, the linear solve, the assembly order and
// every dispatch decision, written once and driven through the `Device` seam.
// Whether it is compiled is decided in build.rs (`Backend::links_neutral_driver`).
#[cfg(neutral_driver)]
pub mod driver;
pub mod cvec;
pub mod cvecvec;
pub mod data;
pub mod force_field;
pub mod mesh;
pub mod plastic_state;
pub mod raw_vec;
pub mod scene;
#[cfg(unix)]
pub mod signal_sidecar;
pub mod statistics;
pub mod status_writer;
pub mod triutils;

extern crate nalgebra as na;

// Crate-root names the submodules expect via `super::`.
pub use args::{ProgramArgs, SimArgs};
pub use backend::MeshSet;
pub use builder::Props;
pub use cvec::CVec;
pub use data::{DataSet, ParamSet};
pub use mesh::Mesh as SimMesh;
pub use scene::Scene;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use log::*;
use std::io::Read;

/// Which library this build dispatches the solve through, asked of the
/// loaded library itself (`be_backend_name`). `None` when the build links no
/// backend library, i.e. the solve would run on the host renderings.
pub fn linked_backend_name() -> Option<&'static str> {
    #[cfg(abi_backend_linked)]
    {
        Some(ppf_cts_compute::abi::linked_backend_name())
    }
    #[cfg(not(abi_backend_linked))]
    {
        None
    }
}

/// When true, the solver's console transcript (`driver::log::message_text`:
/// the per-step `====== advance ======`, `* time = ...`, `------ newton step`
/// lines) is not printed. The `.out` stream files and every warning, error and
/// panic path are unaffected. The binary never sets it; the C API sets it by
/// default (`ppf_cts_set_verbose`), since an embedding host owns the console.
pub static QUIET_TRANSCRIPT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The backend `build.rs` selected (`cuda`, `metal`, `rocm`, `cpu`).
pub const SELECTED_BACKEND: &str = env!("PPF_BACKEND");

/// C callback the backend invokes to route its log lines through the Rust
/// logger. `#[no_mangle]` so it is the symbol the C++ side links against;
/// defined here (not in main.rs) so every consumer of the driver library,
/// the binary and the capi staticlib alike, provides it exactly once.
#[no_mangle]
extern "C" fn print_rust(message: *const libc::c_char) {
    let message = unsafe { std::ffi::CStr::from_ptr(message) }
        .to_str()
        .unwrap();
    info!("{message}");
}

fn decompress_from_gz(compressed_data: Vec<u8>, path: &str) -> Vec<u8> {
    let mut decoder = GzDecoder::new(compressed_data.as_slice());
    let mut decompressed_data = Vec::new();
    if let Err(e) = decoder.read_to_end(&mut decompressed_data) {
        panic!(
            "Failed to decompress '{path}': {e}. The file is likely truncated \
             or corrupt, for example an auto-save that was interrupted or an \
             incomplete file transfer. Resuming from this checkpoint is not \
             possible; re-run the simulation or resume from a valid checkpoint."
        );
    }
    decompressed_data
}

/// 64 KiB I/O write buffer.
pub const IO_BUF_CAPACITY: usize = 64 * 1024;

/// Streams serialization directly to a gzip-compressed file without buffering entire data in RAM.
///
/// The bytes are written to a temporary sibling file and then atomically
/// renamed into `path`. A process killed mid-write (e.g. an interrupted
/// auto-save) therefore leaves either the previous valid file or the fully
/// written new one, never a truncated gzip that fails to decompress on resume
/// with `UnexpectedEof`. The temp file is fsynced before the rename so a crash
/// right after the rename cannot expose partially flushed contents.
pub fn save<T: serde::Serialize>(obj: &T, path: &str) {
    let tmp_path = format!("{}.tmp.{}", path, std::process::id());
    {
        let file = std::fs::File::create(&tmp_path).unwrap();
        let buf_writer = std::io::BufWriter::with_capacity(IO_BUF_CAPACITY, file);
        let mut encoder = GzEncoder::new(buf_writer, Compression::default());
        bincode::serialize_into(&mut encoder, obj).unwrap();
        let buf_writer = encoder.finish().unwrap();
        let file = buf_writer.into_inner().unwrap();
        file.sync_all().unwrap();
    }
    std::fs::rename(&tmp_path, path).unwrap();
}

pub fn read_gz(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read '{path}': {e}"));
    decompress_from_gz(bytes, path)
}

pub fn read<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> T {
    // bincode is POSITIONAL: it records no field names and honors no
    // `serde(default)`, so a checkpoint written by a build whose structs have
    // since gained a field cannot be read back, and the failure arrives here
    // as an opaque complaint about trailing bytes or an implausible length.
    // Name the likely cause instead. The dataset is write-once per session
    // (see the checkpoint layout notes), so there is no self-healing path and
    // the answer is always to rebuild the scene rather than to retry.
    let type_name = std::any::type_name::<T>();
    bincode::deserialize(data).unwrap_or_else(|e| {
        panic!(
            "Failed to deserialize a checkpoint of type {type_name}: {e}.\n\
             This usually means it was written by a build with a different \
             in-memory layout for {type_name}, which bincode cannot migrate. \
             Rebuild the scene to produce a fresh session directory; an \
             existing dataset.bin.gz cannot be upgraded in place."
        )
    })
}

pub fn remove_old_files(dirpath: &str, prefix: &str, suffix: &str, keep_number: i32, max_frame: i32) {
    if keep_number <= 0 {
        return;
    }

    let mut n = 0;
    for i in 0..=max_frame {
        let path = format!("{dirpath}/{prefix}{i}{suffix}");
        if std::path::Path::new(&path).exists() {
            n += 1;
        }
    }

    let mut i = 0;
    while n > keep_number {
        let filename = format!("{prefix}{i}{suffix}");
        let path = format!("{dirpath}/{filename}");
        if std::path::Path::new(&path).exists() {
            info!("Removing {filename}...");
            std::fs::remove_file(path).unwrap_or(());
            n -= 1;
        }
        i += 1;
        if i > max_frame {
            break;
        }
    }
}
