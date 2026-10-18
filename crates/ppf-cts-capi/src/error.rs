// File: crates/ppf-cts-capi/src/error.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Status codes, a thread-local last-error message, and the panic guard.
// Every `extern "C"` entry point routes through `guard` so a Rust panic
// is caught and reported as `PPF_CTS_ERR_PANIC` instead of unwinding
// across the FFI boundary (which is undefined behavior).

use std::cell::RefCell;
use std::ffi::{c_char, CString};

/// C-visible status enum. Mirrors `ppf_cts_status` in DESIGN.md §3.1.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CtsStatus {
    Ok = 0,
    Null = 1,
    NoCudaDevice = 2,
    InitFailed = 3,
    /// Called out of lifecycle order (e.g. step before initialize).
    BadState = 4,
    BadArg = 5,
    /// CCD / PCG / intersection failure during a step.
    StepFailed = 6,
    /// A Rust panic was caught at the boundary.
    Panic = 7,
    /// The GPU failed during a step (out of memory, a lost context, a
    /// failed launch). The process stays alive; the context can only be
    /// destroyed. [`crate::ppf_cts_last_error`] names the CUDA error.
    DeviceFault = 8,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

/// Record a thread-local error message surfaced via `ppf_cts_last_error`.
pub fn set_last_error(msg: impl Into<Vec<u8>>) {
    // Interior NULs can't live in a CString; fall back to a fixed note so
    // the caller still gets a pointer rather than a crash.
    let cstring = CString::new(msg)
        .unwrap_or_else(|_| CString::new("error message contained interior NUL").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = cstring);
}

/// Borrow the thread-local error message as a `*const c_char`. The pointer
/// is valid until the next call that mutates the thread-local on this
/// thread; callers should copy it if they need to retain it.
pub fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Run `f`, catching any panic and converting it to `CtsStatus::Panic`
/// (with the panic payload recorded as the last error). `AssertUnwindSafe`
/// is sound here: the only state crossing the boundary is the opaque
/// context pointer and caller-owned buffers, and on panic we report an
/// error rather than observing partially-mutated state as valid.
pub fn guard<F: FnOnce() -> CtsStatus>(f: F) -> CtsStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(payload) => {
            let msg = panic_message(&payload);
            set_last_error(format!("panic in ppf-cts-capi: {msg}"));
            CtsStatus::Panic
        }
    }
}

/// Variant of [`guard`] for entry points that return a plain value rather
/// than a status. On panic the error is recorded and `on_panic` returned.
pub fn guard_value<T, F: FnOnce() -> T>(on_panic: T, f: F) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let msg = panic_message(&payload);
            set_last_error(format!("panic in ppf-cts-capi: {msg}"));
            on_panic
        }
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
