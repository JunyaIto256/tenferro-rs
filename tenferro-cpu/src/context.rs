use std::env;

use crate::{Error, Result};

/// Reusable CPU execution context carrying CPU parallelism policy.
///
/// `CpuContext` stores the requested thread count as a kernel-level
/// parallelism hint. It does not own a Rayon thread pool; faer-backed kernels
/// use Rayon through faer's global/current-pool integration.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuContext;
///
/// let ctx = CpuContext::with_threads(1);
/// let value = ctx.install(|| 1 + 1);
/// assert_eq!(value, 2);
/// assert_eq!(ctx.num_threads(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct CpuContext {
    num_threads: usize,
}

impl CpuContext {
    /// Create a CPU context from `TENFERRO_NUM_THREADS` (preferred) or
    /// `RAYON_NUM_THREADS`, falling back to the process-visible CPU count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::from_env();
    /// assert!(ctx.num_threads() >= 1);
    /// ```
    pub fn from_env() -> Self {
        Self::try_from_env()
            .unwrap_or_else(|_| Self::with_threads(super::affinity::available_parallelism()))
    }

    /// Try to create a CPU context from `TENFERRO_NUM_THREADS`, falling back
    /// to `RAYON_NUM_THREADS` when the former is not set.
    ///
    /// `TENFERRO_NUM_THREADS` allows separating tenferro/faer gemm parallelism
    /// from the application rayon pool.  Setting it to 1 prevents deadlocks
    /// where a rayon worker holding `Mutex<CpuBackend>` spawns faer work into
    /// the same rayon pool via `Par::rayon(…)` → circular wait.
    /// See `~/qtt_runs/futex_hang_bt_38981_sepmpo_ns2mu2.txt` for the
    /// full backtrace analysis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_from_env()
    ///     .unwrap_or_else(|_| CpuContext::with_threads(1));
    /// assert!(ctx.num_threads() >= 1);
    /// ```
    pub fn try_from_env() -> Result<Self> {
        // TENFERRO_NUM_THREADS takes priority (futex deadlock fix).
        match env::var("TENFERRO_NUM_THREADS") {
            Ok(value) => {
                let num_threads = value.parse::<usize>().map_err(|err| Error::InvalidConfig {
                    op: "CpuContext::try_from_env",
                    message: format!("invalid TENFERRO_NUM_THREADS value {value:?}: {err}"),
                })?;
                Self::try_with_threads(num_threads).map_err(|err| match err {
                    Error::InvalidConfig { message, .. } => Error::InvalidConfig {
                        op: "CpuContext::try_from_env",
                        message: format!(
                            "invalid TENFERRO_NUM_THREADS value {value:?}: {message}"
                        ),
                    },
                    err => err,
                })
            }
            Err(env::VarError::NotPresent) => {
                // Fall back to RAYON_NUM_THREADS (traditional behaviour).
                match env::var("RAYON_NUM_THREADS") {
                    Ok(value) => {
                        let num_threads =
                            value.parse::<usize>().map_err(|err| Error::InvalidConfig {
                                op: "CpuContext::try_from_env",
                                message: format!(
                                    "invalid RAYON_NUM_THREADS value {value:?}: {err}"
                                ),
                            })?;
                        Self::try_with_threads(num_threads).map_err(|err| match err {
                            Error::InvalidConfig { message, .. } => Error::InvalidConfig {
                                op: "CpuContext::try_from_env",
                                message: format!(
                                    "invalid RAYON_NUM_THREADS value {value:?}: {message}"
                                ),
                            },
                            err => err,
                        })
                    }
                    Err(env::VarError::NotPresent) => {
                        Self::try_with_threads(super::affinity::available_parallelism())
                    }
                    Err(err) => Err(Error::InvalidConfig {
                        op: "CpuContext::try_from_env",
                        message: format!("failed to read RAYON_NUM_THREADS: {err}"),
                    }),
                }
            }
            Err(err) => Err(Error::InvalidConfig {
                op: "CpuContext::try_from_env",
                message: format!("failed to read TENFERRO_NUM_THREADS: {err}"),
            }),
        }
    }

    /// Create a CPU context with a fixed parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        match Self::try_with_threads(num_threads) {
            Ok(ctx) => ctx,
            Err(err) => panic!("{err}"),
        }
    }

    /// Try to create a CPU context with a fixed parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_with_threads(1).unwrap();
    /// assert_eq!(ctx.num_threads(), 1);
    /// ```
    pub fn try_with_threads(num_threads: usize) -> Result<Self> {
        if num_threads == 0 {
            return Err(Error::InvalidConfig {
                op: "CpuContext::try_with_threads",
                message: "thread count must be at least 1".into(),
            });
        }
        Ok(Self { num_threads })
    }

    /// Return this context's CPU parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Run a closure on the caller thread.
    ///
    /// This method preserves the public execution-scope API without creating or
    /// entering a tenferro-owned Rayon pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(1);
    /// let value = ctx.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        op()
    }

    /// Return the faer parallelism policy for this context.
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn faer_par(&self) -> faer::Par {
        if self.num_threads == 1 {
            faer::Par::Seq
        } else {
            faer::Par::rayon(self.num_threads)
        }
    }
}

#[cfg(test)]
mod tests;
