mod collapse;
mod arg_true;
mod multi_index;
mod arg_minmax;
mod is_minmax;
mod arg_first_null;
use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
