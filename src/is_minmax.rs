use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use polars::chunked_array::ops::{ChunkFull, ChunkSet};

#[polars_expr(output_type=Boolean)]
fn is_max(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let len = s.len();

    match s.arg_max() {
        Some(idx) => {
            let mask: ChunkedArray<BooleanType> = BooleanChunked::full(PlSmallStr::EMPTY, false, len);
            let mask: ChunkedArray<BooleanType> = mask.scatter_single([idx as u32], Some(true))?;
            Ok(mask.into_series())
        }
        None => Ok(BooleanChunked::full(PlSmallStr::EMPTY, false, len).into_series()),
    }
}

