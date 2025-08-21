use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=String)]
fn multi_index(inputs: &[Series]) -> PolarsResult<Series> {
    let idx_series: &Series = &inputs[0];

    let _lookup: &Series = &inputs[1];
    let lookup: &ChunkedArray<StringType> = _lookup.str()?;

    let out: ChunkedArray<StringType> = match idx_series.dtype() {
        DataType::UInt32 => {
            let idx: &ChunkedArray<UInt32Type> = idx_series.u32()?;
            lookup.take(idx)?
        }
        DataType::Int32 => {
            // reinterpret as u32
            let idx: &ChunkedArray<Int32Type> = idx_series.i32()?;
            let idx_u32: UInt32Chunked = idx.cast(&DataType::UInt32)?.u32()?.clone();
            lookup.take(&idx_u32)?
        }
        DataType::Int64 => {
            // Downcast to 32 (not sure if this is good or bad!)
            let idx: &ChunkedArray<Int64Type> = idx_series.i64()?;
            let idx_u32: UInt32Chunked = idx.cast(&DataType::UInt32)?.u32()?.clone();
            lookup.take(&idx_u32)?
        }
        dt => {
            return Err(PolarsError::ComputeError(
                format!("Unsupported index dtype: {:?}", dt).into(),
            ))
        }
    };

    Ok(out.into_series())
}
