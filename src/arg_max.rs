use polars::datatypes::PlSmallStr;
use polars::prelude::*;
use polars_arrow::legacy::trusted_len::TrustedLenPush;
use pyo3_polars::derive::polars_expr;

fn _check_types(inputs: &[Series]) -> PolarsResult<()> {
    let first_type: &DataType = inputs[0].dtype();
    for s in inputs.iter().skip(1) {
        if s.dtype() != first_type {
            return Err(PolarsError::SchemaMismatch(
                "All input Series must have the same data type".into(),
            ));
        }
    }
    Ok(())
}

macro_rules! impl_argmax_for_type {
    ($inputs:expr, $len:expr, $polars_type:ident, $rust_type:ty) => {{
        let typed_inputs: Vec<_> = $inputs.iter().map(|s| s.$polars_type().unwrap()).collect();
        let mut result = Vec::with_capacity($len);

        for row_idx in 0..$len {
            let mut max_idx: Option<u32> = None;
            let mut max_value: Option<$rust_type> = None;

            for (col_idx, ca) in typed_inputs.iter().enumerate() {
                let opt_value = unsafe { ca.get_unchecked(row_idx) };
                if let Some(value) = opt_value {
                    if max_value.map_or(true, |current_max| value > current_max) {
                        max_value = Some(value);
                        max_idx = Some(col_idx as u32);
                    }
                }
            }
            unsafe { result.push_unchecked(max_idx) };
        }
        result
    }};
}

fn _arg_max_horizontal_idx(inputs: &[Series]) -> PolarsResult<Series> {
    let len: usize = inputs[0].len();

    _check_types(inputs)?;

    let dtype: &DataType = inputs[0].dtype();

    let result: Vec<Option<u32>> = match dtype {
        DataType::Float64 => impl_argmax_for_type!(inputs, len, f64, f64),
        DataType::Float32 => impl_argmax_for_type!(inputs, len, f32, f32),
        DataType::Int64 => impl_argmax_for_type!(inputs, len, i64, i64),
        DataType::Int32 => impl_argmax_for_type!(inputs, len, i32, i32),
        DataType::UInt64 => impl_argmax_for_type!(inputs, len, u64, u64),
        DataType::UInt32 => impl_argmax_for_type!(inputs, len, u32, u32),
        _ => {
            return Err(PolarsError::ComputeError(
                format!("Unsupported dtype: {:?}", dtype).into(),
            ))
        }
    };

    Ok(
        UInt32Chunked::from_iter_options(PlSmallStr::from_static(""), result.into_iter())
            .into_series(),
    )
}

#[polars_expr(output_type=String)]
fn arg_max_horizontal_colname(inputs: &[Series]) -> PolarsResult<Series> {
    let idx_ser: Series = _arg_max_horizontal_idx(inputs)?;

    // Extract column names from the inputs
    let colnames: Vec<&str> = inputs.iter().map(|s| s.name().as_str()).collect();

    let result: ChunkedArray<StringType> = idx_ser
        .u32()?
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| colnames.get(idx as usize).copied().unwrap_or("")))
        .collect();

    Ok(result.into_series())
}

#[polars_expr(output_type=UInt32)]
fn arg_max_horizontal(inputs: &[Series]) -> PolarsResult<Series> {
    _arg_max_horizontal_idx(inputs)
}
