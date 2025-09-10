use polars::datatypes::PlSmallStr;
use polars::prelude::*;
use polars_arrow::legacy::trusted_len::TrustedLenPush;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=UInt32)]
fn arg_first_null_horizontal(inputs: &[Series]) -> PolarsResult<Series> {
    // Get the idx of the first null value in each row
    let vec_size: usize = inputs[0].len();

    let mut result: Vec<Option<u32>> = Vec::with_capacity(vec_size);

    for row_idx in 0..vec_size {
        let mut found: Option<u32> = None;

        for (col_idx, s) in inputs.iter().enumerate() {
            if s.get(row_idx).unwrap().is_null() {
                found = Some(col_idx as u32);
                break;
            }
        }
        unsafe { result.push_unchecked(found) };
    }

    Ok(
        UInt32Chunked::from_iter_options(PlSmallStr::from_str(""), result.into_iter())
            .into_series(),
    )
}
