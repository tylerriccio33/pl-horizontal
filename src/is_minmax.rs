use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Boolean)]
fn is_max(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    
    let max_i: Option<usize> = s.arg_max();
    
    // Create mask where only the max index is true
    match max_i {
        Some(idx) => {
            // Most performant: use polars' range comparison
            let indices: Series = Series::new(PlSmallStr::from_str(""), 0u32..s.len() as u32);
            let target_idx: Series = Series::new(PlSmallStr::from_str(""), [idx as u32]);
            indices.equal(&target_idx).map(|ca| ca.into_series())
        }
        None => {
            // All false for null case
            Ok(BooleanChunked::full(PlSmallStr::from_str(""), false, s.len()).into_series())
        }
    }
}
