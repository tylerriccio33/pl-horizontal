use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn collapse_columns_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        PlSmallStr::from_static(""), // Empty name, will be set by the calling context
        DataType::List(Box::new(DataType::String))
    );
    Ok(field)
}

#[polars_expr(output_type_func=collapse_columns_output_type)]
fn collapse_columns(inputs: &[Series]) -> PolarsResult<Series> {
    // Ensure we have at least one input
    if inputs.is_empty() {
        polars_bail!(ComputeError: "collapse_columns requires at least one input column");
    }
    
    // Verify all inputs are string columns
    for (i, series) in inputs.iter().enumerate() {
        if !matches!(series.dtype(), DataType::String) {
            polars_bail!(ComputeError: "Input {} is not a string column, got: {}", i, series.dtype());
        }
    }
    
    let length = inputs[0].len();
    
    // Verify all inputs have the same length
    for series in inputs.iter() {
        if series.len() != length {
            polars_bail!(ComputeError: "All input columns must have the same length");
        }
    }
    
    let mut result_builder = ListStringChunkedBuilder::new(
        PlSmallStr::from_static(""),
        length,
        length * inputs.len() // rough estimate of total string capacity
    );
    
    // Iterate through each row
    for row_idx in 0..length {
        let mut row_values = Vec::new();
        
        // Collect non-null values from all columns for this row
        for series in inputs.iter() {
            let str_chunked = series.str().unwrap();
            if let Some(value) = str_chunked.get(row_idx) {
                row_values.push(value);
            }
        }
        
        // Add the list of values for this row
        result_builder.append_values_iter(row_values.into_iter());
    }
    
    Ok(result_builder.finish().into_series())
}