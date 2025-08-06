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
    
    let mut result_builder: ListStringChunkedBuilder = ListStringChunkedBuilder::new(
        PlSmallStr::from_static(""),
        length,
        length * inputs.len() // rough estimate of total string capacity
    );
    
    // Iterate through each row
    let mut row_buf: Vec<&str> = Vec::with_capacity(inputs.len());

    for row_idx in 0..length {
        row_buf.clear();
        for series in inputs {
            let str_chunked: &ChunkedArray<StringType> = series.str().unwrap();
            if let Some(value) = str_chunked.get(row_idx) {
                row_buf.push(value);
            }
        }
        result_builder.append_values_iter(row_buf.iter().copied());
    }
    
    Ok(result_builder.finish().into_series())
}