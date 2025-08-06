use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use smallvec::SmallVec;

fn collapse_columns_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        PlSmallStr::from_static(""),
        DataType::List(Box::new(DataType::String)),
    );
    Ok(field)
}

#[derive(Deserialize)]
struct CollapseColumnsArgs {
    stop_on_first_null: bool,
}


#[polars_expr(output_type_func=collapse_columns_output_type)]
fn collapse_columns(inputs: &[Series], kwargs: CollapseColumnsArgs) -> PolarsResult<Series> {
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
    let stop_on_first_null = kwargs.stop_on_first_null;

    // Pre-extract chunked arrays so we don't call .str().unwrap() every row
    let str_arrays: Vec<&ChunkedArray<StringType>> = 
        inputs.iter().map(|s| s.str().unwrap()).collect();

    // Row buffer: small fixed array on stack
    let mut row_buf: SmallVec<[&str; 16]> = SmallVec::with_capacity(inputs.len());

    let mut result_builder = ListStringChunkedBuilder::new(
        PlSmallStr::from_static(""),
        length,
        length * inputs.len(),
    );

    for row_idx in 0..length {
        row_buf.clear();

        for arr in &str_arrays {
            match arr.get(row_idx) {
                Some(val) => row_buf.push(val),
                None if stop_on_first_null => break,
                None => {}
            }
        }
        result_builder.append_values_iter(row_buf.iter().copied());
    }

    Ok(result_builder.finish().into_series())
}
