use polars::chunked_array::builder::ListStringChunkedBuilder;
use polars::datatypes::PlSmallStr;
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

fn _stop_on_first_null_fp(inputs: &[Series]) -> PolarsResult<Series> {
    // Stop on first null fast path
    let length = inputs[0].len();

    // Allocate a result builder for List<String>
    let mut result_builder: ListStringChunkedBuilder =
        ListStringChunkedBuilder::new(PlSmallStr::from_static(""), length, length * inputs.len());

    let str_arrays: Vec<&ChunkedArray<StringType>> =
        inputs.iter().map(|s: &Series| s.str().unwrap()).collect();

    // Row buffer: small fixed array on stack
    let mut row_buf: SmallVec<[&str; 128]> = SmallVec::with_capacity(length);

    for row_idx in 0..length {
        row_buf.clear();

        for arr in &str_arrays {
            match arr.get(row_idx) {
                Some(val) => row_buf.push(val),
                None => break, // stop on first null
            }
        }
        result_builder.append_values_iter(row_buf.iter().copied());
    }
    return Ok(result_builder.finish().into_series());
}

fn _all_nulls_backloaded_fp(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].len();
    let width = inputs.len();
    let mut builder =
        ListStringChunkedBuilder::new(PlSmallStr::from_static("res"), len, len * inputs.len());

    // Borrow the chunked arrays once
    let chunks: Vec<&ChunkedArray<StringType>> = inputs.iter().map(|s| s.str().unwrap()).collect();

    for row_idx in 0..len {
        let mut vals: SmallVec<[&str; 128]> = SmallVec::with_capacity(width);

        for arr in &chunks {
            let v: Option<&str> = unsafe { arr.get_unchecked(row_idx) };
            if let Some(s) = v {
                vals.push(s);
            } else {
                break;
            }
        }
        builder.append_values_iter(vals.into_iter());
    }

    Ok(builder.finish().into_series())
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

    let stop_on_first_null = kwargs.stop_on_first_null;

    // Pre-extract chunked arrays so we don't call .str().unwrap() every row
    // This avoids repeated calls to .str() which can be expensive
    // when there are many rows.
    let str_arrays: Vec<&ChunkedArray<StringType>> =
        inputs.iter().map(|s| s.str().unwrap()).collect();

    // FAST PATH: Check if all columns have nulls only at the end
    if stop_on_first_null {
        return _all_nulls_backloaded_fp(&inputs);
    }
    // TODO: Split these up
    // FAST PATH: Path for when we stop on the first null
    if stop_on_first_null {
        return _stop_on_first_null_fp(&inputs);
    }

    // Row buffer: small fixed array on stack
    let length = inputs[0].len();
    let mut row_buf: SmallVec<[&str; 128]> = SmallVec::with_capacity(inputs.len());

    let mut result_builder: ListStringChunkedBuilder =
        ListStringChunkedBuilder::new(PlSmallStr::from_static(""), length, length * inputs.len());

    // DEFAULT/SLOW PATH: Path for when we do not stop on the first null
    for row_idx in 0..length {
        row_buf.clear();

        for arr in &str_arrays {
            if let Some(val) = arr.get(row_idx) {
                row_buf.push(val);
            }
            // else: skip nulls, do not break
        }
        result_builder.append_values_iter(row_buf.iter().copied());
    }

    Ok(result_builder.finish().into_series())
}
