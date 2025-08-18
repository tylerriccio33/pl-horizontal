use polars::datatypes::PlSmallStr;
use polars::prelude::*;
use polars_arrow::legacy::trusted_len::TrustedLenPush;
use pyo3_polars::derive::polars_expr;

// -- Arg True

fn arg_true_output_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        PlSmallStr::from_static(""),
        DataType::List(Box::new(DataType::Int32)),
    );
    Ok(field)
}

#[polars_expr(output_type_func=arg_true_output_type)]
fn arg_true_horizontal(inputs: &[Series]) -> PolarsResult<Series> {
    let len: usize = inputs[0].len();
    let width: usize = inputs.len();

    // Preallocate a vector of Lists (one per row)
    let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
        PlSmallStr::from_static(""),
        len,
        width,
        DataType::Int32,
    );

    for row_idx in 0..len {
        let mut indices = Vec::with_capacity(width);

        for (col_idx, s) in inputs.iter().enumerate() {
            let bool_s: Series = s.cast(&DataType::Boolean)?;

            // We're assuming boolean-like inputs (could be bool or numeric)
            // Null values are treated as false
            match bool_s.bool().ok().and_then(|ca| ca.get(row_idx)) {
                Some(true) => indices.push(col_idx as i32),
                _ => {}
            }
        }

        builder.append_slice(&indices);
    }

    Ok(builder.finish().into_series())
}

// -- Arg First True

#[polars_expr(output_type=UInt32)]
fn arg_first_true_horizontal(inputs: &[Series]) -> PolarsResult<Series> {
    let vec_size: usize = inputs[0].len();

    let mut result: Vec<Option<u32>> = Vec::with_capacity(vec_size);

    let bools: Vec<_> = inputs.iter().map(|s| s.bool().unwrap()).collect();

    for row_idx in 0..vec_size {
        let mut found = None;

        for (col_idx, col) in bools.iter().enumerate() {
            if unsafe { col.get_unchecked(row_idx).unwrap_unchecked() } {
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
