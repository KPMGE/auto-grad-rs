const EPOCHS: usize = 10;
const LR: f64 = 0.2;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointStyle};
use plotlib::view::ContinuousView;

use crate::{add, prod, tensor};
use crate::{sin, tensor::TensorRef};

fn sin_objective_fn(input: &[TensorRef]) -> TensorRef {
    let x = &input[0];
    sin!(add!(prod!(2.0, x), 0.5))
}

// TODO: Move this to a helper module
pub fn float_range(start: f64, end: f64, step: f64) -> Vec<f64> {
    let start_i = (start / step).ceil() as usize;
    let end_i = (end / step).floor() as usize;

    (start_i..=end_i).map(|i| (i as f64) * step).collect()
}

pub fn sin_regression() {
    let x = tensor!(3.5);
    let (input_values, loss_values) = gradient_descent(sin_objective_fn, EPOCHS, LR, &[x]);

    let original_x: Vec<f64> = float_range(1.5, 5.0, 0.1);
    let original_y: Vec<f64> = original_x
        .iter()
        .map(|xv| {
            let loss = sin_objective_fn(&[tensor!(*xv)]);
            let arr = loss.borrow();
            let values = arr.arr.rows().into_iter().flatten().collect::<Vec<&f64>>();
            *values[0]
        })
        .collect();

    plot_fn(&original_x, &original_y, &input_values, &loss_values);
}

type ObjectiveFn = fn(&[TensorRef]) -> TensorRef;

fn gradient_descent(
    objective_fn: ObjectiveFn,
    n_epochs: usize,
    lr: f64,
    inputs: &[TensorRef],
) -> (Vec<f64>, Vec<f64>) {
    let mut loss_vals = Vec::new();
    let mut input_vals = Vec::new();

    for _ in 0..n_epochs {
        for input in inputs {
            input.borrow_mut().zero_grad();
        }

        let loss = objective_fn(inputs);
        loss.clone().borrow_mut().backward(None);

        let arr = loss.borrow();

        let values = arr.arr.rows().into_iter().flatten().collect::<Vec<&f64>>();
        loss_vals.push(*values[0]);

        let arr2 = &inputs[0];
        let borrow = arr2.borrow();
        let values2 = borrow
            .arr
            .rows()
            .into_iter()
            .flatten()
            .collect::<Vec<&f64>>();
        input_vals.push(*values2[0]);

        for input in inputs {
            let borrow = input.borrow();
            let input_grad_arr = borrow.grad.as_ref().unwrap().borrow();
            let new_arr_value = -lr * &input_grad_arr.arr + &borrow.arr;

            let mut input_borrow = input.borrow_mut();
            input_borrow.set_arr(new_arr_value);
        }
    }

    (input_vals, loss_vals)
}

fn plot_fn(original_x: &[f64], original_y: &[f64], out_x: &[f64], out_y: &[f64]) {
    let data: Vec<(f64, f64)> = original_x
        .into_iter()
        .zip(original_y)
        .map(|(xi, yi)| (*xi, *yi))
        .collect();
    let train_data: Vec<(f64, f64)> = out_x
        .into_iter()
        .zip(out_y)
        .map(|(xi, yi)| (*xi, *yi))
        .collect();

    let line = Plot::new(data)
        .line_style(LineStyle::new().colour("blue"))
        .point_style(PointStyle::new().colour("blue").size(1.0));

    let dots = Plot::new(train_data)
        .line_style(LineStyle::new().width(0.0))
        .point_style(PointStyle::new().size(2.5).colour("red"));

    let view = ContinuousView::new()
        .add(line)
        .add(dots)
        .x_range(-3.0, 3.0)
        .y_range(-1.0, 1.0)
        .x_label("x")
        .y_label("y");

    Page::single(&view).save("plot.svg").unwrap();
}
