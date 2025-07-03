use std::{cell::RefCell, rc::Rc};

mod examples;
mod functions;
mod name_manager;
mod operation;
mod tensor;

use ndarray::{Array2, Axis};
use plotlib::{
    page::Page,
    repr::Plot,
    style::{LineStyle, PointStyle},
    view::ContinuousView,
};
use rand::rng;
use rand_distr::{Distribution, Normal};
use tensor::Tensor;

use crate::examples::sin_regression;

fn main() {
    sin_regression();
}

type ObjectiveFn = fn(&[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>>;

fn gradient_descent(
    objective_fn: ObjectiveFn,
    n_epochs: usize,
    lr: f64,
    inputs: &[Rc<RefCell<Tensor>>],
) -> (Vec<f64>, Vec<f64>) {
    let mut loss_vals = Vec::new();
    let mut input_vals = Vec::new();

    for _ in 0..n_epochs {
        for input in inputs {
            input.borrow_mut().zero_grad();
        }

        let loss = objective_fn(inputs);
        loss.clone().borrow_mut().backward(None);

        let arr = loss.borrow().arr();
        let values = arr.rows().into_iter().flatten().collect::<Vec<&f64>>();
        loss_vals.push(*values[0]);

        let arr2 = inputs[0].borrow().arr();
        let values2 = arr2.rows().into_iter().flatten().collect::<Vec<&f64>>();
        input_vals.push(*values2[0]);

        for input in inputs {
            let mut input_borrow = input.borrow_mut();
            let input_grad_arr = input_borrow.grad().as_ref().unwrap().borrow().arr();
            let new_arr_value = -lr * input_grad_arr + input_borrow.arr();

            input_borrow.set_arr(new_arr_value);
        }
    }

    (input_vals, loss_vals)
}

fn plot_fn(original_x: Vec<f64>, original_y: Vec<f64>, out_x: Vec<f64>, out_y: Vec<f64>) {
    let data: Vec<(f64, f64)> = original_x.into_iter().zip(original_y).collect();
    let data2: Vec<(f64, f64)> = out_x.into_iter().zip(out_y).collect();

    let line = Plot::new(data)
        .line_style(LineStyle::new().colour("blue"))
        .point_style(PointStyle::new().colour("blue").size(1.0));

    let dots = Plot::new(data2)
        .line_style(LineStyle::new().width(0.0))
        .point_style(PointStyle::new().size(2.5).colour("red"));

    let view = ContinuousView::new()
        .add(line)
        .add(dots)
        .x_range(1.5, 5.0)
        .y_range(-1.0, 1.0)
        .x_label("x")
        .y_label("y");

    Page::single(&view).save("plot.svg").unwrap();
}
