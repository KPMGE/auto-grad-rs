use std::{cell::RefCell, rc::Rc};

const EPOCHS: usize = 10;
const LR: f64 = 0.2;

use crate::{add, prod, tensor};
use crate::{gradient_descent, plot_fn, sin, tensor::Tensor};

fn sin_objective_fn(input: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
    let x = &input[0];
    sin!(add!(prod!(2.0, x), 0.5))
}

fn float_range(start: f64, end: f64, step: f64) -> Vec<f64> {
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
            let arr = loss.borrow().arr();
            let values = arr.rows().into_iter().flatten().collect::<Vec<&f64>>();
            *values[0]
        })
        .collect();

    plot_fn(original_x, original_y, input_values, loss_values);
}
