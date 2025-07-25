use std::iter::zip;

use ndarray::Array2;
use plotlib::{
    page::Page,
    repr::Plot,
    style::{LineStyle, PointStyle},
    view::ContinuousView,
};
use rand::rng;
use rand_distr::{Distribution, Normal};

use crate::{add, examples::float_range, matmul, prod, sub, tanh, tensor::TensorRef};
use crate::{square, tensor};

const EPOCHS: usize = 500;
const LR: f64 = 1e-1;

pub fn perform_sin_regression_mlp() {
    let mut sin_reg_mlp = SinRegressionMlp::new(|x| tanh!(x));
    sin_reg_mlp.train(EPOCHS, LR);
    sin_reg_mlp.plot("sin_regression_mlp.svg");
}

pub struct SinRegressionMlp<F> {
    xs: Vec<f64>,
    ys: Vec<f64>,
    mlp: Mlp<F>,
}

impl<F> SinRegressionMlp<F>
where
    F: Fn(TensorRef) -> TensorRef,
{
    pub fn new(activation_fn: F) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rng();

        let xs = float_range(0.0, 6.0, 0.05);
        let ys: Vec<f64> = xs
            .iter()
            .map(|x| x.sin() + normal.sample(&mut rng) * 0.2)
            .collect();

        let mlp = Mlp::new(activation_fn);

        Self { mlp, xs, ys }
    }

    pub fn train(&mut self, epochs: usize, lr: f64) {
        let params = self.mlp.parameters();
        self.gradient_descent(epochs, lr, &params);
    }

    pub fn plot(&self, file_path: &str) {
        let original_ys = self.xs.iter().map(|x| x.sin());

        let original_data: Vec<(f64, f64)> = self
            .xs
            .iter()
            .zip(original_ys)
            .map(|(xi, yi)| (*xi, yi))
            .collect();

        let train_data: Vec<(f64, f64)> = self
            .xs
            .iter()
            .zip(self.ys.iter())
            .map(|(xi, yi)| (*xi, *yi))
            .collect();

        let model_ys: Vec<f64> = self
            .xs
            .iter()
            .map(|x| {
                let y = self.mlp.forward(tensor!(*x));
                let arr = y.borrow();
                let values = arr.arr.rows().into_iter().flatten().collect::<Vec<&f64>>();
                *values[0]
            })
            .collect();

        let model_data: Vec<(f64, f64)> = self
            .xs
            .iter()
            .zip(model_ys.iter())
            .map(|(xi, yi)| (*xi, *yi))
            .collect();

        let line_original_data = Plot::new(original_data)
            .legend("Original".to_string())
            .line_style(LineStyle::new().colour("blue"))
            .point_style(PointStyle::new().colour("blue").size(1.0));

        let line_original_model_data = Plot::new(model_data)
            .legend("Model".to_string())
            .line_style(LineStyle::new().colour("#008000"))
            .point_style(PointStyle::new().colour("#008000").size(1.0));

        let dots = Plot::new(train_data)
            .legend("Points".to_string())
            .line_style(LineStyle::new().width(0.0))
            .point_style(PointStyle::new().size(2.5).colour("red"));

        let view = ContinuousView::new()
            .add(line_original_data)
            .add(line_original_model_data)
            .add(dots)
            .x_range(0.0, 7.0)
            .y_range(-2.0, 2.0)
            .x_label("x")
            .y_label("y");

        Page::single(&view).save(file_path).unwrap();
    }

    fn loss(&self, _inputs: &[TensorRef]) -> TensorRef {
        let mut total_loss = tensor!(0.0);
        let xs_len = self.xs.len() as f64;

        for (xi, yi) in zip(&self.xs, &self.ys) {
            let predicted = self.mlp.forward(tensor!(*xi));
            let diff = sub!(tensor!(*yi), predicted);
            let loss = prod!(square!(diff), tensor!(1.0 / xs_len));
            total_loss = add!(total_loss, loss);
        }

        total_loss
    }

    fn gradient_descent(&self, n_epochs: usize, lr: f64, inputs: &[TensorRef]) {
        for epoch in 0..n_epochs {
            for input in inputs {
                input.borrow_mut().zero_grad();
            }

            let loss = self.loss(&[]);
            loss.clone().borrow_mut().backward(None);

            let current_loss: Vec<f64> = loss.borrow().arr.iter().map(|x| *x).collect();
            assert!(current_loss.len() == 1, "loss value must be a scalar!");

            let current_loss_value = current_loss[0];

            for input in inputs {
                let mut input_borrow = input.borrow_mut();
                if let Some(grad_tensor_rc) = &input_borrow.grad {
                    let new_arr_value = &input_borrow.arr - (lr * &grad_tensor_rc.borrow().arr);

                    input_borrow.set_arr(new_arr_value);
                } else {
                    println!("Warning: Parameter did not receive a gradient.");
                }
            }

            println!("Epoch {}: LOSS: {:.6}", epoch + 1, current_loss_value);
        }
    }
}

struct Mlp<F> {
    activation_fn: F,
    w0: TensorRef,
    b0: TensorRef,
    w1: TensorRef,
    b1: TensorRef,
    w2: TensorRef,
    b2: TensorRef,
}

impl<F> Mlp<F>
where
    F: Fn(TensorRef) -> TensorRef,
{
    pub fn new(activation_fn: F) -> Self {
        let w0 = tensor!(Self::init_matrix(64, 1));
        let b0 = tensor!(Self::init_matrix(64, 1));
        let w1 = tensor!(Self::init_matrix(64, 64));
        let b1 = tensor!(Self::init_matrix(64, 1));
        let w2 = tensor!(Self::init_matrix(1, 64));
        let b2 = tensor!(Self::init_matrix(1, 1));

        Self {
            activation_fn,
            b0,
            w0,
            b1,
            w1,
            b2,
            w2,
        }
    }

    fn forward(&self, x: TensorRef) -> TensorRef {
        let z0 = add!(matmul!(self.w0, x), self.b0);
        let h0 = (self.activation_fn)(z0);

        let z1 = add!(matmul!(self.w1, h0), self.b1);
        let h1 = (self.activation_fn)(z1);

        let y = add!(matmul!(self.w2, h1), self.b2);
        y
    }

    fn parameters(&self) -> Vec<TensorRef> {
        vec![
            self.w0.clone(),
            self.b0.clone(),
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        ]
    }

    fn init_matrix(rows: usize, cols: usize) -> Array2<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rng();

        let data: Vec<f64> = (0..rows * cols)
            .map(|_| normal.sample(&mut rng) * 0.1)
            .collect();

        Array2::from_shape_vec((rows, cols), data).unwrap()
    }
}
