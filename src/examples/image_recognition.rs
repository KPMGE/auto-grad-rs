use std::{cell::RefCell, num, ops::Neg, process::exit, rc::Rc};

use ndarray::{s, Array2};
use rand::rng;
use rand_distr::{Distribution, Normal};

use crate::{add, ln, prod, softmax, sum, tensor::Tensor};
use crate::{matmul, square, sub, tensor};

pub struct MnistMlp {
    mlp: Mlp,
    images: Array2<f64>,
    labels: Array2<f64>,
}

impl MnistMlp {
    pub fn new(activation_fn: ActivationFn, images: Array2<f64>, labels: Array2<f64>) -> Self {
        MnistMlp {
            mlp: Mlp::new(activation_fn),
            images,
            labels,
        }
    }

    pub fn train(&mut self, epochs: usize, lr: f64) {
        let params = self.mlp.parameters();
        self.gradient_descent(epochs, lr, &params);
    }

    pub fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        self.mlp.forward(x)
    }

    fn cross_entropy_loss(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let mut total_loss = tensor!(0.0);
        let num_samples = self.images.shape()[0] as f64;

        for (image, label_one_hot_arr) in self.images.outer_iter().zip(self.labels.outer_iter()) {
            let image_vec: Vec<f64> = image.iter().map(|&x| x).collect();
            let label_one_hot = label_one_hot_arr
                .to_owned()
                .into_shape_clone((10, 1))
                .unwrap();

            let predicted = self.mlp.forward(tensor!(image_vec));
            let predicted_probs = softmax!(predicted);

            let log_probs = ln!(predicted_probs);
            let neg_log_probs = prod!(log_probs, -1.0);

            let loss = sum!(prod!(tensor!(neg_log_probs), tensor!(label_one_hot)));

            total_loss = add!(total_loss, loss)
        }

        prod!(total_loss, 1.0 / num_samples)
    }

    fn loss(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let mut total_loss = tensor!(0.0);
        let num_samples = self.images.shape()[0] as f64;

        for (image, label_one_hot_arr) in self.images.outer_iter().zip(self.labels.outer_iter()) {
            let image_vec: Vec<f64> = image.iter().map(|&x| x).collect();
            let label_one_hot = label_one_hot_arr
                .to_owned()
                .into_shape_clone((10, 1))
                .unwrap();

            let predicted = self.mlp.forward(tensor!(image_vec));
            let diff = sub!(tensor!(label_one_hot), predicted);
            let loss = prod!(square!(diff), 1.0 / num_samples);

            total_loss = add!(total_loss, loss);
        }

        sum!(total_loss)
    }

    fn gradient_descent(&self, n_epochs: usize, lr: f64, inputs: &[Rc<RefCell<Tensor>>]) {
        for epoch in 0..n_epochs {
            for input in inputs {
                input.borrow_mut().zero_grad();
            }

            let loss = self.cross_entropy_loss(&[]);
            loss.clone().borrow_mut().backward(None);

            let current_loss: Vec<f64> = loss.borrow().arr().iter().map(|x| *x).collect();
            assert!(current_loss.len() == 1, "loss value must be a scalar!");

            let current_loss_value = current_loss[0];

            for input in inputs {
                let mut input_borrow = input.borrow_mut();
                if let Some(grad_tensor_rc) = input_borrow.grad() {
                    let new_arr_value = input_borrow.arr() - (lr * grad_tensor_rc.borrow().arr());

                    input_borrow.set_arr(new_arr_value);
                } else {
                    println!("Warning: Parameter did not receive a gradient.");
                }
            }

            println!("Epoch {}: LOSS: {:.6}", epoch + 1, current_loss_value);
        }
    }
}

type ActivationFn = fn(Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>>;
pub struct Mlp {
    activation_fn: ActivationFn,
    w0: Rc<RefCell<Tensor>>,
    b0: Rc<RefCell<Tensor>>,
    w1: Rc<RefCell<Tensor>>,
    b1: Rc<RefCell<Tensor>>,
    w2: Rc<RefCell<Tensor>>,
    b2: Rc<RefCell<Tensor>>,
}

impl Mlp {
    pub fn new(activation_fn: ActivationFn) -> Self {
        // Input Layer: 784 features (28x28 flattened image)
        // Hidden Layer 1: 128 neurons
        let w0 = tensor!(Self::init_matrix(128, 784));
        let b0 = tensor!(Self::init_matrix(128, 1));

        // Hidden Layer 2: 64 neurons
        // The number of columns in w1 must match the number of rows in w0 (outputs of previous layer)
        let w1 = tensor!(Self::init_matrix(64, 128));
        let b1 = tensor!(Self::init_matrix(64, 1));

        // Output Layer: 10 neurons (for 10 classes: 0-9)
        let w2 = tensor!(Self::init_matrix(10, 64));
        let b2 = tensor!(Self::init_matrix(10, 1));

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

    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let z0 = add!(matmul!(self.w0, x), self.b0);
        let h0 = (self.activation_fn)(z0);

        let z1 = add!(matmul!(self.w1, h0), self.b1);
        let h1 = (self.activation_fn)(z1);

        let y = add!(matmul!(self.w2, h1), self.b2);
        y
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
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
