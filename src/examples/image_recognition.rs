use mnist::{Mnist, MnistBuilder};
#[allow(dead_code)]
use ndarray::Array2;
use rand::rng;
use rand_distr::{Distribution, Normal};

use image::{GrayImage, Luma};

use crate::{add, ln, prod, softmax, sum, tensor::TensorRef};
use crate::{matmul, relu, square, sub, tensor};

const EPOCHS: usize = 50;
const LR: f64 = 1e-1;
const TRAIN_SIZE: usize = 1000;
const TEST_SIZE: usize = 100;

pub fn perform_image_recognition() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAIN_SIZE as u32)
        .test_set_length(TEST_SIZE as u32)
        .finalize();

    let train_images_flat: Array2<f64> = Array2::from_shape_vec(
        (TRAIN_SIZE, 28 * 28),
        trn_img
            .clone()
            .into_iter()
            .map(|x| x as f64 / 255.0)
            .collect(),
    )
    .expect("Error converting training images to flat Array2 struct");

    let train_labels_one_hot: Array2<f64> = Array2::from_shape_fn((TRAIN_SIZE, 10), |(i, j)| {
        if trn_lbl[i] as usize == j {
            1.0
        } else {
            0.0
        }
    });

    let mut mnist_mlp = MnistMlp::new(|x| relu!(x), train_images_flat, train_labels_one_hot);

    mnist_mlp.train(EPOCHS, LR);

    let mut correct_guesses = 0;

    for i in 1..TEST_SIZE {
        let test_image: Vec<f64> = tst_img[784 * (i - 1)..784 * i]
            .iter()
            .map(|x| *x as f64 / 255.0)
            .collect();
        let pred_logits = mnist_mlp.forward(tensor!(test_image));
        let pred_probs = softmax!(pred_logits);

        let model_predicted_label = pred_probs
            .borrow()
            .arr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let correct_label = tst_lbl[i - 1];

        if model_predicted_label as u8 == correct_label {
            correct_guesses += 1;
        }

        let image_data = tst_img[784 * (i - 1)..784 * i].to_vec();
        save_img_to_disk(&image_data, &format!("test_img_{}.png", i));

        println!("CORRECT LABEL: {}", correct_label);

        println!("MODEL PREDICTION: ");
        for (idx, pred) in pred_probs
            .borrow()
            .arr
            .iter()
            .collect::<Vec<&f64>>()
            .into_iter()
            .map(|x| *x * 100.0)
            .collect::<Vec<f64>>()
            .iter()
            .enumerate()
        {
            println!("{}: {:.2}%", idx, pred);
        }
    }

    println!(
        "ACCURACY: {:.2}%",
        (correct_guesses as f64 / TEST_SIZE as f64) * 100.0
    );
}

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

    pub fn forward(&self, x: TensorRef) -> TensorRef {
        self.mlp.forward(x)
    }

    fn cross_entropy_loss(&self, _inputs: &[TensorRef]) -> TensorRef {
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

    fn loss(&self, _inputs: &[TensorRef]) -> TensorRef {
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

    fn gradient_descent(&self, n_epochs: usize, lr: f64, inputs: &[TensorRef]) {
        for epoch in 0..n_epochs {
            for input in inputs {
                input.borrow_mut().zero_grad();
            }

            let loss = self.cross_entropy_loss(&[]);
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

type ActivationFn = fn(TensorRef) -> TensorRef;
pub struct Mlp {
    activation_fn: ActivationFn,
    w0: TensorRef,
    b0: TensorRef,
    w1: TensorRef,
    b1: TensorRef,
    w2: TensorRef,
    b2: TensorRef,
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

fn save_img_to_disk(img_data: &[u8], name: &str) {
    let mut img = GrayImage::new(28, 28);

    for (i, pixel) in img_data.into_iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        img.put_pixel(x, y, Luma([*pixel]));
    }

    img.save(name).expect("Failed to save image");
}
