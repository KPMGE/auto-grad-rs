use image::{GrayImage, Luma};
use ndarray::{s, Array2};

use crate::examples::MnistMlp;

mod examples;
mod functions;
mod name_manager;
mod operation;
mod tensor;

use mnist::{Mnist, MnistBuilder};

const EPOCHS: usize = 20;
const LR: f64 = 1e-3;

fn main() {
    let train_size: usize = 1000;
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(train_size as u32) // Use actual dataset size
        .test_set_length(10_000) // Use actual dataset size
        .finalize();

    // --- Reshape training images for MLP input ---
    // The MnistBuilder returns trn_img as a flattened Vec<u8> of size 60_000 * 28 * 28.
    // We want to reshape it to (60_000, 784) for MLP input.
    let train_images_flat: Array2<f64> = Array2::from_shape_vec(
        (train_size, 28 * 28), // Reshape to (number_of_images, 784)
        trn_img
            .clone()
            .into_iter()
            .map(|x| x as f64 / 255.0)
            .collect(), // Normalize pixels to 0.0-1.0
    )
    .expect("Error converting training images to flat Array2 struct");

    println!(
        "Training images flat shape: {:?}",
        train_images_flat.shape()
    );

    let train_labels_one_hot: Array2<f64> = Array2::from_shape_fn(
        (train_size, 10), // Now (number_of_labels, number_of_classes)
        |(i, j)| if trn_lbl[i] as usize == j { 1.0 } else { 0.0 },
    );

    let mut mnist_mlp = MnistMlp::new(
        |x| tanh!(x),
        train_images_flat.clone(),
        train_labels_one_hot.clone(),
    );

    mnist_mlp.train(EPOCHS, LR);

    // Each image is 28 * 28 = 784 pixels
    let image_data = tst_img[0..784].to_vec();

    // Create a new grayscale image (28x28)
    let mut img = GrayImage::new(28, 28);

    for (i, pixel) in image_data.into_iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        img.put_pixel(x, y, Luma([pixel]));
    }

    // Save to disk
    img.save("mnist_sample.png").expect("Failed to save image");

    // println!("LABEL: {:#?}", train_labels_one_hot.slice(s![12, ..]));

    let image: Vec<f64> = tst_img[0..784].iter().map(|x| *x as f64 / 255.0).collect();

    let pred = mnist_mlp.forward(tensor!(image));
    let pred_soft = softmax!(pred);

    println!(
        "PREDICTION LOGITS: {:#?}",
        pred.borrow().arr().into_iter().collect::<Vec<&f64>>()
    );

    println!("MODEL PREDICTION: ");
    for (idx, pred) in pred_soft
        .borrow()
        .arr()
        .into_iter()
        .collect::<Vec<&f64>>()
        .iter()
        .map(|x| *x * 100.0)
        .collect::<Vec<f64>>()
        .iter()
        .enumerate()
    {
        println!("{}: {:.2}%", idx, pred);
    }
}
