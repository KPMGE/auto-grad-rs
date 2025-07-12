use image::{GrayImage, Luma};
use ndarray::Array2;

use crate::examples::MnistMlp;

mod examples;
mod functions;
mod name_manager;
mod operation;
mod tensor;

use mnist::{Mnist, MnistBuilder};

const EPOCHS: usize = 50;
const LR: f64 = 1e-4;
const TRAIN_SIZE: usize = 1000;
const TEST_SIZE: usize = 10;

fn save_img_to_disk(img_data: &[u8], name: &str) {
    let mut img = GrayImage::new(28, 28);

    for (i, pixel) in img_data.into_iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        img.put_pixel(x, y, Luma([*pixel]));
    }

    img.save(name).expect("Failed to save image");
}

fn main() {
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
            .arr()
            .into_iter()
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

    println!(
        "ACCURACY: {:.2}%",
        (correct_guesses as f64 / TEST_SIZE as f64) * 100.0
    );
}
