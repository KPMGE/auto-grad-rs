use crate::examples::{
    perform_image_recognition, perform_sin_regression, perform_sin_regression_mlp,
};

mod examples;
mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    perform_sin_regression();
    perform_sin_regression_mlp();
    perform_image_recognition();
}
