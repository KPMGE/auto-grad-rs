use crate::examples::{Mlp, SinRegressionMlp};

mod examples;
mod functions;
mod name_manager;
mod operation;
mod tensor;

const EPOCHS: usize = 500;
const LR: f64 = 1e-1;

fn main() {
    let mlp = Mlp::new(|x| tanh!(x));
    let mut sin_reg_mlp = SinRegressionMlp::new(mlp);

    sin_reg_mlp.train(EPOCHS, LR);
    sin_reg_mlp.plot("plot.svg");
}
