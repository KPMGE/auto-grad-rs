use std::{cell::RefCell, rc::Rc};

use ndarray::Array2;

use crate::tensor;
use crate::{
    name_manager::{NameManager, NAME_MANAGER},
    operation::Operation,
    tensor::{Tensor, TensorBuilder},
};

#[macro_export]
macro_rules! softmax {
    ($val1:expr) => {{
        use crate::functions::Softmax;
        use crate::operation::Operation;
        use crate::tensor;

        let t = tensor!($val1.clone());

        let softmax = Softmax::new();
        softmax.apply(&[t])
    }};
}

#[derive(Debug)]
pub struct Softmax {
    name_manager: Rc<RefCell<NameManager>>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax {
            name_manager: NAME_MANAGER.with(|mn| mn.clone()),
        }
    }

    fn apply(inputs: &Array2<f64>) -> Array2<f64> {
        let exps = inputs.mapv(|x| x.exp());
        let sum_exps: f64 = exps.iter().sum();
        exps / sum_exps
    }

    fn jacobian(xs: &Vec<f64>, ys: &Vec<f64>) -> Array2<f64> {
        let mut jacobian = Array2::zeros((ys.len(), ys.len()));

        for (k, _) in xs.iter().enumerate() {
            for (i, _) in ys.iter().enumerate() {
                if i == k {
                    jacobian[[i, k]] = ys[i] * (1.0 - ys[i]);
                } else {
                    jacobian[[i, k]] = -(ys[k] * ys[i]);
                }
            }
        }

        jacobian
    }
}

impl Operation for Softmax {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Rc<RefCell<Tensor>> {
        let a = &inputs[0];

        let softmax = Softmax::apply(a.borrow().arr());
        let op_name = self.name_manager.clone().borrow_mut().new_name("softmax");

        let tensor = TensorBuilder::new(softmax)
            .name(&op_name)
            .parents(vec![inputs[0].clone()])
            .operation(Box::new(Softmax::new()))
            .build();

        Rc::new(RefCell::new(tensor))
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        let x = &args[0];
        let y = Softmax::apply(x.borrow().arr());

        let xs_flat: Vec<f64> = x.borrow().arr().iter().map(|x| *x).collect();
        let ys_flat: Vec<f64> = y.iter().map(|y| *y).collect();
        let j = Self::jacobian(&xs_flat, &ys_flat);

        let grad = j.t().dot(back_grad.borrow().arr());

        vec![tensor!(grad, name: "softmax_grad")]
    }
}
