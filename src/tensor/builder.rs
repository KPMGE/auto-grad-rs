use ndarray::Array2;
use std::{cell::RefCell, rc::Rc};

use crate::{
    operation::{Operation, ToArray2},
    tensor,
};

#[derive(Debug)]
pub struct Tensor {
    arr: Array2<f64>,
    parents: Vec<Rc<RefCell<Tensor>>>,
    requires_grad: bool,
    #[allow(dead_code)]
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    grad: Option<Rc<RefCell<Tensor>>>,
}

impl Tensor {
    pub fn arr(&self) -> &Array2<f64> {
        &self.arr
    }

    pub fn backward(&mut self, mut my_grad: Option<Rc<RefCell<Tensor>>>) {
        if !self.requires_grad {
            return;
        }

        if my_grad.is_none() {
            let ones_arr: Array2<f64> = Array2::from_elem(self.arr.raw_dim(), 1.0);
            let ones_tensor = Rc::new(RefCell::new(TensorBuilder::new(ones_arr).build()));
            my_grad = Some(ones_tensor.clone());
        }

        if self.grad.is_none() {
            self.grad = Some(my_grad.clone().unwrap());
        } else {
            let acc = self.grad.as_ref().unwrap().borrow().arr()
                + my_grad.as_ref().unwrap().as_ref().borrow().arr();
            let new_grad = tensor!(acc, requires_grad: false);

            self.grad = Some(new_grad);
        }

        if let Some(operation) = &self.operation {
            let parent_grads = operation.grad(my_grad.unwrap(), &self.parents);

            for (parent, parent_grad) in self.parents.iter().zip(parent_grads) {
                parent.borrow_mut().backward(Some(parent_grad));
            }
        }
    }

    pub fn grad(&self) -> &Option<Rc<RefCell<Tensor>>> {
        &self.grad
    }

    pub fn zero_grad(&mut self) {
        let arr_dim = self.arr.raw_dim();
        let ones_arr: Array2<f64> = Array2::from_elem(arr_dim, 0.0);
        self.grad = Some(tensor!(ones_arr));
    }

    #[allow(dead_code)]
    pub fn set_grad(&mut self, grad: Rc<RefCell<Tensor>>) {
        self.grad = Some(grad);
    }

    pub fn set_arr(&mut self, val: impl ToArray2) {
        self.arr = val.to_array2();
    }
}

pub struct TensorBuilder {
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    arr: Array2<f64>,
    parents: Vec<Rc<RefCell<Tensor>>>,
    requires_grad: bool,
}

impl TensorBuilder {
    pub fn new<T: ToArray2>(arr: T) -> Self {
        Self {
            arr: arr.to_array2(),
            parents: Vec::new(),
            requires_grad: true,
            name: None,
            operation: None,
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn parents(mut self, parents: Vec<Rc<RefCell<Tensor>>>) -> Self {
        self.parents = parents;
        self
    }

    pub fn operation(mut self, operation: Box<dyn Operation>) -> Self {
        self.operation = Some(operation);
        self
    }

    #[allow(dead_code)]
    pub fn arr<T: ToArray2>(mut self, arr: T) -> Self {
        self.arr = arr.to_array2();
        self
    }

    pub fn requires_grad(mut self, value: bool) -> Self {
        self.requires_grad = value;
        self
    }

    pub fn build(self) -> Tensor {
        Tensor {
            arr: self.arr,
            parents: self.parents,
            requires_grad: self.requires_grad,
            name: self.name,
            operation: self.operation,
            grad: None,
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.arr())
    }
}
