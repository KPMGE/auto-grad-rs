use ndarray::Array2;
use std::{cell::RefCell, rc::Rc, vec};

use crate::{
    gd_tensor,
    operation::{Operation, ToArray2},
};

#[derive(Debug)]
pub struct Tensor {
    arr: Array2<f64>,
    parents: Vec<Rc<RefCell<Tensor>>>,
    requires_grad: bool,
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    grad: Option<Rc<RefCell<Tensor>>>,
}

impl Tensor {
    pub fn new<T: ToArray2>(
        arr: T,
        parents: Vec<Rc<RefCell<Tensor>>>,
        requires_grad: bool,
        name: Option<String>,
        operation: Option<Box<dyn Operation>>,
    ) -> Self {
        Tensor {
            arr: arr.to_array2(),
            parents,
            requires_grad,
            name,
            operation,
            grad: None,
        }
    }

    pub fn arr(&self) -> Array2<f64> {
        self.arr.clone()
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
            let self_grad_arr = self.grad.as_ref().unwrap().borrow().arr();
            let my_grad_arr = my_grad.as_ref().unwrap().as_ref().borrow().arr();
            let acc = self_grad_arr + my_grad_arr;
            let new_grad = gd_tensor!(acc, requires_grad: false);

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

impl Default for Tensor {
    fn default() -> Self {
        Tensor {
            arr: Array2::zeros((2, 2)),
            parents: vec![],
            requires_grad: true,
            name: None,
            operation: None,
            grad: None,
        }
    }
}
