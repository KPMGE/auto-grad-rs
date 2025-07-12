use ndarray::Array2;
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use crate::{
    operation::{Operation, ToArray2},
    tensor,
};

#[derive(Debug, Clone)]
pub struct TensorRef(Rc<RefCell<Tensor>>);

impl TensorRef {
    pub fn new(tensor: Tensor) -> Self {
        TensorRef(Rc::new(RefCell::new(tensor)))
    }

    pub fn borrow(&self) -> Ref<'_, Tensor> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Tensor> {
        self.0.borrow_mut()
    }

    pub fn try_borrow(&self) -> Result<Ref<'_, Tensor>, std::cell::BorrowError> {
        self.0.try_borrow()
    }

    pub fn try_borrow_mut(&self) -> Result<RefMut<'_, Tensor>, std::cell::BorrowMutError> {
        self.0.try_borrow_mut()
    }

    pub fn try_unwrap(self) -> Result<Tensor, Self> {
        match Rc::try_unwrap(self.0) {
            Ok(ref_cell) => Ok(ref_cell.into_inner()),
            Err(rc) => Err(TensorRef(rc)),
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub arr: Array2<f64>,
    pub parents: Vec<TensorRef>,
    pub requires_grad: bool,
    #[allow(dead_code)]
    pub name: Option<String>,
    pub operation: Option<Box<dyn Operation>>,
    pub grad: Option<TensorRef>,
}

impl Tensor {
    pub fn backward(&mut self, my_grad_input: Option<TensorRef>) {
        if !self.requires_grad {
            return;
        }

        let my_grad: TensorRef = if let Some(g) = my_grad_input {
            g
        } else {
            tensor!(Array2::ones(self.arr.raw_dim()))
        };

        if self.grad.is_none() {
            self.grad = Some(my_grad.clone());
        } else {
            let mut existing_grad_tensor = self.grad.as_ref().unwrap().borrow_mut();
            existing_grad_tensor.arr += &my_grad.borrow().arr;
        }

        if let Some(operation) = &self.operation {
            let parent_grads = operation.grad(my_grad, &self.parents);

            for (parent, parent_grad) in self.parents.iter().zip(parent_grads) {
                parent.borrow_mut().backward(Some(parent_grad));
            }
        }
    }

    pub fn zero_grad(&mut self) {
        if let Some(grad_rc) = &self.grad {
            let mut grad_borrow = grad_rc.borrow_mut();
            grad_borrow.arr.fill(0.0);
        } else {
            let zeros_arr: Array2<f64> = Array2::from_elem(self.arr.raw_dim(), 0.0);
            self.grad = Some(tensor!(zeros_arr));
        }
    }

    pub fn set_arr(&mut self, val: impl ToArray2) {
        self.arr = val.to_array2();
    }
}

pub struct TensorBuilder {
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    arr: Array2<f64>,
    parents: Vec<TensorRef>,
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

    pub fn parents(mut self, parents: Vec<TensorRef>) -> Self {
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
        writeln!(f, "{}", self.arr)
    }
}
