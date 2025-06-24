use std::{cell::RefCell, fmt::Debug, rc::Rc, vec};

use ndarray::Array2;

fn main() {
    let a_arr = vec![1.3, 2.3];
    let arr_b = vec![2.3, 3.3];

    let a = Rc::new(RefCell::new(
        TensorBuilder::new(a_arr.clone()).name("a").build(),
    ));
    let b = Rc::new(RefCell::new(
        TensorBuilder::new(arr_b.clone()).name("b").build(),
    ));

    let add = Add {};
    let inputs = vec![a.clone(), b.clone()];
    let mut c = add.apply(&inputs);

    c.backward(None);

    println!("{:?}", a.borrow().grad());
}

trait Operation: Debug {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Tensor;
    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>>;
}

#[derive(Debug)]
struct Tensor {
    arr: Array2<f64>,
    parents: Vec<Rc<RefCell<Tensor>>>,
    requires_grad: bool,
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    grad: Option<Rc<RefCell<Tensor>>>,
}

impl Tensor {
    fn new<T: ToArray2>(
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

    fn arr(&self) -> Array2<f64> {
        self.arr.clone()
    }

    fn backward(&mut self, mut my_grad: Option<Rc<RefCell<Tensor>>>) {
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
        }

        let grad_val = self.grad.as_ref().unwrap();
        let acc = grad_val.borrow().arr() + self.grad.as_ref().unwrap().borrow().arr();
        let tensor = Rc::new(RefCell::new(
            TensorBuilder::new(acc).requires_grad(false).build(),
        ));
        self.grad = Some(tensor);

        if let Some(operation) = &self.operation {
            let parent_grads = operation.grad(my_grad.unwrap(), &self.parents);

            for (parent, parent_grad) in self.parents.iter().zip(parent_grads) {
                parent.borrow_mut().backward(Some(parent_grad));
            }
        }
    }

    fn grad(&self) -> &Option<Rc<RefCell<Tensor>>> {
        &self.grad
    }
}

struct TensorBuilder {
    name: Option<String>,
    operation: Option<Box<dyn Operation>>,
    arr: Array2<f64>,
    parents: Vec<Rc<RefCell<Tensor>>>,
    requires_grad: bool,
}

impl TensorBuilder {
    fn new<T: ToArray2>(arr: T) -> Self {
        Self {
            arr: arr.to_array2(),
            parents: Vec::new(),
            requires_grad: true,
            name: None,
            operation: None,
        }
    }

    fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn parents(mut self, parents: Vec<Rc<RefCell<Tensor>>>) -> Self {
        self.parents = parents;
        self
    }

    fn operation(mut self, operation: Box<dyn Operation>) -> Self {
        self.operation = Some(operation);
        self
    }

    fn arr<T: ToArray2>(mut self, arr: T) -> Self {
        self.arr = arr.to_array2();
        self
    }

    fn requires_grad(mut self, value: bool) -> Self {
        self.requires_grad = value;
        self
    }

    fn build(self) -> Tensor {
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

#[derive(Debug)]
struct Add {}

trait ToArray2 {
    fn to_array2(self) -> Array2<f64>;
}

impl Operation for Add {
    fn apply(&self, inputs: &[Rc<RefCell<Tensor>>]) -> Tensor {
        let a = &inputs[0].borrow().arr;
        let b = &inputs[1].borrow().arr;
        let sum = a + b;

        TensorBuilder::new(sum.clone())
            .name("add")
            .parents(vec![inputs[0].clone(), inputs[1].clone()])
            .operation(Box::new(Add {}))
            .build()
    }

    fn grad(
        &self,
        back_grad: Rc<RefCell<Tensor>>,
        args: &[Rc<RefCell<Tensor>>],
    ) -> Vec<Rc<RefCell<Tensor>>> {
        vec![back_grad.clone(), back_grad.clone()]
    }
}

impl ToArray2 for f64 {
    fn to_array2(self) -> Array2<f64> {
        // SAFETY: It's safe to assume this will always be correct, since we have only one element
        Array2::from_shape_vec((1, 1), [self].to_vec()).unwrap()
    }
}

impl ToArray2 for Vec<f64> {
    fn to_array2(self) -> Array2<f64> {
        Array2::from_shape_vec((self.len(), 1), self).expect("Invalid shape!")
    }
}

impl ToArray2 for Array2<f64> {
    fn to_array2(self) -> Array2<f64> {
        self
    }
}
