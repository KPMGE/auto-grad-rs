use std::vec;

mod functions;
mod name_manager;
mod operation;
mod tensor;

fn main() {
    let w = gd_tensor!(matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let v = gd_tensor!(vec![1.0, 2.0, 3.0]);
    let z = matmul!(w, v);

    println!("{:#?}", z.borrow().arr());

    z.borrow_mut().backward(None);

    println!("{:#?}", w.borrow().grad());
    println!("{:#?}", v.borrow().grad());
}
