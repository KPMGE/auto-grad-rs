# auto-grad-rs

This is a simple library that lets you compute partial derivatives. It is specially handfull when implementing neural networks specially as it simplifies the back propagation process.
You can find examples on how to use the library in the `examples` folder.

## How to build it?

First of all, make sure you have [Rust](https://www.rust-lang.org/tools/install) propertly installed on you machine, then in order to build this project, you can run:

```bash
cargo build --release
```

## How to run it?

Once the build is complete, you will have an executable under the `target/release/` folder, you can run it directly

```bash
./target/release/auto-grad-rs
```
