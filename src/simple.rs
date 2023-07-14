#[cfg(test)]
mod test {
    use dfdx::{
        prelude::{GeLU, Module},
        shapes::Const,
        tensor::{Cpu, TensorFromVec},
    };

    #[test]
    fn test_gelu() {
        let dev = Cpu::default();
        let tensor = dev.tensor_from_vec(vec![-5., -1., 0., 1., 5.], (Const::<1>, Const::<5>));
        let gelu = GeLU::default();
        let forward = gelu.forward(tensor);
        println!("{:#?}", forward.as_vec())
    }
}
