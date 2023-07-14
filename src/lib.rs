#![feature(generic_const_exprs)]
#![feature(inline_const)]
use dfdx::{
    prelude::{Dropout, LayerNorm1D, Module},
    shapes::{Const, Dtype, HasShape, Rank1, Rank2},
    tensor::Tensor,
    tensor_ops::{Device, PermuteTo, RealizeTo, ReshapeTo},
};
use gpt2::Conv1D;

pub mod embeddings;
pub mod gpt2;
pub mod simple;

pub(crate) type TfmrHidden<const HIDDEN_D: usize, E, D> = Tensor<(usize, Const<HIDDEN_D>), E, D>;

#[cfg(test)]
pub mod test {
    use std::path::Path;

    use dfdx::{
        shapes::{Const, Dtype, HasShape, Shape},
        tensor::{safetensors::SafeDtype, Cpu, Tensor, TensorFromVec},
        tensor_ops::Device,
    };

    use memmap::MmapOptions;
    use safetensors::{serialize_to_file, tensor::TensorView, SafeTensors};
    use tokenizers::Tokenizer;

    pub fn get_tokenized_sample(dev: &Cpu) -> Tensor<(usize, Const<1>), i32, Cpu> {
        let tokenizer = Tokenizer::from_file("test_data/tokenizer.bin/tokenizer.json")
            .expect("Failed to open tokenizer");

        let encoded = tokenizer
            .encode("Hello my name is Bort", true)
            .expect("Failed to tokenize \"Hello my name is Bort\"");

        let ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as i32).collect();
        let size = ids.len();

        dev.tensor_from_vec(ids, (size, Const::<1>))
    }

    pub fn save_tensor<P, S: Shape, E, D>(path: P, name: &str, tensor: &Tensor<S, E, D>)
    where
        P: AsRef<Path>,
        E: Dtype + SafeDtype,
        D: Device<E>,
    {
        let _tensor_data = tensor.as_vec();

        let shape = tensor.shape().concrete().into();
        let data = tensor.as_vec();
        let data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = TensorView::new(safetensors::Dtype::F32, shape, &data).unwrap();
        serialize_to_file([(name, &view)], &None, path.as_ref()).expect("Can't save this")
    }

    pub fn load_tensor<P: AsRef<Path>, S: Shape, E: Dtype + SafeDtype, D: Device<E>>(
        path: P,
        name: &str,
        tensor: &mut Tensor<S, E, D>,
    ) {
        let path_ref = path.as_ref();
        let f = std::fs::File::open(path_ref)
            .unwrap_or_else(|_| panic!("unable to open file {path_ref:#?} should be there"));
        let buffer = unsafe { MmapOptions::new().map(&f).expect("Should be able to MMap") };
        let tensors = SafeTensors::deserialize(&buffer).unwrap_or_else(|err| {
            panic!("Unable be able to read safe_tensor from file, {path_ref:#?}: error: {err}")
        });

        tensor.load_safetensor(&tensors, name).unwrap_or_else(|err| {
            panic!("Unable to find tensor name {name} in safetensors file {path_ref:#?}: error: {err:#?}")
        })
    }
}
