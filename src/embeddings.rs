use dfdx::{
    prelude::{Module, TensorCollection},
    shapes::{Const, Dtype, HasShape, Rank2},
    tensor::{Storage, Tensor, TensorFromVec, ZerosTensor},
    tensor_ops::{Device, TryConcat},
};

use crate::TfmrHidden;

pub struct Embeddings<const N_EMBEDDINGS: usize, const DIM: usize, E: Dtype, D: Device<E>> {
    pub weight: Tensor<Rank2<N_EMBEDDINGS, DIM>, E, D>,
}

pub type Gpt2TokenEmbeddings<E, D> = Embeddings<50257, 768, E, D>;

const MAX_POS: usize = 2048;
pub type Gpt2PositionEmbeddings<E, D> = Embeddings<MAX_POS, 768, E, D>;

pub struct Gpt2EmbeddingLayer<E: Dtype, D: Device<E>> {
    wte: Gpt2TokenEmbeddings<E, D>,
    wpe: Gpt2PositionEmbeddings<E, D>,
}

impl<E: Dtype, D: Device<E> + TensorFromVec<i32>> Module<Tensor<(usize, Const<1>), i32, D>>
    for Gpt2EmbeddingLayer<E, D>
{
    type Output = TfmrHidden<768, E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<1>), i32, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = D::default();
        let size = input.shape().0;
        if size > MAX_POS {
            panic!("mismatched dimension")
        } else {
            let positions: Tensor<(usize, Const<1>), i32, D> =
                dev.tensor_from_vec((0..size).map(|i| i as i32).collect(), (size, Const::<1>));
            Ok(self.wte.try_forward(input)? + self.wpe.try_forward(positions)?)
        }
    }
}

impl<const N_EMBEDDINGS: usize, const DIM: usize, E: Dtype, D: Device<E> + Storage<i32>>
    Module<Tensor<(usize, Const<1>), i32, D>> for Embeddings<N_EMBEDDINGS, DIM, E, D>
{
    type Output = TfmrHidden<DIM, E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<1>), i32, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev: D = Default::default();
        let _size = input.shape().0;
        let return_tensor: TfmrHidden<DIM, E, D> = dev.zeros_like(&(0, Const::<DIM>));
        input
            .as_vec()
            .into_iter()
            .map(|i| {
                let i = i as usize;
                self.weight.clone().slice((i..i + 1, ..))
            })
            .try_fold(return_tensor, |accum, new| accum.try_concat(new))
    }
}

impl<const N_EMBEDDINGS: usize, const DIM: usize, E: Dtype, D: Device<E>> TensorCollection<E, D>
    for Embeddings<N_EMBEDDINGS, DIM, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = Embeddings<N_EMBEDDINGS, DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (Self::module("weight", |s| &s.weight, |s| &mut s.weight),),
            |(weight,)| Embeddings { weight },
        )
    }
}

#[cfg(test)]
pub mod test {

    use dfdx::{
        prelude::{BuildModule, LoadFromSafetensors, Module, SaveToSafetensors},
        shapes::Const,
        tensor::{Cpu, Tensor, TensorFromVec, ZerosTensor},
    };

    use tokenizers::Tokenizer;

    use crate::{
        embeddings::{Gpt2PositionEmbeddings, Gpt2TokenEmbeddings},
        test::{load_tensor, save_tensor},
        TfmrHidden,
    };

    use super::Gpt2EmbeddingLayer;

    pub fn get_embeddings(dev: &Cpu) -> Gpt2EmbeddingLayer<f32, Cpu> {
        let mut wte = Gpt2TokenEmbeddings::<f32, Cpu>::build(dev);
        wte.load_safetensors("test_data/wte.safetensors")
            .expect("Failed to load wte weights");
        let mut wpe = Gpt2PositionEmbeddings::<f32, Cpu>::build(dev);
        wpe.load_safetensors("test_data/wpe.safetensors")
            .expect("Failed to load wpe weights");
        Gpt2EmbeddingLayer { wte, wpe }
    }

    #[test]
    fn test_tokenizer_and_embeddings() {
        let tokenizer = Tokenizer::from_file("test_data/tokenizer.bin/tokenizer.json")
            .expect("Failed to open tokenizer");

        let encoded = tokenizer
            .encode("Hello my name is Bort", true)
            .expect("Failed to tokenize \"Hello my name is Bort\"");
        let ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as i32).collect();

        let dev: Cpu = Default::default();
        let mut embeddings = Gpt2TokenEmbeddings::<f32, Cpu>::build(&dev);
        embeddings
            .save_safetensors("test_data/dfdx_wte.safetensors")
            .expect("Failed to save model");

        embeddings
            .load_safetensors("test_data/wte.safetensors")
            .expect("Failed to load wte weights");

        let size = ids.len();
        let ids: Tensor<(usize, Const<1>), i32, Cpu> = dev.tensor_from_vec(ids, (size, Const::<1>));
        let y = embeddings.try_forward(ids).expect("forward pass failes");

        let mut tensor: TfmrHidden<768, f32, Cpu> = dev.zeros_like(&(size, Const::<768>));
        save_tensor(
            "test_data/dfdx_embeddings.safetensors",
            "test_embeddings",
            &tensor,
        );

        load_tensor(
            "test_data/test_embeddings.safetensors",
            "test_embeddings",
            &mut tensor,
        );

        let z = y - (tensor);
        assert!(z.abs().as_vec().iter().sum::<f32>() <= 1.0e-7);

        let mut wpe = Gpt2PositionEmbeddings::<f32, Cpu>::build(&dev);

        wpe.load_safetensors("test_data/wpe.safetensors")
            .expect("Failed to wpe load weights");

        let positions: Tensor<(usize, Const<1>), i32, Cpu> =
            dev.tensor_from_vec((0..size).map(|i| i as i32).collect(), (size, Const::<1>));

        let pos_embd = wpe
            .try_forward(positions)
            .expect("Couldn't encode positions");

        assert!(pos_embd.clone().abs().as_vec().into_iter().sum::<f32>() > 0.0f32);

        let mut test_pos_tensor: TfmrHidden<768, f32, Cpu> = dev.zeros_like(&(size, Const::<768>));

        load_tensor(
            "test_data/test_position_embeddings.safetensors",
            "test_position_embeddings",
            &mut test_pos_tensor,
        );

        assert!(
            (pos_embd - test_pos_tensor)
                .abs()
                .as_vec()
                .iter()
                .sum::<f32>()
                <= 1.0e-7
        );
    }
}
