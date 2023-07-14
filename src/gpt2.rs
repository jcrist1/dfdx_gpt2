use dfdx::{
    prelude::{modules::LayerNorm1D, Dropout, Module, Repeated, TensorCollection, TensorOptions},
    shapes::{Axis, Const, Dtype, HasShape, Rank1, Rank2, Shape},
    tensor::{Cpu, Tensor, TensorFromVec, TriangleTensor, ZerosTensor},
    tensor_ops::{
        BroadcastTo, ChooseFrom, Device, PermuteTo, RealizeTo, ReshapeTo, TryAdd, TryDiv, TryMatMul,
    },
};

use crate::{embeddings::Embeddings, TfmrHidden};

pub struct Conv1D<const INPUT: usize, const OUTPUT: usize, E: Dtype, D: Device<E>> {
    weight: Tensor<Rank2<INPUT, OUTPUT>, E, D>,
    bias: Tensor<Rank1<OUTPUT>, E, D>,
}

// todo: implement TensorColletion for Conv1D
//
impl<const IN: usize, const OUT: usize, E, D> Module<Tensor<Rank1<IN>, E, D>>
    for Conv1D<IN, OUT, E, D>
where
    E: Dtype,
    D: Device<E>,
{
    type Output = Tensor<Rank1<OUT>, E, D>;

    type Error = D::Err;

    fn try_forward(&self, input: Tensor<Rank1<IN>, E, D>) -> Result<Self::Output, Self::Error> {
        let new = input.try_matmul(self.weight.retaped())?;
        new.try_add(self.bias.retaped())
    }
}

//
impl<const IN: usize, const OUT: usize, E, D> Module<TfmrHidden<IN, E, D>> for Conv1D<IN, OUT, E, D>
where
    E: Dtype,
    D: Device<E>,
{
    type Output = Tensor<(usize, Const<OUT>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<IN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let input_size = input.shape().0;
        let new = input.try_matmul(self.weight.retaped())?;

        new.try_add(
            self.bias
                .retaped()
                .try_broadcast_like(&(input_size, Const::<OUT>))?,
        )
    }
}

impl<const IN: usize, const OUT: usize, E: Dtype, D: Device<E>> TensorCollection<E, D>
    for Conv1D<IN, OUT, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = Conv1D<IN, OUT, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("weight", |s| &s.weight, |s| &mut s.weight),
                Self::module("bias", |s| &s.bias, |s| &mut s.bias),
            ),
            |(weight, bias)| Conv1D { weight, bias },
        )
    }
}
type AttnOut<const N_HEADS: usize, E, D> = Tensor<(Const<N_HEADS>, usize, usize), E, D>;

struct Gpt2SelfAttention<
    const HEAD_DIM: usize,
    const MAX_POSITION: usize,
    const N_HEADS: usize,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
{
    bias: Tensor<Rank2<MAX_POSITION, MAX_POSITION>, bool, Cpu>,
    masked_bias: Tensor<(), E, D>,
    c_attention: Conv1D<{ HEAD_DIM * N_HEADS }, { 3 * HEAD_DIM * N_HEADS }, E, D>,
    c_proj: Conv1D<{ HEAD_DIM * N_HEADS }, { HEAD_DIM * N_HEADS }, E, D>,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    // todo: add bias, and and masked_bias as non-backprop tensors, values
}

impl<
        const HEAD_DIM: usize,
        const MAX_POSITION: usize,
        const N_HEADS: usize,
        E: Dtype,
        D: Device<E> + ZerosTensor<bool>,
    > TensorCollection<E, D> for Gpt2SelfAttention<HEAD_DIM, MAX_POSITION, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Gpt2SelfAttention<HEAD_DIM, MAX_POSITION, N_HEADS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("c_attn", |s| &s.c_attention, |s| &mut s.c_attention),
                Self::module("c_proj", |s| &s.c_proj, |s| &mut s.c_proj),
                Self::tensor(
                    "masked_bias.",
                    |s| &s.masked_bias,
                    |s| &mut s.masked_bias,
                    TensorOptions::detached(|_tens| Ok(())),
                ),
            ),
            |(c_attention, c_proj, masked_bias)| {
                let cpu = Cpu::default();
                let bias = cpu.lower_tri(true, None);
                Gpt2SelfAttention {
                    bias,
                    masked_bias,
                    c_attention,
                    c_proj,
                    attn_dropout: Dropout { p: 0.0 },
                    resid_dropout: Dropout { p: 0.0 },
                }
            },
        )
    }
}

type AttnTnsr<const N_HEADS: usize, const HEAD_DIM: usize, E, D> =
    Tensor<(Const<N_HEADS>, usize, Const<HEAD_DIM>), E, D>;

impl<
        const HEAD_DIM: usize,
        const MAX_POSITION: usize,
        const N_HEADS: usize,
        E: Dtype,
        D: Device<E>,
    > Gpt2SelfAttention<HEAD_DIM, MAX_POSITION, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    Tensor<(Const<N_HEADS>, usize, usize), E, D>: TryDiv<E>,
    Tensor<(Const<N_HEADS>, usize, usize), bool, D>:
        ChooseFrom<AttnOut<N_HEADS, E, D>, AttnOut<N_HEADS, E, D>, Output = AttnOut<N_HEADS, E, D>>,
{
    fn split_heads(
        &self,
        tensor: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> AttnTnsr<N_HEADS, HEAD_DIM, E, D> {
        let seq_size = tensor.shape().0;

        let reshaped = tensor.reshape_like::<(usize, Const<N_HEADS>, Const<HEAD_DIM>)>(&(
            seq_size,
            Const::<N_HEADS>,
            Const::<HEAD_DIM>,
        ));
        // .expect("We should have the same dimensions");
        reshaped.permute()
    }

    fn attention(
        &self,
        dev: &D,
        queries: AttnTnsr<N_HEADS, HEAD_DIM, E, D>,
        keys: AttnTnsr<N_HEADS, HEAD_DIM, E, D>,
        values: AttnTnsr<N_HEADS, HEAD_DIM, E, D>,
    ) -> Result<
        (
            AttnTnsr<N_HEADS, HEAD_DIM, E, D>,
            Tensor<(Const<N_HEADS>, usize, usize), E, D>,
        ),
        D::Err,
    >
    where
        D: TensorFromVec<bool>,
    {
        let attn_weights: Tensor<(Const<N_HEADS>, usize, usize), E, D> =
            queries.clone().try_matmul(
                keys.clone()
                    .try_permute::<(Const<N_HEADS>, Const<HEAD_DIM>, usize), _>()?,
            )?;

        let size = *attn_weights.shape();

        let attn_weights = attn_weights / E::from_f32((HEAD_DIM as f32).sqrt()).unwrap_or(E::ONE);

        let query_length = queries.shape().1;
        let key_length = keys.shape().1;

        let causal_mask: Tensor<(Const<N_HEADS>, usize, usize), bool, Cpu> = self
            .bias
            .clone()
            .slice(((key_length - query_length)..key_length, ..key_length))
            .broadcast_like(&size);
        let causal_mask = causal_mask.to_device(dev);
        let mask_value = E::from_f32(f32::MIN)
            .expect("Failed to get the sqrt of - MIN of f32 as required Dtype");
        let mut mask_value_t: Tensor<(), E, D> = dev.ones();
        mask_value_t = mask_value_t * mask_value;

        let choose = causal_mask.choose(attn_weights, mask_value_t.broadcast_like(&size));
        let attn_weights = self.attn_dropout.try_forward(choose.softmax::<Axis<2>>())?;
        let attn_output = attn_weights
            .clone()
            .try_matmul(values)
            .expect("todo: Convert this error");
        Ok((attn_output, attn_weights))

        //        Transformers does the following, but it's not clear how we can backpropagate this
        //        param, so the above _should_ be good
        //        if self.scale_attn_weights:
        //            attn_weights = attn_weights / torch.full(
        //                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        //            )
    }

    fn merge_heads(
        &self,
        tensor: AttnTnsr<N_HEADS, HEAD_DIM, E, D>,
    ) -> Result<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>, D::Err> {
        let (_, size, _) = *tensor.shape();
        let tensor: Tensor<(usize, Const<N_HEADS>, Const<HEAD_DIM>), E, D> = tensor.permute();
        let tensor = tensor.try_reshape_like(&(size, Const::<{ HEAD_DIM * N_HEADS }>));
        let tensor = tensor?;
        //.expect("Failed to get reshaped tensor for merge_heads");

        Ok(tensor)
    }
    // def _merge_heads(self, tensor, num_heads, attn_head_size):
    //     """
    //     Merges attn_head_size dim and num_attn_heads dim into hidden_size
    //     """
    //     tensor = tensor.permute(0, 2, 1, 3).contiguous()
    //     new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    //     return tensor.view(new_shape)
}

impl<
        const HEAD_DIM: usize,
        const MAX_POSITION: usize,
        const N_HEADS: usize,
        E: Dtype,
        D: Device<E> + TensorFromVec<bool>,
    > Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for Gpt2SelfAttention<HEAD_DIM, MAX_POSITION, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    Tensor<(Const<N_HEADS>, usize, usize), bool, D>:
        ChooseFrom<AttnOut<N_HEADS, E, D>, AttnOut<N_HEADS, E, D>, Output = AttnOut<N_HEADS, E, D>>,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = D::default();
        let forward = self.c_attention.try_forward(input)?;
        let query: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D> = forward
            .clone()
            .slice((.., 0..{ HEAD_DIM * N_HEADS }))
            .realize();
        //.expect("Failed to reshape");
        let key: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D> = forward
            .clone()
            .slice((.., { HEAD_DIM * N_HEADS }..{ 2 * HEAD_DIM * N_HEADS }))
            .realize();
        //.expect("Failed to reshape");
        let value: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D> = forward
            .slice((.., { 2 * HEAD_DIM * N_HEADS }..{ 3 * HEAD_DIM * N_HEADS }))
            .realize();
        //.expect("Failed to reshape");
        let key = self.split_heads(key);
        let query = self.split_heads(query);
        let value = self.split_heads(value);
        let (attn_out, _) = self.attention(&dev, query, key, value)?;

        let attn_out = self.merge_heads(attn_out)?;
        let attn_out = self.c_proj.try_forward(attn_out)?;
        self.resid_dropout.try_forward(attn_out)
    }
}

struct Gpt2Mlp<
    const HEAD_DIM: usize,
    const INTERMEDIATE_SIZE: usize,
    const N_HEADS: usize,
    Activation,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
{
    c_fc: Conv1D<{ HEAD_DIM * N_HEADS }, INTERMEDIATE_SIZE, E, D>,
    c_proj: Conv1D<INTERMEDIATE_SIZE, { HEAD_DIM * N_HEADS }, E, D>,
    activation: Activation,
    dropout: Dropout,
}

impl<
        const HEAD_DIM: usize,
        const INTERMEDIATE_SIZE: usize,
        const N_HEADS: usize,
        Activation,
        E: Dtype,
        D: Device<E>,
    > Module<Tensor<Rank1<{ HEAD_DIM * N_HEADS }>, E, D>>
    for Gpt2Mlp<HEAD_DIM, INTERMEDIATE_SIZE, N_HEADS, Activation, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    Activation: Module<
        Tensor<Rank1<INTERMEDIATE_SIZE>, E, D>,
        Output = Tensor<Rank1<INTERMEDIATE_SIZE>, E, D>,
        Error = D::Err,
    >,
{
    type Output = Tensor<Rank1<{ HEAD_DIM * N_HEADS }>, E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<Rank1<{ HEAD_DIM * N_HEADS }>, E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let x = self.c_fc.try_forward(input)?;
        let x = self.activation.try_forward(x)?;
        let x = self.c_proj.try_forward(x)?;
        self.dropout.try_forward(x)
    }
}

impl<
        const HEAD_DIM: usize,
        const INTERMEDIATE_SIZE: usize,
        const N_HEADS: usize,
        Activation,
        E: Dtype,
        D: Device<E>,
    > Module<TfmrHidden<{ HEAD_DIM * N_HEADS }, E, D>>
    for Gpt2Mlp<HEAD_DIM, INTERMEDIATE_SIZE, N_HEADS, Activation, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    Activation: Module<
        TfmrHidden<INTERMEDIATE_SIZE, E, D>,
        Output = TfmrHidden<INTERMEDIATE_SIZE, E, D>,
        Error = D::Err,
    >,
{
    type Output = TfmrHidden<{ HEAD_DIM * N_HEADS }, E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: TfmrHidden<{ HEAD_DIM * N_HEADS }, E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let x = self.c_fc.try_forward(input)?;
        let x = self.activation.try_forward(x)?;
        let x = self.c_proj.try_forward(x)?;
        self.dropout.try_forward(x)
    }
}

impl<
        const HEAD_DIM: usize,
        const INTERMEDIATE_SIZE: usize,
        const N_HEADS: usize,
        Activation,
        E: Dtype,
        D: Device<E>,
    > TensorCollection<E, D> for Gpt2Mlp<HEAD_DIM, INTERMEDIATE_SIZE, N_HEADS, Activation, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    Activation: Default,
{
    type To<E2: Dtype, D2: Device<E2>> =
        Gpt2Mlp<HEAD_DIM, INTERMEDIATE_SIZE, N_HEADS, Activation, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("c_fc", |s| &s.c_fc, |s| &mut s.c_fc),
                Self::module("c_proj", |s| &s.c_proj, |s| &mut s.c_proj),
            ),
            |(c_fc, c_proj)| Gpt2Mlp {
                c_fc,
                c_proj,
                activation: Activation::default(),
                dropout: Dropout { p: 0.0 },
            },
        )
    }
}

struct Gpt2Block<
    const HEAD_DIM: usize,
    const N_HEADS: usize,
    const INTERMEDIATE_SIZE: usize,
    const MAX_POSITION: usize,
    Activation,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
{
    layer_idx: Option<usize>,
    // we don't use cross attention for now
    // cross_attention: GPTCrossAttention<HEAD_DIM, MAX_POSITION, ATTENTION_HEADS, E, D>,
    //
    ln_1: TransformersCompatLayrNorm1D<{ HEAD_DIM * N_HEADS }, E, D>,
    attention: Gpt2SelfAttention<HEAD_DIM, MAX_POSITION, N_HEADS, E, D>,
    ln_2: TransformersCompatLayrNorm1D<{ HEAD_DIM * N_HEADS }, E, D>,
    mlp: Gpt2Mlp<HEAD_DIM, INTERMEDIATE_SIZE, N_HEADS, Activation, E, D>,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const INTERMEDIATE_SIZE: usize,
        const MAX_POSITION: usize,
        Activation,
        E: Dtype,
        D: Device<E> + TensorFromVec<bool>,
    > Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for Gpt2Block<HEAD_DIM, N_HEADS, INTERMEDIATE_SIZE, MAX_POSITION, Activation, E, D>
where
    Activation: Module<
            TfmrHidden<INTERMEDIATE_SIZE, E, D>,
            Output = TfmrHidden<INTERMEDIATE_SIZE, E, D>,
            Error = D::Err,
        > + Default,

    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let normalized = self.ln_1.try_forward(input.clone())?;
        let attn_out = self.attention.try_forward(normalized)?;
        let hidden_state = input + attn_out;
        let normalized = self.ln_2.try_forward(hidden_state.clone())?;
        let out = self.mlp.try_forward(normalized)? + hidden_state;
        Ok(out)
    }
}
impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const INTERMEDIATE_SIZE: usize,
        const MAX_POSITION: usize,
        Activation,
        E: Dtype,
        D: Device<E> + TensorFromVec<bool>,
    > TensorCollection<E, D>
    for Gpt2Block<HEAD_DIM, N_HEADS, INTERMEDIATE_SIZE, MAX_POSITION, Activation, E, D>
where
    Activation: Default,

    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> =
        Gpt2Block<HEAD_DIM, N_HEADS, INTERMEDIATE_SIZE, MAX_POSITION, Activation, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("ln_1", |s| &s.ln_1, |s| &mut s.ln_1),
                Self::module("attn", |s| &s.attention, |s| &mut s.attention),
                Self::module("ln_2", |s| &s.ln_2, |s| &mut s.ln_2),
                Self::module("mlp", |s| &s.mlp, |s| &mut s.mlp),
            ),
            |(ln_1, attn, ln_2, mlp)| Gpt2Block {
                layer_idx: None,
                attention: attn,
                ln_1,
                mlp,
                ln_2,
            },
        )
    }
}

struct TransformersCompatLayrNorm1D<const DIM: usize, E: Dtype, D: Device<E>>(
    LayerNorm1D<DIM, E, D>,
);

impl<const DIM: usize, E: Dtype, D: Device<E>> TensorCollection<E, D>
    for TransformersCompatLayrNorm1D<DIM, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = TransformersCompatLayrNorm1D<DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err>
    where
        V::E2: num_traits::FromPrimitive,
    {
        visitor.visit_fields(
            (
                Self::module("bias", |s| &s.0.beta, |s| &mut s.0.beta),
                Self::module("weight", |s| &s.0.gamma, |s| &mut s.0.gamma),
            ),
            |(weight, bias)| {
                TransformersCompatLayrNorm1D(LayerNorm1D {
                    gamma: weight,
                    beta: bias,
                    epsilon: 1e-5,
                    //epsilon: <V::E2 as num_traits::FromPrimitive>::from_f32(1e-5).unwrap(),
                })
            },
        )
    }
}

impl<const DIM: usize, B: Shape, E: Dtype, D: Device<E>> Module<Tensor<B, E, D>>
    for TransformersCompatLayrNorm1D<DIM, E, D>
where
    LayerNorm1D<DIM, E, D>: Module<Tensor<B, E, D>>,
{
    type Output = <LayerNorm1D<DIM, E, D> as Module<Tensor<B, E, D>>>::Output;

    type Error = <LayerNorm1D<DIM, E, D> as Module<Tensor<B, E, D>>>::Error;

    fn try_forward(&self, input: Tensor<B, E, D>) -> Result<Self::Output, Self::Error> {
        self.0.try_forward(input)
    }
}

struct Gpt2Module<
    const HEAD_DIM: usize,
    const N_HEADS: usize,
    const VOCAB_SIZE: usize,
    const INTERMEDIATE_SIZE: usize,
    const MAX_POSITION: usize,
    const HIDDEN_LAYERS: usize,
    Activation,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
{
    token_embeddings: Embeddings<VOCAB_SIZE, { HEAD_DIM * N_HEADS }, E, D>,
    positional_embeddings: Embeddings<MAX_POSITION, { HEAD_DIM * N_HEADS }, E, D>,
    hidden_layers: Repeated<
        Gpt2Block<HEAD_DIM, N_HEADS, INTERMEDIATE_SIZE, MAX_POSITION, Activation, E, D>,
        HIDDEN_LAYERS,
    >,
    ln_f: TransformersCompatLayrNorm1D<{ HEAD_DIM * N_HEADS }, E, D>,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const VOCAB_SIZE: usize,
        const INTERMEDIATE_SIZE: usize,
        const MAX_POSITION: usize,
        const HIDDEN_LAYERS: usize,
        Activation,
        E: Dtype,
        D: Device<E> + TensorFromVec<i32> + TensorFromVec<bool>,
    > Module<Tensor<(usize, Const<1>), i32, D>>
    for Gpt2Module<
        HEAD_DIM,
        N_HEADS,
        VOCAB_SIZE,
        INTERMEDIATE_SIZE,
        MAX_POSITION,
        HIDDEN_LAYERS,
        Activation,
        E,
        D,
    >
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
    Activation: Module<
            Tensor<(usize, Const<{ INTERMEDIATE_SIZE }>), E, D>,
            Error = D::Err,
            Output = Tensor<(usize, Const<INTERMEDIATE_SIZE>), E, D>,
        > + Default,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<1>), i32, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = D::default();
        let input_shape = input.shape();
        let positions: Tensor<(usize, Const<1>), i32, D> =
            dev.tensor_from_vec((0i32..input_shape.0 as i32).collect(), *input_shape);
        let tok_embeddings = self.token_embeddings.try_forward(input)?;
        let pos_embeddings = self.positional_embeddings.try_forward(positions)?;
        let init_embeddings = pos_embeddings + tok_embeddings;
        let all_hidden = self
            .hidden_layers
            .modules
            .iter()
            .try_fold(init_embeddings, |hidden_output, layer| {
                layer.try_forward(hidden_output)
            })?;
        self.ln_f.try_forward(all_hidden)
    }
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const VOCAB_SIZE: usize,
        const INTERMEDIATE_SIZE: usize,
        const MAX_POSITION: usize,
        const HIDDEN_LAYERS: usize,
        Activation,
        E: Dtype,
        D: Device<E> + TensorFromVec<i32> + TensorFromVec<bool>,
    > TensorCollection<E, D>
    for Gpt2Module<
        HEAD_DIM,
        N_HEADS,
        VOCAB_SIZE,
        INTERMEDIATE_SIZE,
        MAX_POSITION,
        HIDDEN_LAYERS,
        Activation,
        E,
        D,
    >
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); 3 * HEAD_DIM * N_HEADS]: Sized,
    [(); 2 * HEAD_DIM * N_HEADS]: Sized,
    Activation: Default,
{
    type To<E2: Dtype, D2: Device<E2>> = Gpt2Module<
        HEAD_DIM,
        N_HEADS,
        VOCAB_SIZE,
        INTERMEDIATE_SIZE,
        MAX_POSITION,
        HIDDEN_LAYERS,
        Activation,
        E2,
        D2,
    >;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("wte", |s| &s.token_embeddings, |s| &mut s.token_embeddings),
                Self::module(
                    "wpe",
                    |s| &s.positional_embeddings,
                    |s| &mut s.positional_embeddings,
                ),
                Self::module("ln_f", |s| &s.ln_f, |s| &mut s.ln_f),
                Self::module("h", |s| &s.hidden_layers, |s| &mut s.hidden_layers),
            ),
            |(wte, wpe, ln_f, hidden_layers)| Gpt2Module {
                positional_embeddings: wpe,
                token_embeddings: wte,
                hidden_layers,
                ln_f,
            },
        )
    }
}

#[cfg(test)]
mod test {

    use dfdx::{
        prelude::{BuildModule, GeLUCorrect, LoadFromSafetensors, Module},
        shapes::{Const, HasShape, Shape},
        tensor::{Cpu, Tensor, ZerosTensor},
    };
    use pyo3::{
        types::PyModule,
        types::{PyList, PyUnicode},
        PyAny, Python,
    };

    use crate::{
        gpt2::Gpt2Module,
        test::{get_tokenized_sample, load_tensor},
    };

    trait Prod {
        fn prod(&self) -> f32;
    }

    impl<const R: usize> Prod for (usize, Const<R>) {
        fn prod(&self) -> f32 {
            (self.0 * R) as f32
        }
    }

    impl<const S: usize, const R: usize> Prod for (Const<R>, usize, Const<S>) {
        fn prod(&self) -> f32 {
            (S * self.1 * R) as f32
        }
    }
    impl<const R: usize> Prod for (Const<R>, usize, usize) {
        fn prod(&self) -> f32 {
            (self.2 * self.1 * R) as f32
        }
    }

    #[test]
    fn test_gpt2_against_py() {
        const GPT2_DATA: &str = "test_data/gpt2.safetensors";
        const GPT2_OUT: &str = "test_data/gpt2_out.safetensors";
        const GPT2_OUT_KEY: &str = "gpt2_output";
        let dev = Cpu::default();
        let tokens = get_tokenized_sample(&dev);
        let tokens_clone = tokens.clone();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(move |py| {
            let transformers =
                PyModule::import(py, "transformers").expect("Failed to import transformers");

            let st_to = PyModule::import(py, "safetensors.torch")
                .expect("failed to get torch submodule of safetensors");

            let model = transformers
                .getattr("AutoModelForCausalLM")
                .expect("Failed to get AutoModel")
                .getattr("from_pretrained")
                .expect("Failed to get from_pretrained method")
                .call((PyUnicode::new(py, "cerebras/Cerebras-GPT-111M"),), None)
                .expect("failed to load model in py")
                .getattr("transformer")
                .expect("failed to get transformer from causal lm model");

            let state_dict = model
                .getattr("state_dict")
                .expect("Failed to get state dict")
                .call((), None)
                .expect("failed to call state dict");
            let dict_transform: &PyAny = PyModule::from_code(
                py,
                r#"""
def dict_transform(dict):
    return {key + ".": val for key, val in dict.items()}
"""#,
                "",
                "",
            )
            .expect("failed to create dict transform function")
            .getattr("dict_transform")
            .expect("failed to get function we just defined");

            let state_dict = dict_transform
                .call((state_dict,), None)
                .expect("Failed to transform dict");

            st_to
                .getattr("save_file")
                .expect("Failed to get method save_file")
                .call((state_dict, GPT2_DATA), None)
                .expect("failed to save gpt2 safetensors");

            let transformer = model;
            //             let transformer = PyModule::from_code(
            //                 py,
            //                 r#"""
            // import torch as to
            // def modify_activation(transformer):
            //     for layer in transformer.h:
            //         layer.mlp.act = to.nn.GELU(approximate='tanh')
            //     return transformer
            // """#,
            //                 "",
            //                 "",
            //             )
            //             .expect("failed to create function modify_activation")
            //             .getattr("modify_activation")
            //             .expect("failed to get just created function modify_activation")
            //             .call((model,), None)
            //             .expect("failed to call just created function modify activation");

            let py_tokens = PyList::new(py, tokens_clone.as_vec().iter());

            PyModule::from_code(
                py,
                r#"""
import torch as to
import safetensors as sf
import datetime as dt
def apply_and_save(tokens, transformer, key, file):
    tokens = to.LongTensor(tokens)
    t1 = dt.datetime.now()
    output = transformer(tokens)[0]
    delta_t = dt.datetime.now() - t1
    print(f"pytorch took {delta_t}")
    sf.torch.save_file({key: output}, file)
"""#,
                "",
                "",
            )
            .expect("failed to create apply_and_save_method")
            .getattr("apply_and_save")
            .expect("failed to get apply_and_save")
            .call(
                (
                    py_tokens,
                    transformer,
                    PyUnicode::new(py, GPT2_OUT_KEY),
                    PyUnicode::new(py, GPT2_OUT),
                ),
                None,
            )
            .expect("failed to call apply_and_save");
        });
        const HEAD_DIM: usize = 64;
        const N_HEADS: usize = 12;
        const MAX_POSITION: usize = 2048;
        const HIDDEN_LAYERS: usize = 10;
        const VOCAB_SIZE: usize = 50257;
        const INTERMEDIATE_SIZE: usize = 3072;

        let mut gpt2: Gpt2Module<
            HEAD_DIM,
            N_HEADS,
            VOCAB_SIZE,
            INTERMEDIATE_SIZE,
            MAX_POSITION,
            HIDDEN_LAYERS,
            GeLUCorrect,
            f32,
            Cpu,
        > = Gpt2Module::build(&dev);

        // gpt2.save_safetensors("test_data/gpt2_test.safetensors");
        gpt2.load_safetensors(GPT2_DATA);

        // gpt2.ln_f
        //     .load_safetensors("test_data/gpt2_ln_f.safetensors");

        let tokens = get_tokenized_sample(&dev);
        std::thread::sleep(std::time::Duration::from_secs(10));
        let t1 = std::time::Instant::now();
        for _ in 0..500 {
            let output = gpt2
                .try_forward(tokens.clone())
                .expect("failed to call gpt forward");
        }
        let dt = std::time::Instant::now() - t1;

        let output = gpt2
            .try_forward(tokens)
            .expect("failed to call gpt forward");
        println!("dfdx took {} µs", dt.as_micros() / 500);

        let mut output_should = dev.zeros_like(output.shape());
        load_tensor(GPT2_OUT, GPT2_OUT_KEY, &mut output_should);
        let output_slice = output.slice((5..6, ..));
        let should_slice = output_should.slice((5..6, ..));
        let (abs, diff) = diff_tens(&output_slice, &should_slice);
        println!("Diff: {diff}, abs: {abs}");
        assert!(diff <= 1.0e-5);
    }

    fn diff_tens<S: Shape + Prod>(
        left: &Tensor<S, f32, Cpu>,
        right: &Tensor<S, f32, Cpu>,
    ) -> (f32, f32) {
        let entries = left.shape().prod();
        let diff = (left.clone() - right.clone())
            .abs()
            .as_vec()
            .into_iter()
            .sum::<f32>()
            / entries;
        let abs = left.clone().abs().as_vec().into_iter().sum::<f32>() / entries;
        (abs, diff)
    }
}
