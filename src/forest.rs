#![allow(unused)]
use crate::reg_tree::RegressionTree;
use conv::ValueFrom;
use core::hash::Hash;
use ndarray::prelude::*;
use ndarray::Array1;
use num::{Float, FromPrimitive, Signed};
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::Add;

// type G = Float64; // G is type of feature data Vec<>
// type R = Float64; // R is the predict return Type

pub struct Forest<G, R>
where
    G: From<f32> + FromPrimitive + Float,
    R: Clone + num::Zero,
{
    amount_trees: usize,
    trees: Vec<RegressionTree<G, R>>,
    n_min: usize,
    dtry: Option<usize>,
}

impl<G, R> Forest<G, R>
where
    G: Add
        + From<usize>
        + Debug
        + ValueFrom<usize>
        + From<f32>
        + Hash
        + FromPrimitive
        + Float
        + From<u8>
        + From<R>
        + Ord
        + ndarray::ScalarOperand
        + std::ops::Add<Output = G>
        + core::ops::Add
        + Copy
        +Into<usize>+Send+Sync,
    R: Clone
        + num::Zero
        + From<f32>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed
        + FromPrimitive+Send+Sync,
{

    pub fn evaluate<T: Copy + Clone + Into<G> + Into<R> + Debug>(
        &self,
        data: &Array2<G>,
        responses: &Array1<T>,
    ) -> G {
        let results = self.predict_batch(data);
        let sum_correct = results
            .iter()
            .zip(responses)
            .fold(0, |acc, (result, response)| {
                acc + ((result.signum() - (*response).into()).abs() < R::from(0.1_f32)) as usize
            });

        let sum_correct_g: G = sum_correct.into();
        let len: G = responses.len().into();
        let percentage = sum_correct_g / len;
        return percentage;
    }

    pub fn predict_batch(&self, features: &Array2<G>) -> Array1<R> {
        let mut results = Array1::zeros([features.dim().0]);
        for (row, e) in zip(features.rows(), 0..) {
            results[e] = self.predict(&row.to_owned());
        }
        return results;
    }

    pub fn new(amount_trees: usize, n_min: usize, d_try: usize) -> Self {
        //let mut trees = Vec::with_capacity(amount_trees);
        let trees = (0..amount_trees)
            .map(|_| RegressionTree {
                root: None,
                n_min
            })
            .collect();
        Self {
            amount_trees,
            trees,
            n_min,
            dtry: Some(d_try),
        }
    }

    pub fn train<T: Clone + Into<R> + Into<G> + num::Zero + Debug+Send+Sync>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
    ) {
        let n = responses.dim();
        for tree in &mut self.trees {
            let indices = bootstrap_sampling(n);
            let to_train = features.select(Axis(0), &indices);
            let to_train_resp = responses.select(Axis(0), &indices);
            tree.train(&to_train, &to_train_resp, None);
        }
    }

    pub fn predict(&self, features: &Array1<G>) -> R {
        let mut results: Vec<R> = Vec::with_capacity(self.amount_trees);
        for tree in &self.trees {
            results.push(tree.predict(features))
        }
        let array = Array1::from_vec(results);
        return array.mean().unwrap();
    }
}

pub fn bootstrap_sampling(n: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
    return indices;
}
#[cfg(test)]
mod tests {

    use super::Forest;
    use super::*;
    use crate::float64::Float64;
    use ndarray_npy::read_npy;
    use rand::seq::SliceRandom;

    #[test]
    fn easiest_test() {
        type G = Float64;
        type R = Float64;

        let data: Array2<f32> =
            read_npy("src/three_9_data.npy").expect("file should be present and correct");
        let data = data.mapv(|x| Float64(x as f32));
        let target: Array1<i64> =
            read_npy("src/three_9_target.npy").expect("file should be present and correct");

        let mut forest = Forest::<G, R>::new(20, 10, 8);

        forest.train(&data, &target);
    }

    #[test]
    fn test_untrained() {
        type G = Float64;
        type R = Float64;

        let data: Array2<f32> =
            read_npy("src/three_9_data.npy").expect("file is present and correct");
        //let data = array![[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]];
        let data = data.mapv(|x| Float64(x as f32));
        let target: Array1<i64> =
            read_npy("src/three_9_target.npy").expect("file is present and correct");

        let mut all_indices: Vec<_> = (0..363).collect();
        let mut rng = ndarray_rand::rand::thread_rng();
        all_indices.shuffle(&mut rng);
        let train_indices = all_indices[73..].to_vec();
        let test_indices = all_indices[..73].to_vec();

        let ts = data.select(Axis(0), &train_indices);
        let ts_target = target.select(Axis(0), &train_indices);

        let test_set_feat = data.select(Axis(0), &test_indices);
        let test_set_resp = target.select(Axis(0), &test_indices);

        let mut forest = Forest::<G, R>::new(20, 10, 8);
        forest.train(&ts,&ts_target);
        let perf = forest.evaluate(&test_set_feat, &test_set_resp);
        dbg!(perf);
    }
}
