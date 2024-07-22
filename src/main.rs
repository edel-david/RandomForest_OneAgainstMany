#![allow(unused)]
// TODO: all trees per class have same trainin data. change!
mod float64;
mod forest;
mod node;
mod reg_tree;
mod utils;
use float64::Float64;
use forest::bootstrap_sampling;
use node::Node;
use rand::{thread_rng, Rng};
use reg_tree::RegressionTree;
//extern crate supa_cube_forest;
//use crate::utils::*;
use core::hash::Hash;
use core::ops::Add;
use std::cmp::Ordering;
use std::fmt::{Debug, Error};
use std::hash::Hasher;
use std::iter::zip;
use std::ops::{Deref, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;
use std::{mem, usize};
//use core::slice::SlicePattern;
use crate::forest::Forest;
use conv::prelude::*;
use ndarray::{prelude::*, ScalarOperand, ViewRepr};
use ndarray_npy::{read_npy, ReadableElement};
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};
use num::{Float, FromPrimitive, NumCast, One, Signed, ToPrimitive, Zero};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::{collections::HashMap, hash};
use std::{f64::INFINITY, marker::PhantomData};

pub struct VoterForest<G, R>
where
    G: From<f64> + FromPrimitive + Float,
    R: Clone + num::Zero,
{
    amount_trees: usize,
    class: usize,
    trees: Vec<RegressionTree<G, R>>,
    n_min: usize,
    dtry: Option<usize>,
}

impl<G, R> VoterForest<G, R>
where
    G: Add
        + From<usize>
        + Debug
        + ValueFrom<usize>
        + From<f64>
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
        + From<i8>
        ,
    R: Clone
        + num::Zero
        + From<f64>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed
        + FromPrimitive
        +From<i8>,
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
                acc + ((result.signum() - (*response).into()).abs() < R::from(0.1_f64)) as usize
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

    pub fn new(class: usize, amount_trees: usize, n_min: usize, D_try: usize) -> Self {
        //let mut trees = Vec::with_capacity(amount_trees);
        let trees = (0..amount_trees)
            .map(|_| RegressionTree {
                root: None,
                n_min,
                d_try: D_try,
            })
            .collect();
        Self {
            class,
            amount_trees,
            trees,
            n_min,
            dtry: Some(D_try),
        }
    }

    pub fn train<T: Clone + Into<R> + Into<G> + num::Zero + Debug  +From<i8>+Eq>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
    ) {
        let n = responses.dim();
        for tree in &mut self.trees {
            let mut voter_indices = vec![];
            let mut other_indices = vec![];
            let t1 :T = 1i8.into();
            //let indices_voter_index

            for (index_response, resp) in responses.iter().enumerate() {
                if *resp == t1 {
                    voter_indices.push(index_response);
                } else {
                    other_indices.push(index_response);
                }
            }
            // now do balanced_bootstrap_sampling with the indices;
            //            let voter_indidces = Array1::from_vec(voter_indices);
            //let other_indices = Array1::from_vec(other_indices);
            let (mut rand_sampl_votes, rand_sampl_other) =
                balanced_bootstrap_sampling(n, voter_indices, other_indices);

            rand_sampl_votes.extend(rand_sampl_other);
            let to_train = features.select(Axis(0), &rand_sampl_votes);
            let to_train_resp = responses.select(Axis(0), &rand_sampl_votes);
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

pub struct OneAgainstMany<G, R>
where
    G: From<f64> + FromPrimitive + Float,
    R: Clone + num::Zero,
{
    pub amount_trees: usize,
    pub amount_classes: usize, // aka amount_forests
    forests: Vec<VoterForest<G, R>>,
    pub n_min: usize,
    pub dtry: Option<usize>,
}
impl<G, R> OneAgainstMany<G, R>
where
    G: Add
        + From<usize>
        + Debug
        + ValueFrom<usize>
        + From<f64>
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
        + From<i8>
        ,
    R: Clone
        + num::Zero
        + From<f64>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed
        + FromPrimitive
        + From<usize>
        +From<i8>,
{
    pub fn evaluate(&self, data: &Array2<G>, responses: &Array1<usize>) -> G {
        let results = self.predict_batch(data);
        let sum_correct = results
            .iter()
            .zip(responses)
            .fold(0, |acc: usize, (result, response)| {
                acc + ((*result == *response) as usize)
            });
        dbg!(&sum_correct);
        let sum_correct_g: G = sum_correct.into();
        let len: G = responses.len().into();
        let percentage = sum_correct_g / len;
        return percentage;
    }
    pub fn predict_batch(&self, features: &Array2<G>) -> Array1<usize> {
        let mut results = Array1::zeros([features.dim().0]);
        for (row, e) in zip(features.rows(), 0..) {
            results[e] = self.predict(&row.to_owned());
        }
        return results;
        todo!();
    }
    pub fn predict(&self, features: &Array1<G>) -> usize {
        let results = self.predict_stoch(features);
        let res_arr = Array1::from_vec(results);
        let winner = res_arr.argmax().expect("no errors for usize");
        return winner;
    }
    pub fn predict_stoch(&self, features: &Array1<G>) -> Vec<R> {
        let mut results = vec![];
        for fores in &self.forests {
            results.push(fores.predict(&features));
        }
        return results;
    }

    pub fn new(
        amount_trees: usize,
        amount_classes: usize,
        n_min: usize,
        d_try: Option<usize>,
    ) -> Self {
        let forests: Vec<_> = (0..amount_classes)
            .map(|index| VoterForest::<G, R>::new(index, amount_trees, n_min, d_try.unwrap_or(8)))
            .collect();
        Self {
            amount_classes,
            amount_trees,
            forests,
            n_min,
            dtry: d_try,
        }
    }

    pub fn train<T: Clone + Into<R> + Into<G> + num::Zero + Debug + Into<usize>>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
    ) {
        let n = responses.dim();
        for (index_forest, class_forest) in (self.forests).iter_mut().enumerate() {
            // train forest
            // prepare training features and responses
            // the features and responses both should be 50% of the voter class and 50% everyting else
            // we need to change our responses to -1 and 1.

            let work_response = responses.clone();
            let work_response =
                work_response.mapv(|class| {let g = (<T as Into<usize>>::into(class) == index_forest)as i8;
                    // trasform 0,1 into -1,1
                    g-(1-g)
                }) ;

            class_forest.train(features, &work_response);
        }
    }
}

pub fn balanced_bootstrap_sampling(
    n: usize,
    voter_indices: Vec<usize>,
    other_indices: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let half_amount = n / 2; // dont care what this rounds to
    let mut rng = thread_rng();
    let random_samples_votes: Vec<usize> = (0..half_amount)
        .map(|_| voter_indices[rng.gen_range(0..voter_indices.len())])
        .collect();
    let random_samples_other: Vec<usize> = (0..half_amount)
        .map(|_| other_indices[rng.gen_range(0..other_indices.len())])
        .collect();
    return (random_samples_votes, random_samples_other);

    todo!();
    //    return indices;
}

fn main() {
    type G = Float64;
    type R = Float64;
    let data: Array2<f64> = read_npy("src/digits_data.npy").unwrap();
    let targets: Array1<i64> = read_npy("src/digits_target.npy").unwrap();
    let data = data.mapv(|x| Float64(x as f64));
    let targets = targets.mapv(|x| x as usize);

    let mut all_indices: Vec<_> = (0..targets.dim()).collect();
    let mut rng = ndarray_rand::rand::thread_rng();
    all_indices.shuffle(&mut rng);
    //dbg!(targets.slice(s![0..20]));

    let train_indices = all_indices[(targets.dim() / 5)..].to_vec();
    let test_indices = all_indices[..(targets.dim() / 5)].to_vec();

    // let train_indices = all_indices.clone();
    // let test_indices = all_indices;

    let train_features = data.select(Axis(0), &train_indices);
    let train_targets = targets.select(Axis(0), &train_indices);

    let test_features = data.select(Axis(0), &test_indices);
    let test_targets = targets.select(Axis(0), &test_indices);
    let mut reg_classi = OneAgainstMany::<G, R>::new(20, 10, 5, Some(8));
    reg_classi.train(&train_features, &train_targets);
    let accu = reg_classi.evaluate(&test_features, &test_targets);
    println!("{:?}", accu);

    // let mut reg_classi = OneAgainstMany::<G, R>::new(20, 4, 1, Some(2));

    // let data = array![[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]];
    // let data = data.mapv(|x| Float64(x as f64));
    // let targets = array![0,1,2,3];

    // reg_classi.train(&data, &targets);
    // let accu = reg_classi.evaluate(&data, &targets);
    // println!("{:?}", accu);
}
// T is the type of the response values
//G is type of feature data Vec<>
// R is the predict return Type
use std::thread::sleep;
#[cfg(test)]
mod tests {
    use super::*;
    type G = Float64;
    type T = i64;
    type R = Float64;
    #[test]
    fn one_against_many() {}
}
