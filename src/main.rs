#![allow(clippy::needless_return)]
// TODO: all trees per class have same trainin data. change!
mod float64;
mod forest;
mod node;
mod reg_tree;
mod utils;
use conv::prelude::*;
use core::hash::Hash;
use core::ops::Add;
use float64::Float64;
use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use reg_tree::RegressionTree;
use std::fmt::Debug;
use std::iter::zip;

use ndarray_stats::QuantileExt;
use num::{integer::Roots, Float, FromPrimitive, Signed};

#[allow(unused)]
use ndarray_npy::read_npy;
#[allow(unused)] // needed for tests and debugging
use rand::seq::SliceRandom;

type R = Float64;
type G = Float64;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

pub struct VoterForest<G, R>
where
    G: From<f64> + FromPrimitive + Float,
    R: Clone + num::Zero,
{
    amount_trees: usize,
    trees: Vec<RegressionTree<G, R>>,
    // n_min: usize, // has implicit n_min in the trees
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
        + Into<usize>+Send+Sync,
    R: Clone
        + num::Zero
        + From<f64>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed
        + FromPrimitive
        + From<i8>+Send+Sync,
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

    pub fn new(amount_trees: usize, n_min: usize) -> Self {
        //let mut trees = Vec::with_capacity(amount_trees);
        let trees = (0..amount_trees)
            .map(|_| RegressionTree { root: None, n_min })
            .collect();
        Self {
            amount_trees,
            trees,
        }
    }

    pub fn train<T: Clone + Into<R> + Into<G> + num::Zero + Debug + From<i8> + Eq+Send+Sync>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
        voter_indices: &Vec<usize>,
        other_indices: &Vec<usize>,
        dtry: Option<usize>,
    ) {
        let n = responses.dim();
        let dtry = dtry.unwrap_or_else(|| features.dim().1.sqrt());
        for tree in &mut self.trees {
            // now do balanced_bootstrap_sampling with the indices;
            //            let voter_indidces = Array1::from_vec(voter_indices);
            //let other_indices = Array1::from_vec(other_indices);
            // let (mut rand_sampl_votes, rand_sampl_other) =
            //     balanced_bootstrap_sampling(n, voter_indices, other_indices);

            // rand_sampl_votes.extend(rand_sampl_other);

            let rand_samples = balanced_bootstrap_sampling(n, voter_indices, other_indices);
            let to_train = features.select(Axis(0), &rand_samples);
            let to_train_resp = responses.select(Axis(0), &rand_samples);
            let _ = tree.train(&to_train, &to_train_resp, Some(dtry));
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
}
impl<G, R> OneAgainstMany<G, R>
where
    G: Add
    +Send
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
        + Into<usize>
        +Sync,
    R: Clone
    +Send
        + num::Zero
        + From<f64>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed
        + FromPrimitive
        + From<usize>
        + From<i8>
        +Sync,
{
    pub fn evaluate(&self, data: &Array2<G>, responses: &Array1<i8>) -> G {
        let results = self.predict_batch(data);
        let sum_correct =
            results
                .iter()
                .zip(responses)
                .fold(0, |acc: usize, (result, response)| {
                    acc + ((*result == response.clone() as usize) as usize)
                });
        dbg!(&sum_correct);
        let sum_correct_g: G = sum_correct.into();
        let len: G = responses.len().into();
        let percentage = sum_correct_g / len;
        return percentage;
    }
    pub fn predict_batch(&self, features: &Array2<G>) -> Array1<usize> {
        let results :Vec<usize>= features.axis_iter(Axis(0)).into_par_iter().map(|x|self.predict(&x.to_owned())).collect();
        // for (row, e) in zip(features.rows(), 0..) {
        //     results[e] = self.predict(&row.to_owned());
        // }
        return Array1::from(results);
    }
    pub fn predict(&self, features: &Array1<G>) -> usize {
        let results = self.predict_stoch(features);
        let res_arr = Array1::from_vec(results);
        let winner = res_arr
            .argmax()
            .expect("no errors for usize and nontrivial classes");
        return winner;
    }
    pub fn predict_stoch(&self, features: &Array1<G>) -> Vec<R> {
        let mut results = vec![];
        for fores in &self.forests {
            results.push(fores.predict(&features));
        }
        return results;
    }

    pub fn new(amount_trees: usize, amount_classes: usize, n_min: usize) -> Self {
        let forests: Vec<_> = (0..amount_classes)
            .map(|_| VoterForest::<G, R>::new(amount_trees, n_min))
            .collect();
        Self {
            amount_classes,
            amount_trees,
            forests,
            n_min,
        }
    }

    pub fn train<T: Into<i8> + Clone + Into<R> + Into<G> + num::Zero + Debug + Eq>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
        dtry: Option<usize>,
    ) {
        let responses: ArrayBase<ndarray::OwnedRepr<i8>, Dim<[usize; 1]>> =
            responses.mapv(|x| x.into());
        // try to train all Voter-Forests in parallel


        (self.forests)
            .par_iter_mut()
            .enumerate()
            .for_each(|(index_forest, class_forest)| {
                // train forest
                // prepare training features and responses
                // the features and responses both should be 50% of the voter class and 50% everyting else
                // we need to change our responses to -1 and 1.
                let mut responses_class = responses.clone();
                let mut voter_indices = vec![];
                let mut other_indices = vec![];

                //let indices_voter_index

                for (index_response, resp) in responses_class.iter_mut().enumerate() {
                    if *resp == index_forest as i8 {
                        voter_indices.push(index_response);
                        *resp = 1;
                    } else {
                        other_indices.push(index_response);
                        *resp = -1;
                    }
                }

                class_forest.train(
                    &features.clone(),
                    &responses_class,
                    &voter_indices,
                    &other_indices,
                    dtry,
                );
            });
    }
}

pub fn balanced_bootstrap_sampling(
    n: usize,
    voter_indices: &Vec<usize>,
    other_indices: &Vec<usize>,
) -> Vec<usize> {
    let half_amount = n / 2; // dont care what this rounds to
    let mut rng = thread_rng();

    let mut random_samples = Vec::with_capacity(n);
    random_samples
        .extend((0..half_amount).map(|_| voter_indices[rng.gen_range(0..voter_indices.len())]));
    random_samples
        .extend((0..half_amount).map(|_| other_indices[rng.gen_range(0..other_indices.len())]));

    // let random_samples_votes: Vec<usize> = (0..half_amount)
    //     .map(|_| voter_indices[rng.gen_range(0..voter_indices.len())])
    //     .collect();
    // let random_samples_other: Vec<usize> = (0..half_amount)
    //     .map(|_| other_indices[rng.gen_range(0..other_indices.len())])
    //     .collect();
    // return (random_samples_votes, random_samples_other);
    return random_samples;
}

fn main() {
    let x_train: Array2<u8> = read_npy("src/fash_X_train.npy").unwrap();
    let x_test: Array2<u8> = read_npy("src/fash_X_test.npy").unwrap();
    let z_train: Array1<u8> = read_npy("src/fash_Y_train.npy").unwrap();
    let z_test: Array1<u8> = read_npy("src/fash_Y_test.npy").unwrap();

    let x_train = x_train.mapv(|x| Float64::from(x));
    let x_test = x_test.mapv(|x| Float64::from(x));
    let z_train = z_train.mapv(|x| x as i8);
    let z_test = z_test.mapv(|x| x as i8);
    


    let mut reg_classi: OneAgainstMany<Float64, Float64> = OneAgainstMany::<G, R>::new(20, 10, 5);
    reg_classi.train(&x_train, &z_train, Some(8));
    let accu = reg_classi.evaluate(&x_test, &z_test);
    println!("{:?}", accu);
}

#[cfg(test)]
mod tests {
    use super::*;
    //G is type of feature data Vec<>
    // R is the predict return Type
    type R = Float64;
    type G = Float64;

    #[test]
    fn one_against_many() {
        let data: Array2<f64> = read_npy("src/digits_data.npy").unwrap();
        let targets: Array1<i64> = read_npy("src/digits_target.npy").unwrap();
        let data = data.mapv(|x| Float64(x as f64));
        let targets = targets.mapv(|x| x as i8);
    
        let mut all_indices: Vec<_> = (0..targets.dim()).collect();
        let mut rng = ndarray_rand::rand::thread_rng();
        all_indices.shuffle(&mut rng);
        let train_indices = all_indices[(targets.dim() / 5)..].to_vec();
        let test_indices = all_indices[..(targets.dim() / 5)].to_vec();
    
        // let train_indices = all_indices.clone();  // in case you want to use all indices
        // let test_indices = all_indices;
    
        let train_features = data.select(Axis(0), &train_indices);
        let train_targets = targets.select(Axis(0), &train_indices);
    
        let test_features = data.select(Axis(0), &test_indices);
        let test_targets = targets.select(Axis(0), &test_indices);
        let mut reg_classi: OneAgainstMany<Float64, Float64> = OneAgainstMany::<G, R>::new(10, 10, 20);
        reg_classi.train(&train_features, &train_targets, Some(16));
        let accu = reg_classi.evaluate(&test_features, &test_targets);
        println!("{:?}", accu);
    }
    #[test]
    fn simplest_one_against_many() {
        let mut reg_classi = OneAgainstMany::<G, R>::new(20, 4, 1);

        let data = array![[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]];
        let data = data.mapv(|x| Float64(x as f64));
        let targets = array![0, 1, 2, 3];

        reg_classi.train(&data, &targets, Some(2));
        let accu = reg_classi.evaluate(&data, &targets);
        println!("{:?}", accu);
        assert_eq!(accu, Float64(1.0));
    }
}
