use std::{collections::HashSet, f64::INFINITY, iter::zip, ops::Add};

use crate::node::Node;
use conv::ValueFrom;
use core::hash::Hash;
use ndarray::{prelude::*, ScalarOperand, ViewRepr};
use num::{Float, FromPrimitive, Signed};
use rand::seq::SliceRandom;
use std::fmt::{Debug, Error};
#[derive(Debug)]
pub struct RegressionTree<G: Clone, R>
where
    G: From<f64> + FromPrimitive + Float, //From<usize>,
    R: Clone + num::Zero,
{
    pub root: Option<Box<Node<R, G>>>,
    pub n_min: usize,
    pub d_try: usize, //input_data: PhantomData<G>,
}

impl<'a, G: Clone + std::cmp::PartialOrd, R> RegressionTree<G, R>
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
        + Copy,
    R: Clone
        + num::Zero
        + From<f64>
        + From<usize>
        + std::ops::Div<Output = R>
        + std::ops::Sub<Output = R>
        + std::cmp::PartialOrd
        + Signed,
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

    pub fn predict_batch(&self, data: &Array2<G>) -> Array1<R> {
        let mut results = Array1::zeros([data.dim().0]);
        for (row, e) in zip(data.rows(), 0..) {
            results[e] = self.predict(&row.to_owned())
        }
        return results;
    }

    pub fn predict(&self, x: &Array1<G>) -> R {
        let mut node = &(self.root);
        for _i in 0..100 {
            match node.as_ref() {
                Some(val) => match val.as_ref() {
                    Node::LeafNode { prediction } => {
                        return prediction.clone();
                    }
                    Node::SplitNode {
                        left,
                        right,
                        split_index,
                        threashold,
                    } => {
                        if x[*split_index] <= *threashold {
                            node = left
                        } else {
                            node = right
                        }
                    }
                },
                None => {
                    return R::from(0_f64); // if not trained yet, return 0 always
                                           // if trained but still reached => bug
                                           // consider logging here, as it does not make sense to use predict before training
                }
            }
        }

        todo!("predict followed split node for over 100 splits! not possible?");
    }
    /// j is feature index and t is threashold
    fn compute_loss_for_split<T: Clone + Into<G> + Debug>(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
        j: usize,
        t: G,
    ) -> G {
        //let arr = Array1::from(features[0].clone());

        let temp = features.slice(s![.., j]);

        let mut left: Vec<G> = vec![];
        let mut right: Vec<G> = vec![];
        for (x, target) in zip(temp, responses) {
            if x <= &t {
                left.push(target.clone().into())
            } else {
                right.push(target.clone().into())
            }
        }
        let amount_left = (left.len());
        let amount_right = (right.len());
        if (amount_left < self.n_min) | (amount_right < self.n_min) {
            return <G as From<f64>>::from(INFINITY);
        }

        let left = Array1::from_vec(left);
        let right = Array1::from_vec(right);

        let zero = <G as From<f64>>::from(0_f64);
        let loss = left.var(zero) * G::value_from(amount_left).expect("msg")// into G
            + right.var(zero) * G::value_from(amount_right).expect("msg");
        return loss;
    }

    /// return a 1-D array with D_try randomly selected indices from 0...(D-1).
    fn select_active_indices(D: usize, d_try: usize) -> Vec<usize> {
        let mut rng = ndarray_rand::rand::thread_rng();
        let mut myvec: Vec<usize> = (0..D).collect();
        myvec.shuffle(&mut rng);
        myvec[..d_try].to_vec()
    }
    fn find_threasholds<T>(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
        j: usize,
    ) -> Array1<G> {
        // return: a 1-D array with all possible thresholds along feature j
        //(find midpoints between instances of feature j)
        let mut feature_slice = features.select(Axis(1), &[j]);
        let unique_feature: HashSet<_> = feature_slice.iter().cloned().collect();
        let mut unique_feature_array = Array1::from_iter(unique_feature);
        unique_feature_array
            .as_slice_mut()
            .expect("idk if this workds")
            .sort();
        let temp2 = unique_feature_array.clone().select(
            Axis(0),
            Vec::from_iter(1..(unique_feature_array.dim())).as_slice(),
        );
        let mut new = unique_feature_array
            .slice_mut(s![..unique_feature_array.len() - 1])
            .add(temp2);

        let new = new / <G as From<u8>>::from(2);

        return new;
    }
    pub fn create_correct_node<T: Clone + Into<G> + Into<R> + num::Zero + Debug>(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
    ) -> Node<R, G> {
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let mut l_min = <G as From<f64>>::from(INFINITY);
        let mut j_min = 0;
        let mut t_min = <G as From<f64>>::from(0_f64);
        let active_indices = RegressionTree::<G, R>::select_active_indices(D, self.d_try);
        for j in active_indices {
            let threasholds =
                RegressionTree::<G, R>::find_threasholds(&self, &features, responses, j);
            for threashold in threasholds {
                let loss = RegressionTree::<G, R>::compute_loss_for_split(
                    self, features, responses, j, threashold,
                );
                if loss < l_min {
                    l_min = loss;
                    j_min = j;
                    t_min = threashold;
                }
            }
        }
        if l_min == <G as From<f64>>::from(INFINITY) {
            // we did not find a threshold, so this should become a leaf node
            let sum = responses.sum();
            let mut sum_r: R = sum.into();
            let len_r: R = responses.dim().into();
            sum_r = sum_r / len_r;
            return Node::LeafNode::<R, G> { prediction: sum_r };
        } else {
            // threashold exists => split node

            // get left and right
            let temp_to_split = features.slice(s![.., j_min]);
            let mut left_feat: Vec<G> = vec![];
            let mut right_feat: Vec<G> = vec![];
            let mut left_res: Vec<T> = vec![];
            let mut right_res: Vec<T> = vec![];
            for ((x, target), feat) in zip(temp_to_split, responses).zip(features.rows()) {
                if x <= &t_min {
                    left_res.push(target.clone());
                    left_feat.extend(feat);
                } else {
                    right_res.push(target.clone());
                    right_feat.extend(feat);
                }
            }

            let left_feat = Array2::from_shape_vec((left_res.len(), D), left_feat).unwrap();
            let right_feat = Array2::from_shape_vec((right_res.len(), D), right_feat).unwrap();
            let left_res = Array1::from_vec(left_res);
            let right_res = Array1::from_vec(right_res);
            let left = Self::create_correct_node(&self, &left_feat, &left_res);
            let right = Self::create_correct_node(&self, &right_feat, &right_res);
            return Node::SplitNode::<R, G> {
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                split_index: j_min,
                threashold: t_min,
            };
        }

        todo!()
    }
    pub fn train<T: Clone + Into<G> + Into<R> + num::Zero + Debug>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
        d_try: Option<usize>,
    ) -> Result<String, String> {
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let d_try = d_try.unwrap_or(self.d_try);

        //let working_node = &mut self.root;

        let mut l_min = <G as From<f64>>::from(INFINITY);
        let mut j_min = 0;
        let mut t_min = <G as From<f64>>::from(0_f64);

        l_min = <G as From<f64>>::from(INFINITY);
        j_min = 0;
        t_min = <G as From<f64>>::from(0_f64);
        let active_indices = RegressionTree::<G, R>::select_active_indices(D, d_try);
        for j in active_indices {
            let threasholds =
                RegressionTree::<G, R>::find_threasholds(&self, &features, responses, j);
            for threashold in threasholds {
                let loss = RegressionTree::<G, R>::compute_loss_for_split(
                    self, features, responses, j, threashold,
                );
                if loss < l_min {
                    l_min = loss;
                    j_min = j;
                    t_min = threashold;
                }
            }
        }
        let node = if (l_min == <G as From<f64>>::from(INFINITY)) {
            // we did not find a threshold, so this should become a leaf node
            // append leaf node
            //root
            let sum = responses.sum();
            let mut sum_r: R = sum.into();
            let len_r: R = responses.dim().into();
            sum_r = sum_r / len_r;
            Node::LeafNode::<R, G> { prediction: sum_r }
        } else {
            // ELSE : assume a split has been found.
            // Make first Node:

            // make left and right
            let temp_to_split = features.slice(s![.., j_min]);

            // let left_indices = temp_to_split.mapv(|x| (x <= t_min) as usize);
            // let right_indices = left_indices.mapv(|x| 1 - x);
            // let left_feat = features.select(Axis(0), left_indices.as_slice().expect("msg"));
            // let right_feat = features.select(Axis(0), right_indices.as_slice().expect("msg"));
            // let left_resp = responses.select(Axis(0), left_indices.as_slice().expect("msg"));
            // let right_resp = responses.select(Axis(0), right_indices.as_slice().expect("msg"));

            let mut left_feat: Vec<G> = vec![];
            let mut right_feat: Vec<G> = vec![];
            let mut left_res: Vec<T> = vec![];
            let mut right_res: Vec<T> = vec![];
            for ((x, target), feat) in zip(temp_to_split, responses).zip(features.rows()) {
                if x <= &t_min {
                    left_res.push(target.clone());
                    left_feat.extend(feat);
                } else {
                    right_res.push(target.clone());
                    right_feat.extend(feat);
                }
            }

            let left_feat = Array2::from_shape_vec((left_res.len(), D), left_feat).unwrap();
            let right_feat = Array2::from_shape_vec((right_res.len(), D), right_feat).unwrap();
            let left_res = Array1::from_vec(left_res);
            let right_res = Array1::from_vec(right_res);

            let left = Self::create_correct_node(&self, &left_feat, &left_res);
            let right = Self::create_correct_node(&self, &right_feat, &right_res);

            Node::SplitNode::<R, G> {
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                split_index: j_min,
                threashold: t_min,
            }
        };
        self.root = Some(Box::new(node));
        return Ok(String::from("sucess"));
        todo!()
    }
}

#[cfg(test)]
mod tests {

    use ndarray_npy::read_npy;

    use crate::float64::Float64;

    use super::*;

    #[test]
    fn easiest_test() {
        type G = Float64;
        type T = i64;
        type R = Float64;
        let data = array![[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]];
        let data = data.mapv(|x| Float64(x as f64));
        let target = array![-1, -1, 1, 1];

        let mut tree: RegressionTree<G, R> = RegressionTree {
            root: None,
            n_min: 2,
            d_try: 2,
        };
        tree.train(&data, &target, None); // use full data to train => 100% correct
        let preds = tree.predict_batch(&data);
        let preds_cast: Array1<i32> = preds.mapv(|x| x.signum().into());
        assert_eq!(preds_cast, target);
    }

    #[test]
    fn test() {
        type G = Float64;
        type T = i64;
        type R = Float64;

        let data: Array2<f64> =
            read_npy("src/three_9_data.npy").expect("file should be present and correct");
        let data = data.mapv(|x| Float64(x as f64));
        let target: Array1<i64> =
            read_npy("src/three_9_target.npy").expect("file should be present and correct");

        let mut tree: RegressionTree<G, R> = RegressionTree {
            root: None,
            n_min: 5,
            d_try: 8,
        };

        tree.train(&data, &target, None);
        let val = tree.evaluate(&data, &target);
        let val = tree.evaluate(&data, &target);
        let val = tree.evaluate(&data, &target);
        let val = tree.evaluate(&data, &target);
        let val = tree.evaluate(&data, &target);
        let val = tree.evaluate(&data, &target);
        println!("Correct percentage: {:?}", val);
    }
}
