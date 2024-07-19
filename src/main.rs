#![allow(unused)]
use core::hash::Hash;
use core::ops::Add;
//use core::slice::SlicePattern;
use ndarray::{prelude::*, ViewRepr};
use ndarray_npy::read_npy;
use num::{Float, FromPrimitive, NumCast};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::{f64::INFINITY, marker::PhantomData};
fn main() {
    // load data and target
    //const dim :Dim<[usize; 2]> = Dim([1797,64]);
    let data: Array2<f64> = read_npy("src/digits_data.npy").expect("file is present and correct");
    let target: Array1<i64> =
        read_npy("src/digits_target.npy").expect("file is present and correct");

    println!("{:?}", data.dim());
    println!("{:?}", target.dim());
}
// T is the type of the prediction value
//G is type of feature data Vec<>
enum Node<T: Clone, G: Clone> {
    SplitNode {
        left: Option<Box<Node<T, G>>>,
        right: Option<Box<Node<T, G>>>,
        split_index: usize,
        threashold: G,
    },
    LeafNode {
        prediction: T,
    },
}

//G is type of feature data Vec<>
struct RegressionTree<T: Clone, G: Clone>
where
    f64: From<G>,
    G: From<f64> + FromPrimitive + Float + From<usize>,
    T: std::convert::From<f64>,
{
    root: Option<Box<Node<T, G>>>,
    n_min: usize,
    //input_data: PhantomData<G>,
}

impl<T: Clone, G: Clone + std::cmp::PartialOrd> RegressionTree<T, G>
where
    f64: From<G>,
    G: Add
        + From<f64>
        + Hash
        + FromPrimitive
        + Float
        + From<usize>
        + From<T>
        + Ord
        + ndarray::ScalarOperand
        + std::ops::Add<Output = G>
        + core::ops::Add
        + Copy,

    T: std::convert::From<f64>,
{
    fn predict(&self, x: Vec<G>) -> T {
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
                    return T::from(0_f64); // if not trained yet, return 0 always
                                           // if trained but still reached => bug
                                           // consider logging here, as it does not make sense to use predict before training
                }
            }
        }

        todo!("predict followed split node for over 100 splits! not possible?");
    }
    /// j is feature index and t is threashold
    fn compute_loss_for_split(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
        j: usize,
        t: G,
    ) -> G {
        //let arr = Array1::from(features[0].clone());
        let temp = features.slice(s![.., j]);
        let left_indices = temp.mapv(|x: G| (x <= t) as usize);
        let right_indices = left_indices.mapv(|x| 1 - x);
        let amount_left = (left_indices.sum());
        let amount_right = (right_indices.sum());
        if (amount_left < self.n_min) | (amount_right < self.n_min) {
            return <G as From<f64>>::from(INFINITY);
        }
        let left = responses
            .select(Axis(1), left_indices.as_slice().expect("idk if this works"))
            .mapv(|x| <G as From<T>>::from(x));

        let right = responses
            .select(
                Axis(1),
                right_indices.as_slice().expect("idk if this works"),
            )
            .mapv(|x| <G as From<T>>::from(x));
        let zero = <G as From<f64>>::from(0_f64);
        let loss = left.var(zero) * <G as From<usize>>::from(amount_left)
            + right.var(zero) * <G as From<usize>>::from(amount_right);
        return loss;
    }

    /// return a 1-D array with D_try randomly selected indices from 0...(D-1).
    fn select_active_indices(D: usize, d_try: usize) -> Vec<usize> {
        let mut rng = ndarray_rand::rand::thread_rng();
        let mut myvec: Vec<usize> = (0..D).collect();
        myvec.shuffle(&mut rng);
        myvec[..d_try].to_vec()
    }
    fn find_threasholds(&self, features: &Array2<G>, responses: &Array1<T>, j: usize) -> Array1<G> {
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
        let mut new = unique_feature_array.add(temp2);
        let new = new / <G as From<usize>>::from(2);
        return new;
    }

    fn train_node(
        &mut self,
        node: &mut Node<T, G>,
        features: &Array2<G>,
        responses: &Array1<T>,
        d_try: Option<usize>,
    ) {
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let d_try = d_try.unwrap_or(8);

        let working_node = node;

        let mut l_min = <G as From<f64>>::from(INFINITY);
        let mut j_min = 0;
        let mut t_min = <G as From<f64>>::from(0_f64);
        loop {
            l_min = <G as From<f64>>::from(INFINITY);
            j_min = 0;
            t_min = <G as From<f64>>::from(0_f64);
            let active_indices = RegressionTree::<T, G>::select_active_indices(D, d_try);
            for j in active_indices {
                let threasholds =
                    RegressionTree::<T, G>::find_threasholds(&self, &features, responses, j);
                for threashold in threasholds {
                    let loss = RegressionTree::<T, G>::compute_loss_for_split(
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
                // append leaf node
                //root
            }
        }

        todo!()
    }
    fn train(&mut self, features: &Array2<G>, responses: &Array1<T>, d_try: Option<usize>) {
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let d_try = d_try.unwrap_or(8);

        let working_node = &mut self.root;

        let mut l_min = <G as From<f64>>::from(INFINITY);
        let mut j_min = 0;
        let mut t_min = <G as From<f64>>::from(0_f64);

        l_min = <G as From<f64>>::from(INFINITY);
        j_min = 0;
        t_min = <G as From<f64>>::from(0_f64);
        let active_indices = RegressionTree::<T, G>::select_active_indices(D, d_try);
        for j in active_indices {
            let threasholds =
                RegressionTree::<T, G>::find_threasholds(&self, &features, responses, j);
            for threashold in threasholds {
                let loss = RegressionTree::<T, G>::compute_loss_for_split(
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
            // append leaf node
            //root
            let node = Node::LeafNode { prediction: responses.mean() }
        }
        // ELSE : assume a split has been found.
        // Make first Node:
        let node = Node::SplitNode { left: None, right: None, split_index: j_min, threashold: t_min };
        self.root = Some(Box::new(node));

        todo!()
    }
}
