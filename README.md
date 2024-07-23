# My implementation of A One-Against-Many ML alorithm using Random Forest

## crate name: supa_cube_forest

### by edel-david on Github

### notable rust crates

- ndarray (+ _stat,_rand,_npy)
- rayon

use the tests (run with cargo test) to learn what this does
you can generate the necesarry data npy files with the provided export_dataset.ipynb notebook.

the standard mnist-digits dataset test is enabled by default

a mnist fashion test can be enabled, but it takes very long (few minutes) and may take more than 16 gigs of ram.

The algorithm was descibed in A ML lecture at my University.

It was a homework to implement it in python.

So naturally ,like any algorithm, I ported it to rust.

