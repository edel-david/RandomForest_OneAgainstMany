use core::hash::Hash;
use core::ops::Add;
use ndarray::ScalarOperand;
use num::{Float, FromPrimitive, NumCast, One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::Error;
use std::hash::Hasher;
use std::ops::{Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;
use std::usize;

/// Note: This file was laregely created by Coding assistants, be cautious

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Float32(pub f32);

impl conv::ValueFrom<usize> for Float32 {
    type Err = Error;
    fn value_from(src: usize) -> Result<Self, Self::Err> {
        Ok(Self(src as f32))
    }
}

impl Into<i32> for Float32 {
    fn into(self) -> i32 {
        self.0 as i32
    }
}
impl From<i8> for Float32 {
    fn from(value: i8) -> Self {
        Self(value as f32)
    }
}

impl Signed for Float32 {
    fn abs(&self) -> Self {
        Self(self.0.abs())
    }
    fn abs_sub(&self, other: &Self) -> Self {
        Self((self.0 - other.0).max(0.0))
    }
    fn is_negative(&self) -> bool {
        self.0.is_sign_negative()
    }
    fn is_positive(&self) -> bool {
        self.0.is_sign_positive()
    }
    fn signum(&self) -> Self {
        Self(self.0.signum())
    }
}

impl From<i32> for Float32 {
    fn from(value: i32) -> Self {
        Self(value as f32)
    }
}

impl ScalarOperand for Float32 {}
impl Hash for Float32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<usize> for Float32 {
    fn from(value: usize) -> Self {
        Self(value as f32)
    }
}

impl From<i64> for Float32 {
    fn from(value: i64) -> Self {
        Self(value as f32)
    }
}

impl Add for Float32 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Float32(self.0 + other.0)
    }
}

impl Sub for Float32 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Float32(self.0 - other.0)
    }
}

impl Mul for Float32 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Float32(self.0 * other.0)
    }
}

impl Div for Float32 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Float32(self.0 / other.0)
    }
}

impl Zero for Float32 {
    fn zero() -> Self {
        Float32(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for Float32 {
    fn one() -> Self {
        Float32(1.0)
    }
}
impl Neg for Float32 {
    type Output = Float32;
    fn neg(self) -> Self::Output {
        Float32(self.0.neg())
    }
}
impl Rem for Float32 {
    type Output = Float32;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}

impl FromStr for Float32 {
    type Err = <f32 as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<f32>().map(Float32)
    }
}

impl num::Num for Float32 {
    type FromStrRadixErr = <f32 as num::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(Float32)
    }
}
impl NumCast for Float32 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(Float32)
    }
}

impl Float for Float32 {
    fn nan() -> Self {
        Float32(f32::NAN)
    }

    fn infinity() -> Self {
        Float32(f32::INFINITY)
    }

    fn neg_infinity() -> Self {
        Float32(f32::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Float32(-0.0)
    }

    fn min_value() -> Self {
        Float32(f32::MIN)
    }

    fn min_positive_value() -> Self {
        Float32(f32::MIN_POSITIVE)
    }

    fn epsilon() -> Self {
        Float32(f32::EPSILON)
    }

    fn max_value() -> Self {
        Float32(f32::MAX)
    }

    fn classify(self) -> std::num::FpCategory {
        self.0.classify()
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn floor(self) -> Self {
        Float32(self.0.floor())
    }

    fn ceil(self) -> Self {
        Float32(self.0.ceil())
    }

    fn round(self) -> Self {
        Float32(self.0.round())
    }

    fn trunc(self) -> Self {
        Float32(self.0.trunc())
    }

    fn fract(self) -> Self {
        Float32(self.0.fract())
    }

    fn abs(self) -> Self {
        Float32(self.0.abs())
    }

    fn signum(self) -> Self {
        Float32(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Float32(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        Float32(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        Float32(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Float32(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        Float32(self.0.sqrt())
    }

    fn exp(self) -> Self {
        Float32(self.0.exp())
    }

    fn exp2(self) -> Self {
        Float32(self.0.exp2())
    }

    fn ln(self) -> Self {
        Float32(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        Float32(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        Float32(self.0.log2())
    }

    fn log10(self) -> Self {
        Float32(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        Float32(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Float32(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self((self.0 - other.0).abs())
    }

    fn cbrt(self) -> Self {
        Float32(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Float32(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        Float32(self.0.sin())
    }

    fn cos(self) -> Self {
        Float32(self.0.cos())
    }

    fn tan(self) -> Self {
        Float32(self.0.tan())
    }

    fn asin(self) -> Self {
        Float32(self.0.asin())
    }

    fn acos(self) -> Self {
        Float32(self.0.acos())
    }

    fn atan(self) -> Self {
        Float32(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        Float32(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (Float32(sin), Float32(cos))
    }

    fn exp_m1(self) -> Self {
        Float32(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        Float32(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        Float32(self.0.sinh())
    }

    fn cosh(self) -> Self {
        Float32(self.0.cosh())
    }

    fn tanh(self) -> Self {
        Float32(self.0.tanh())
    }

    fn asinh(self) -> Self {
        Float32(self.0.asinh())
    }

    fn acosh(self) -> Self {
        Float32(self.0.acosh())
    }

    fn atanh(self) -> Self {
        Float32(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }

    fn to_degrees(self) -> Self {
        Float32(self.0.to_degrees())
    }

    fn to_radians(self) -> Self {
        Float32(self.0.to_radians())
    }
}

impl Eq for Float32 {}
impl Ord for Float32 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Define a custom ordering for NaN values
        if self.0.is_nan() && other.0.is_nan() {
            Ordering::Equal
        } else if self.0.is_nan() {
            Ordering::Less
        } else if other.0.is_nan() {
            Ordering::Greater
        } else {
            self.0.partial_cmp(&other.0).unwrap()
        }
    }
}
impl From<f32> for Float32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}
impl FromPrimitive for Float32 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Float32(n))
    }

    fn from_isize(n: isize) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_i8(n: i8) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_i16(n: i16) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_i32(n: i32) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_usize(n: usize) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_u8(n: u8) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_u16(n: u16) -> Option<Self> {
        Some(Float32(n as f32))
    }

    fn from_u32(n: u32) -> Option<Self> {
        Some(Float32(n as f32))
    }
}

impl ToPrimitive for Float32 {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0)
    }

    fn to_isize(&self) -> Option<isize> {
        Some(self.0 as isize)
    }

    fn to_i8(&self) -> Option<i8> {
        Some(self.0 as i8)
    }

    fn to_i16(&self) -> Option<i16> {
        Some(self.0 as i16)
    }

    fn to_i32(&self) -> Option<i32> {
        Some(self.0 as i32)
    }

    fn to_usize(&self) -> Option<usize> {
        Some(self.0 as usize)
    }

    fn to_u8(&self) -> Option<u8> {
        Some(self.0 as u8)
    }

    fn to_u16(&self) -> Option<u16> {
        Some(self.0 as u16)
    }

    fn to_u32(&self) -> Option<u32> {
        Some(self.0 as u32)
    }
}
impl From<u8> for Float32 {
    fn from(value: u8) -> Self {
        Self(value as f32)
    }
}
