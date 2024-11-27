use core::convert::TryInto;

use crate::{
    biginteger::{
        arithmetic as fa, BigInteger as _BigInteger, BigInteger256,
    },
    bytes::{FromBytes, ToBytes},
    fields::{FftField, Field, LegendreSymbol, PrimeField, SquareRootField},
};
use ark_serialize::*;
use ark_std::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Display, Formatter, Result as FmtResult},
    io::{Read, Result as IoResult, Write},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr, One, Zero,
};

pub trait FieldConstants: ark_std::fmt::Debug + Clone + Copy + Default + Eq + PartialEq + PartialOrd + Ord + core::hash::Hash + 'static + Send + Sync + Sized
// trait FieldConstants: #[derive(Debug, Clone, Copy, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
 {
    const U32_MODULUS: [u32; 9];
    const U64_MODULUS: [u64; 9];
    const U32_R: [u32; 9];
    const U32_R2: [u32; 9];
    const U64_MINV: u64;
    const U32_MINV: u32;
    const REPR_SHAVE_BITS: u32;
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct FpConstant;
impl FieldConstants for FpConstant {
    const U32_MODULUS: [u32; 9] = [
        0x1, 0x9698768, 0x133e46e6, 0xd31f812, 0x224, 0x0, 0x0, 0x0, 0x400000,
    ];
    const U64_MODULUS: [u64; 9] = {
        let mut modulus64 = [0u64; 9];
        let modulus = Self::U32_MODULUS;
        let mut i = 0;
        while i < 9 {
            modulus64[i] = modulus[i] as u64;
            i += 1;
        }
        modulus64
    };
    const U32_R: [u32; 9] = [
        0x1fffff81, 0x14a5d367, 0x141ad3c0, 0x1435eec5, 0x1ffeefef, 0x1fffffff, 0x1fffffff,
        0x1fffffff, 0x3fffff,
    ];
    const U32_R2: [u32; 9] = [
        0x3b6a, 0x19c10910, 0x1a6a0188, 0x12a4fd88, 0x634b36d, 0x178792ba, 0x7797a99, 0x1dce5b8a,
        0x3506bd,
    ];
    // const U32_MINV: u64 = 0x1fffffff;
    const U64_MINV: u64 = 0x1fffffff;
    const U32_MINV: u32 = 0x1fffffff;
    const REPR_SHAVE_BITS: u32 = 1;
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct FqConstant;
impl FieldConstants for FqConstant {
    const U32_MODULUS: [u32; 9] = [
        0x1, 0x2375908, 0x52a3763, 0xd31f813, 0x224, 0x0, 0x0, 0x0, 0x400000,
    ];
    const U64_MODULUS: [u64; 9] = {
        let mut modulus64 = [0u64; 9];
        let modulus = Self::U32_MODULUS;
        let mut i = 0;
        while i < 9 {
            modulus64[i] = modulus[i] as u64;
            i += 1;
        }
        modulus64
    };
    const U32_R: [u32; 9] = [
        0x1fffff81, 0x68ad507, 0x100e85da, 0x1435ee7e, 0x1ffeefef, 0x1fffffff, 0x1fffffff,
        0x1fffffff, 0x3fffff,
    ];
    const U32_R2: [u32; 9] = [
        0x3b6a, 0x2b1b550, 0x1027888a, 0x1ea4ed96, 0x418ad7a, 0x999eb, 0x17fae231,
        0x1e67ed54, 0x3506bd,
    ];
    // const U32_MINV: u64 = 0x1fffffff;
    const U64_MINV: u64 = 0x1fffffff;
    const U32_MINV: u32 = 0x1fffffff;
    const REPR_SHAVE_BITS: u32 = 1;
}

impl<C: Fp256Parameters> Into<BigInteger256> for NewFp256<C> {
    fn into(self) -> BigInteger256 {
        self.into_repr()
    }
}
impl<C: Fp256Parameters> core::convert::TryFrom<BigInteger256> for NewFp256<C> {
    type Error = crate::fields::arithmetic::InvalidBigInt;

    /// Converts `Self::BigInteger` into `Self`
    ///
    /// This method returns an error if `int` is larger than `P::MODULUS`.
    fn try_from(int: BigInteger256) -> Result<Self, Self::Error> {
        Self::from_repr(int).ok_or(crate::fields::arithmetic::InvalidBigInt)
    }
}

const SHIFT: u32 = 29;
const MASK: u32 = (1 << SHIFT) - 1;

const SHIFT64: u64 = SHIFT as u64;
const MASK64: u64 = MASK as u64;

pub const fn from_64x4(pa: [u64; 4]) -> [u32; 9] {
    let mut p = [0u32; 9];
    p[0] = (pa[0] & MASK64) as u32;
    p[1] = ((pa[0] >> 29) & MASK64) as u32;
    p[2] = (((pa[0] >> 58) | (pa[1] << 6)) & MASK64) as u32;
    p[3] = ((pa[1] >> 23) & MASK64) as u32;
    p[4] = (((pa[1] >> 52) | (pa[2] << 12)) & MASK64) as u32;
    p[5] = ((pa[2] >> 17) & MASK64) as u32;
    p[6] = (((pa[2] >> 46) | (pa[3] << 18)) & MASK64) as u32;
    p[7] = ((pa[3] >> 11) & MASK64) as u32;
    p[8] = (pa[3] >> 40) as u32;
    p
}
pub const fn to_64x4(pa: [u32; 9]) -> [u64; 4] {
    let mut p = [0u64; 4];
    p[0] = pa[0] as u64;
    p[0] |= (pa[1] as u64) << 29;
    p[0] |= (pa[2] as u64) << 58;
    p[1] = (pa[2] as u64) >> 6;
    p[1] |= (pa[3] as u64) << 23;
    p[1] |= (pa[4] as u64) << 52;
    p[2] = (pa[4] as u64) >> 12;
    p[2] |= (pa[5] as u64) << 17;
    p[2] |= (pa[6] as u64) << 46;
    p[3] = (pa[6] as u64) >> 18;
    p[3] |= (pa[7] as u64) << 11;
    p[3] |= (pa[8] as u64) << 40;
    p
}

// const U32_MODULUS: [u32; 9] = [
//     0x1, 0x9698768, 0x133e46e6, 0xd31f812, 0x224, 0x0, 0x0, 0x0, 0x400000,
// ];
// const U64_MODULUS: [u64; 9] = {
//     let mut modulus64 = [0u64; 9];
//     let modulus = U32_MODULUS;
//     let mut i = 0;
//     while i < 9 {
//         modulus64[i] = modulus[i] as u64;
//         i += 1;
//     }
//     modulus64
// };
// const U32_R: [u32; 9] = [
//     0x1fffff81, 0x14a5d367, 0x141ad3c0, 0x1435eec5, 0x1ffeefef, 0x1fffffff, 0x1fffffff,
//     0x1fffffff, 0x3fffff,
// ];
// const U32_R2: [u32; 9] = [
//     0x3b6a, 0x19c10910, 0x1a6a0188, 0x12a4fd88, 0x634b36d, 0x178792ba, 0x7797a99, 0x1dce5b8a,
//     0x3506bd,
// ];
// // const U32_MINV: u64 = 0x1fffffff;
// const U64_MINV: u64 = 0x1fffffff;
// const U32_MINV: u32 = 0x1fffffff;

// const REPR_SHAVE_BITS: u32 = 1;

// #[inline]
const fn gte_modulus<C: Fp256Parameters>(x: &Inner) -> bool {
    // dbg!(x, U32_MODULUS);
    let mut i = NewFp256::<C>::NLIMBS - 1;
    loop {
    // for i in (0..9).rev() {
        // eprintln!("gte_modulus2={:?} x[i]={:?} U32_MODULUS[i]={:?}", i, x[i], U32_MODULUS[i]);
        // don't fix warning -- that makes it 15% slower!
        #[allow(clippy::comparison_chain)]
        if x.0[i] > C::MODULUS.0[i] {
            return true;
        } else if x.0[i] < C::MODULUS.0[i] {
            return false;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    true
}

const fn conditional_reduce<C: Fp256Parameters>(mut x: Inner) -> Inner {
    if gte_modulus::<C>(&x) {
        let mut i = 0;
        while i < 9 {
        // for i in 0..9 {
            x.0[i] = x.0[i].wrapping_sub(C::MODULUS.0[i]);
            i += 1;
        }
        let mut i = 1;
        while i < 9 {
        // for i in 1..9 {
            x.0[i] += ((x.0[i - 1] as i32) >> SHIFT) as u32;
            i += 1;
        }
        let mut i = 0;
        while i < 8 {
        // for i in 0..8 {
            x.0[i] &= MASK;
            i += 1;
        }
    }
    x
}

fn add_assign<C: Fp256Parameters>(x: &mut Inner, y: &Inner) {
    // let x = &mut x.0;
    let y = &y.0;
    let mut tmp: u32;
    let mut carry: i32 = 0;

    for i in 0..9 {
        tmp = x.0[i] + y[i] + (carry as u32);
        carry = (tmp as i32) >> SHIFT;
        x.0[i] = tmp & MASK;
    }

    if gte_modulus::<C>(x) {
        carry = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..9 {
            tmp = x.0[i].wrapping_sub(C::MODULUS.0[i]) + (carry as u32);
            carry = (tmp as i32) >> SHIFT;
            x.0[i] = tmp & MASK;
        }
    }
}

// #[derive(Clone, Copy, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
// struct Inner([u32; 9]);
type Inner = BigInteger256;

#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct NewFp256<C: Fp256Parameters> (pub Inner, PhantomData<C>);

/// Note that this implementation of `Ord` compares field elements viewing
/// them as integers in the range 0, 1, ..., P::MODULUS - 1. However, other
/// implementations of `PrimeField` might choose a different ordering, and
/// as such, users should use this `Ord` for applications where
/// any ordering suffices (like in a BTreeMap), and not in applications
/// where a particular ordering is required.
impl<P: Fp256Parameters> Ord for NewFp256<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_repr().cmp(&other.into_repr())
    }
}
/// Note that this implementation of `PartialOrd` compares field elements viewing
/// them as integers in the range 0, 1, ..., `P::MODULUS` - 1. However, other
/// implementations of `PrimeField` might choose a different ordering, and
/// as such, users should use this `PartialOrd` for applications where
/// any ordering suffices (like in a BTreeMap), and not in applications
/// where a particular ordering is required.
impl<P: Fp256Parameters> PartialOrd for NewFp256<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Fp256Parameters> Display for NewFp256<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("{:?}", self.0))
    }
}

impl<C: Fp256Parameters> ark_std::fmt::Debug for NewFp256<C> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        use crate::ark_std::string::ToString;
        let r: BigInteger256 = self.into_repr();
        let bigint: num_bigint::BigUint = r.into();
        let s = bigint.to_string();

        let name = match C::T.0[0] {
            0x192d30ed => "Fp",
            0xc46eb21 => "Fq",
            _ => panic!(),
        };

        f.write_fmt(format_args!("{}({})", name, s))
    }
}

impl<C: Fp256Parameters> NewFp256<C> {
    const NLIMBS: usize = 9;

    #[inline]
    pub const fn new(element: Inner) -> Self {
        Self(element, PhantomData)
        // let BigInteger256(bigint) = element;
        // Self(from_64x4(bigint))
    }
    const fn const_is_zero(&self) -> bool {
        let mut index = 0;
        let mut is_zero = true;
        while index < Self::NLIMBS {
            is_zero &= self.0.0[index] == 0;
            index += 1;
        }
        is_zero
    }
    const fn const_neg(self, modulus: Inner) -> Self {
        if !self.const_is_zero() {
            Self::new(Self::sub_noborrow(&modulus, &self.0))
        } else {
            self
        }
    }

    #[allow(unused_assignments)]
    const fn sub_noborrow(a: &Inner, b: &Inner) -> Inner {
        /// Calculate a - b - borrow, returning the result and modifying
        /// the borrow value.
        macro_rules! sbb {
            ($a:expr, $b:expr, &mut $borrow:expr$(,)?) => {{
                let tmp = (1u64 << 32) + ($a as u64) - ($b as u64) - ($borrow as u64);
                $borrow = if tmp >> 32 == 0 { 1 } else { 0 };
                tmp as u32
            }};
        }
        let mut a = *a;
        let mut borrow = 0;
        let mut i = 0;
        while i < Self::NLIMBS{
            a.0[i] = sbb!(a.0[i], b.0[i], &mut borrow);
            i += 1;
        }
        a
    }

    /// Interpret a string of decimal numbers as a prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    /// For *internal* use only; please use the `field_new` macro instead
    /// of this method
    #[doc(hidden)]
    pub const fn const_from_str(
        limbs: &[u64],
        is_positive: bool,
        r2: Inner,
        modulus: Inner,
        inv: u64,
    ) -> Self {
        let repr = match limbs {
            [a, b, c, d] => BigInteger256::from_64x4([*a, *b, *c, *d]),
            [a, b, c] => BigInteger256::from_64x4([*a, *b, *c, 0]),
            [a, b] => BigInteger256::from_64x4([*a, *b, 0, 0]),
            [a] => BigInteger256::from_64x4([*a, 0, 0, 0]),
            _ => BigInteger256::from_64x4([0, 0, 0, 0])
        };
        let res = Self::const_from_repr(repr, r2, modulus, inv as u32);
        if is_positive {
            res
        } else {
            res.const_neg(modulus)
        }
    }

    // pub const fn const_from_str(
    //     limbs: &[u32],
    //     is_positive: bool,
    //     r2: Inner,
    //     modulus: Inner,
    //     inv: u32,
    // ) -> Self {
    //     let mut repr = BigInteger256([0; 9]);
    //     let mut i = 0;
    //     while i < limbs.len() {
    //         repr.0[i] = limbs[i];
    //         i += 1;
    //     }
    //     let res = Self::const_from_repr(repr, r2, modulus, inv);
    //     if is_positive {
    //         res
    //     } else {
    //         res.const_neg(modulus)
    //     }
    // }

    #[inline]
    pub(crate) const fn const_from_repr(
        repr: Inner,
        r2: Inner,
        modulus: Inner,
        inv: u32,
    ) -> Self {
        let mut r = Self::new(repr);
        if r.const_is_zero() {
            r
        } else {
            r = r.const_mul(&NewFp256(r2, PhantomData), &modulus, inv);
            r
        }
    }

    const U64_MODULUS: [u64; 9] = {
        let mut modulus64 = [0u64; 9];
        let modulus = C::MODULUS;
        let mut i = 0;
        while i < 9 {
            modulus64[i] = modulus.0[i] as u64;
            i += 1;
        }
        modulus64
    };

    const fn mul_without_reduce(self, other: &Self, _modulus: &Inner, _inv: u32) -> Self {

        // const N: usize = 9;

        // how much terms we can add before a carry
        let n_safe_terms = 2u64.pow(64 - 2 * SHIFT) as usize;
        // how much j steps we can do before a carry:
        let n_safe_steps = 2u64.pow(64 - 2 * SHIFT - 1) as usize;

        let mut x = self.0.0;
        let y = &other.0.0;
        // let y_local = other.0.0;

        // load y[i] into local u64s
        // TODO make sure these are locals
        let mut y_local = [0u64; Self::NLIMBS];
        let mut index = 0;
        while index < Self::NLIMBS {
        // for i in 0..9 {
            y_local[index] = y[index] as u64;
            index += 1;
        }

        // assert_eq!(y_local.to_vec(), y.iter().map(|v| *v as u64).collect::<Vec<_>>());

        // locals for result
        // let mut z = [0u64; 8];
        // let mut tmp: u64;

        // dbg!(y_local);

        let mut xy = [0u64; 9];

        // main loop, without intermediate carries except for z0
        // #[allow(clippy::needless_range_loop)]
        let mut i = 0;
        while i < Self::NLIMBS {
        // for i in 0..9 {
            // eprintln!("\n### I={:?} ###", i);

            let xi = x[i] as u64;

            // compute qi and carry z0 result to z1 before discarding z0
            // tmp = xi * y_local[0];
            // let qi = ((tmp & MASK64) * U32_MINV) & MASK64;
            // z[1] += (tmp + qi * U64_MODULUS[0]) >> SHIFT64;
            let tmp = (xi * y_local[0]) + xy[0];
            let qi = (MASK64 + 1) - (tmp & MASK64);
            // let qi2 = ((tmp & MASK64) * U32_MINV) & MASK64;
            // dbg!(qi);
            // assert_eq!(qi, qi2);

            // let carry = add_mul(tmp, qi, U64_MODULUS[0]) >> SHIFT64;
            let carry = (tmp + (qi * Self::U64_MODULUS[0])) >> SHIFT64;

            // // j=0, compute q_i
            // let j = 0;
            // // XY[0] + x[i]*y[0]
            // i64.mul(xi, Y[j]);
            // i64.add($, XY[j]);
            // // qi = (($ & wordMax) * mu) & wordMax
            // local.set(tmp);
            // local.set(qi, computeQ(tmp));
            // local.get(tmp);
            // // (stack, _) = $ + qi*p[0]
            // addMul(qi, P[j]);
            // i64.shr_u($, wn); // we just put carry on the stack, use it later

            // function computeQ(x: Input<i64>) {
            //   // q = ((x & wordMax) * mu) & wordMax, where wordMax = 2^w - 1
            //   x = i64.and(x, wordMax);
            //   if (mu === wordMax) {
            //     // special case relevant for high 2-adicity curves: mu = 2^w - 1
            //     // (mu * x) % 2^w = -x % 2^w  = 2^w - x
            //     return i64.sub(wordMax + 1n, x);
            //   } else {
            //     return i64.and(i64.mul(x, mu), wordMax);
            //   }
            // }

            let mut j = 1;
            while j < Self::NLIMBS - 1 {
            // for j in 1..N - 1 {
                let did_carry = ((j - 1) % n_safe_steps) == 0;
                // let do_carry = (j % n_safe_steps) == 0;

                // assert!(!do_carry);
                // dbg!(did_carry, do_carry);
                let mut xy_j = xy[j];
                if did_carry {
                    xy_j += carry;
                }
                // xy[j - 1] = add_mul((xi * y_local[j]) + xy_j, qi, U64_MODULUS[j]);
                xy[j - 1] = (xy_j + (xi * y_local[j])) + (qi * Self::U64_MODULUS[j]);
                // xy[j - 1] = (xy_j + (xi * y_local[j])) + (qi * U64_MODULUS[j]);
                j += 1;
            }

            // for (j = 1; j < n - 1; j++) {
            //   // XY[j] + x[i]*y[j] + qi*p[j], or
            //   // stack + XY[j] + x[i]*y[j] + qi*p[j]
            //   // ... = XY[j-1], or  = (stack, XY[j-1])
            //   let didCarry = (j - 1) % nSafeSteps === 0;
            //   let doCarry = j % nSafeSteps === 0;
            //   local.get(XY[j]);
            //   Field.optionalCarryAdd(didCarry);
            //   i64.mul(xi, Y[j]);
            //   i64.add();
            //   addMul(qi, P[j]);
            //   Field.optionalCarry(doCarry, $, tmp);
            //   local.set(XY[j - 1]);
            // }

            // dbg!(xy);
            let j = Self::NLIMBS - 1;
            // let did_carry = ((j - 1) % n_safe_steps) == 0;
            let do_carry = (j % n_safe_steps) == 0;

            // assert!(!did_carry);
            // dbg!(did_carry, do_carry);

            if do_carry {
                todo!();
            } else {
                // xy[j - 1] = add_mul(xi * y_local[j], qi, U64_MODULUS[j]);
                xy[j - 1] = (xi * y_local[j]) + (qi * Self::U64_MODULUS[j]);
                // xy[j - 1] = (xi * y_local[j]) + (qi * U64_MODULUS[j]);
            }

            // // if the last iteration doesn't do a carry, then XY[n-1] is never set,
            // // so we also don't have to get it & can save 1 addition
            // i64.mul(xi, Y[j]);
            // Field.optionalCarryAdd(didCarry);
            // addMul(qi, P[j]);
            // local.set(XY[j - 1]);

            // dbg!(xy);

            i += 1;
        }

        // dbg!(xy);

        let mut j = 1;
        while j < Self::NLIMBS {
        // for j in 1..N {
            x[j - 1] = (xy[j - 1] as u32) & MASK;
            xy[j] += xy[j - 1] >> SHIFT64;
            j += 1;
        }
        x[Self::NLIMBS - 1] = xy[Self::NLIMBS - 1] as u32;

        Self(BigInteger256(x), PhantomData)
    }

    const fn const_mul(mut self, other: &Self, modulus: &Inner, inv: u32) -> Self {
        self = self.mul_without_reduce(other, modulus, inv);
        self.const_reduce(modulus)
    }

    const fn const_reduce(self, _modulus: &Inner) -> Self {
        let reduced = conditional_reduce::<C>(self.0);
        Self(reduced, PhantomData)
        // if !self.const_is_valid(modulus) {
        //     self.0 = Self::sub_noborrow(&self.0, &modulus);
        // }
    }

    // don't fix warning -- that makes it 15% slower!
    #[allow(clippy::comparison_chain)]
    const fn const_is_valid(&self, _modulus: &Inner) -> bool {
        let mut i = NewFp256::<C>::NLIMBS - 1;
        loop {
            if self.0.0[i] > C::MODULUS.0[i] {
                return false;
            } else if self.0.0[i] < C::MODULUS.0[i] {
                return true;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        false
    }
}

impl<C: Fp256Parameters> NewFp256<C> {
    pub(crate) fn is_valid(&self) -> bool {
        self.const_is_valid(&C::MODULUS)
    }
    fn reduce(&mut self) {
        *self = self.const_reduce(&C::MODULUS);
        // if !self.is_valid() {
        //     self.0.sub_noborrow(&P::MODULUS);
        // }
    }
}

impl<C: Fp256Parameters> Zero for NewFp256<C> {
    fn zero() -> Self {
        Self(BigInteger256([0; 9]), PhantomData)
    }
    fn is_zero(&self) -> bool {
        self.0.0 == [0u32; 9]
    }
}

impl<C: Fp256Parameters> One for NewFp256<C> {
    fn one() -> Self {
        Self(C::R, PhantomData)
    }
    fn is_one(&self) -> bool {
        self.0 == C::R
    }
}

impl<C: Fp256Parameters> Neg for NewFp256<C> {
    type Output = Self;
    #[must_use]
    fn neg(self) -> Self {
        if !self.is_zero() {
            let mut tmp = C::MODULUS;
            tmp.sub_noborrow(&self.0);
            NewFp256(tmp, PhantomData)
        } else {
            self
        }
    }
}
impl<C: Fp256Parameters> core::ops::DivAssign<Self> for NewFp256<C> {
    fn div_assign(&mut self, other: Self) {
        self.div_assign(&other)
    }
}
impl<C: Fp256Parameters> Add<Self> for NewFp256<C> {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self.add_assign(other);
        self
    }
}
impl<C: Fp256Parameters> Sub<Self> for NewFp256<C> {
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        self.sub_assign(other);
        self
    }
}
impl<C: Fp256Parameters> Div<Self> for NewFp256<C> {
    type Output = Self;
    fn div(mut self, other: Self) -> Self {
        self.div_assign(other);
        self
    }
}
impl<C: Fp256Parameters> core::ops::AddAssign<Self> for NewFp256<C> {
    fn add_assign(&mut self, other: Self) {
        add_assign::<C>(&mut self.0, &other.0)
    }
}
impl<C: Fp256Parameters> Mul<Self> for NewFp256<C> {
    type Output = Self;
    fn mul(mut self, other: Self) -> Self {
        self.mul_assign(other);
        self
    }
}
impl<C: Fp256Parameters> core::ops::MulAssign<Self> for NewFp256<C> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.const_mul(&other, &C::MODULUS, C::INV as u32);
    }
}
impl<C: Fp256Parameters> SubAssign<Self> for NewFp256<C> {
    fn sub_assign(&mut self, other: Self) {
        if other.0 > self.0 {
            self.0.add_nocarry(&C::MODULUS);
        }
        self.0.sub_noborrow(&other.0);
    }
}
impl<C: Fp256Parameters> core::iter::Sum<Self> for NewFp256<C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}
impl<C: Fp256Parameters> core::iter::Product<Self> for NewFp256<C> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl<'a, C: Fp256Parameters> Div<&'a Self> for NewFp256<C> {
    type Output = Self;
    fn div(mut self, other: &'a Self) -> Self {
        self.div_assign(other);
        self
    }
}
impl<'a, C: Fp256Parameters> DivAssign<&'a Self> for NewFp256<C> {
    fn div_assign(&mut self, other: &'a Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}
impl<'a, C: Fp256Parameters> SubAssign<&'a Self> for NewFp256<C> {
    fn sub_assign(&mut self, other: &'a Self) {
        if other.0 > self.0 {
            self.0.add_nocarry(&C::MODULUS);
        }
        self.0.sub_noborrow(&other.0);
    }
}
impl<'a, C: Fp256Parameters> Sub<&'a Self> for NewFp256<C> {
    type Output = Self;
    fn sub(mut self, other: &'a Self) -> Self {
        self.sub_assign(other);
        self
    }
}
impl<'a, C: Fp256Parameters> core::iter::Product<&'a Self> for NewFp256<C> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}
impl<'a, C: Fp256Parameters> core::iter::Sum<&'a Self> for NewFp256<C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}
impl<'a, C: Fp256Parameters> Add<&'a Self> for NewFp256<C> {
    type Output = Self;
    fn add(mut self, other: &'a Self) -> Self {
        self.add_assign(other);
        self
    }
}
impl<'a, C: Fp256Parameters> core::ops::AddAssign<&'a Self> for NewFp256<C> {
    fn add_assign(&mut self, other: &'a Self) {
        add_assign::<C>(&mut self.0, &other.0)
    }
}
impl<'a, C: Fp256Parameters> Mul<&'a Self> for NewFp256<C> {
    type Output = Self;
    fn mul(mut self, other: &'a Self) -> Self {
        self.mul_assign(other);
        self
    }
}
impl<'a, C: Fp256Parameters> core::ops::MulAssign<&'a Self> for NewFp256<C> {
    fn mul_assign(&mut self, other: &'a Self) {
        *self = self.const_mul(&other, &C::MODULUS, C::INV as u32);
    }
}

impl<C: Fp256Parameters> From<u128> for NewFp256<C> {
    fn from(value: u128) -> Self {
        let hi = (value >> 64) as u64;
        let lo = value as u64;
        Self::from_repr(BigInteger256(from_64x4([lo, hi, 0, 0]))).unwrap()
    }
}
impl<C: Fp256Parameters> From<u64> for NewFp256<C> {
    fn from(value: u64) -> Self {
        Self::from_repr(BigInteger256::from_64x4([value, 0, 0, 0])).unwrap()
    }
}
impl<C: Fp256Parameters> From<u32> for NewFp256<C> {
    fn from(value: u32) -> Self {
        Self::from_repr(BigInteger256::from_64x4([value as u64, 0, 0, 0])).unwrap()
    }
}
impl<C: Fp256Parameters> From<i64> for NewFp256<C> {
    fn from(value: i64) -> Self {
        let abs = Self::from(value.unsigned_abs());
        if value.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<C: Fp256Parameters> From<i32> for NewFp256<C> {
    fn from(value: i32) -> Self {
        let abs = Self::from(value.unsigned_abs());
        if value.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<C: Fp256Parameters> From<u16> for NewFp256<C> {
    fn from(value: u16) -> Self {
        Self::from_repr(BigInteger256::from_64x4([value as u64, 0, 0, 0])).unwrap()
    }
}
impl<C: Fp256Parameters> From<u8> for NewFp256<C> {
    fn from(value: u8) -> Self {
        Self::from_repr(BigInteger256::from_64x4([value as u64, 0, 0, 0])).unwrap()
    }
}
impl<C: Fp256Parameters> From<bool> for NewFp256<C> {
    fn from(value: bool) -> Self {
        Self::from_repr(BigInteger256::from_64x4([value as u64, 0, 0, 0])).unwrap()
    }
}

impl<C: Fp256Parameters> CanonicalSerializeWithFlags for NewFp256<C> {
    fn serialize_with_flags<W: ark_std::io::Write, F: Flags>(
        &self,
        mut writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let output_byte_size = buffer_byte_size(C::MODULUS_BITS as usize + F::BIT_SIZE);
        let mut bytes = [0u8; 4 * 8 + 1];
        self.write(&mut bytes[..4 * 8])?;
        bytes[output_byte_size - 1] |= flags.u8_bitmask();
        writer.write_all(&bytes[..output_byte_size])?;
        Ok(())
    }
    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        todo!()
        // buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE)
    }
}
impl<C: Fp256Parameters> CanonicalSerialize for NewFp256<C> {
    fn serialize<W: ark_std::io::Write>(&self, writer: W) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }
    fn serialized_size(&self) -> usize {
        self.serialized_size_with_flags::<EmptyFlags>()
    }
}
impl<C: Fp256Parameters> CanonicalDeserializeWithFlags for NewFp256<C> {
    fn deserialize_with_flags<R: ark_std::io::Read, F: Flags>(
        mut reader: R,
    ) -> Result<(Self, F), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let output_byte_size = buffer_byte_size(C::MODULUS_BITS as usize + F::BIT_SIZE);
        let mut masked_bytes = [0; 4 * 8 + 1];
        reader.read_exact(&mut masked_bytes[..output_byte_size])?;
        let flags = F::from_u8_remove_flags(&mut masked_bytes[output_byte_size - 1])
            .ok_or(SerializationError::UnexpectedFlags)?;
        Ok((Self::read(&masked_bytes[..])?, flags))
    }
}
impl<C: Fp256Parameters> CanonicalDeserialize for NewFp256<C> {
    fn deserialize<R: ark_std::io::Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}

// impl crate::FpParameters for NewFpParameters {
//     // 28948022309329048855892746252171976963363056481941560715954676764349967630337
//     const MODULUS: BigInteger256 = BigInteger256([
//         0x1, 0x9698768, 0x133e46e6, 0xd31f812, 0x224, 0x0, 0x0, 0x0, 0x400000,
//         // 0x992d30ed00000001,
//         // 0x224698fc094cf91b,
//         // 0x0,
//         // 0x4000000000000000,
//     ]);

//     const R: BigInteger256 = BigInteger256([
//         0x1fffff81, 0x14a5d367, 0x141ad3c0, 0x1435eec5, 0x1ffeefef, 0x1fffffff, 0x1fffffff,
//         0x1fffffff, 0x3fffff,
//         // 0x34786d38fffffffd,
//         // 0x992c350be41914ad,
//         // 0xffffffffffffffff,
//         // 0x3fffffffffffffff,
//     ]);

//     const R2: BigInteger256 = BigInteger256([
//         0x3b6a, 0x19c10910, 0x1a6a0188, 0x12a4fd88, 0x634b36d, 0x178792ba, 0x7797a99, 0x1dce5b8a,
//         0x3506bd,
//         // 0x8c78ecb30000000f,
//         // 0xd7d30dbd8b0de0e7,
//         // 0x7797a99bc3c95d18,
//         // 0x96d41af7b9cb714,
//     ]);

//     // TODO
//     const MODULUS_MINUS_ONE_DIV_TWO: BigInteger256 = BigInteger256([
//         0x0, 0x4b4c3b4, 0x99f2373, 0x698fc09, 0x112, 0x0, 0x0, 0x0, 0x200000
//         // 0xcc96987680000000,
//         // 0x11234c7e04a67c8d,
//         // 0x0,
//         // 0x2000000000000000,
//     ]);

//     // TODO
//     // T and T_MINUS_ONE_DIV_TWO, where MODULUS - 1 = 2^S * T
//     const T: BigInteger256 = BigInteger256([
//         0x192d30ed, 0xa67c8dc, 0x11a63f02, 0x44, 0x0, 0x0, 0x0, 0x80000, 0x0,
//         // 0x94cf91b992d30ed, 0x224698fc, 0x0, 0x40000000,
//     ]);

//     // TODO
//     const T_MINUS_ONE_DIV_TWO: BigInteger256 = BigInteger256([
//         0xc969876, 0x533e46e, 0x8d31f81, 0x22, 0x0, 0x0, 0x0, 0x40000, 0x0,
//         // 0x4a67c8dcc969876, 0x11234c7e, 0x0, 0x20000000,
//     ]);

//     // TODO
//     // GENERATOR = 5
//     const GENERATOR: BigInteger256 = BigInteger256([
//         536870273,
//         26674023,
//         395118378,
//         273009041,
//         536520476,
//         536870911,
//         536870911,
//         536870911,
//         4194303,
//         // 0xa1a55e68ffffffed,
//         // 0x74c2a54b4f4982f3,
//         // 0xfffffffffffffffd,
//         // 0x3fffffffffffffff,
//     ]);

//     const MODULUS_BITS: u32 = 255;

//     const CAPACITY: u32 = Self::MODULUS_BITS - 1;

//     const REPR_SHAVE_BITS: u32 = 1;

//     // -(MODULUS^{-1} mod 2^64) mod 2^64
//     const INV: u64 = 11037532056220336127;
// }

impl<C: Fp256Parameters + 'static + Send + Sync + Sized> PrimeField for NewFp256<C> {
    type Params = C;
    type BigInt = BigInteger256;
    #[inline]
    fn from_repr(r: BigInteger256) -> Option<Self> {
        let mut r = Self(r, PhantomData);
        // dbg!(r, r.0.0.iter().map(|n| format!("{:x}", n)).collect::<Vec<_>>());
        if r.is_zero() {
            Some(r)
        } else if r.is_valid() {
            r *= &Self(C::R2, PhantomData);
            Some(r)
        } else {
            None
        }
    }
    #[inline]
    #[allow(clippy::modulo_one)]
    fn into_repr(&self) -> BigInteger256 {

        // let mut tmp = self.0;
        // let mut r = tmp.0;
        // // Montgomery Reduction
        // for i in 0..$limbs {
        //     let k = r[i].wrapping_mul(P::INV);
        //     let mut carry = 0;
        //     mac_with_carry!(r[i], k, P::MODULUS.0[0], &mut carry);
        //     for j in 1..$limbs {
        //         r[(j + i) % $limbs] =
        //             mac_with_carry!(r[(j + i) % $limbs], k, P::MODULUS.0[j], &mut carry);
        //     }
        //     r[i % $limbs] = carry;
        // }
        // tmp.0 = r;
        // tmp

        // let mut tmp = self.0.to_64x4();
        // let mut r = tmp;
        // // Montgomery Reduction
        // for i in 0..4 {
        //     let k = r[i].wrapping_mul(C::INV);
        //     let mut carry = 0;
        //     mac_with_carry!(r[i], k, C::MODULUS.0[0] as _, &mut carry);
        //     for j in 1..4 {
        //         r[(j + i) % 4] =
        //             mac_with_carry!(r[(j + i) % 4], k, C::MODULUS.0[j], &mut carry);
        //     }
        //     r[i % 4] = carry;
        // }
        // tmp = r;
        // tmp

        // todo!()

        // eprintln!("AAAA={:?}", self.0.to_64x4());

        // let mut this = self.clone();
        let one = BigInteger256([1, 0, 0, 0, 0, 0, 0, 0, 0]);
        // this.mul_assign(&Self(one, PhantomData));
        // this.0
        // self.mul_without_reduce(&Self(one, PhantomData), &Default::default(), Default::default()).0
        self.mul(Self(one, PhantomData)).0
        // self.mul(Self::from(1u64)).0
    }
}

impl<C: Fp256Parameters> From<num_bigint::BigUint> for NewFp256<C> {
    fn from(val: num_bigint::BigUint) -> Self {
        Self::from_le_bytes_mod_order(&val.to_bytes_le())
    }
}
impl<C: Fp256Parameters> Into<num_bigint::BigUint> for NewFp256<C> {
    fn into(self) -> num_bigint::BigUint {
        self.into_repr().into()
    }
}

// impl FromStr for NewFp256 {
//     type Err = ();
//     /// Interpret a string of numbers as a (congruent) prime field element.
//     /// Does not accept unnecessary leading zeroes or a blank string.
//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         if s.is_empty() {
//             return Err(());
//         }
//         if s == "0" {
//             return Ok(Self::zero());
//         }
//         let mut res = Self::zero();
//         use core::convert::TryFrom;
//         let ten = Self::try_from(<Self as PrimeField>::BigInt::from(10)).unwrap();
//         let mut first_digit = true;
//         for c in s.chars() {
//             match c.to_digit(10) {
//                 Some(c) => {
//                     if first_digit {
//                         if c == 0 {
//                             return Err(());
//                         }
//                         first_digit = false;
//                     }
//                     res.mul_assign(&ten);
//                     let digit = Self::from(u64::from(c));
//                     res.add_assign(&digit);
//                 },
//                 None => {
//                     return Err(());
//                 },
//             }
//         }
//         if !res.is_valid() {
//             Err(())
//         } else {
//             Ok(res)
//         }
//     }
// }

impl<C: Fp256Parameters> FromStr for NewFp256<C> {
    type Err = ();
    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(());
        }
        if s == "0" {
            return Ok(Self::zero());
        }
        let mut res = Self::zero();
        use core::convert::TryFrom;
        let ten = Self::try_from(<Self as PrimeField>::BigInt::from(10)).unwrap();
        let mut first_digit = true;
        for c in s.chars() {
            match c.to_digit(10) {
                Some(c) => {
                    if first_digit {
                        if c == 0 {
                            return Err(());
                        }
                        first_digit = false;
                    }
                    res.mul_assign(&ten);
                    let digit = Self::from(u64::from(c));
                    res.add_assign(&digit);
                },
                None => {
                    return Err(());
                },
            }
        }
        if !res.is_valid() {
            Err(())
        } else {
            Ok(res)
        }
    }
}

impl<C: Fp256Parameters> ToBytes for NewFp256<C> {
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.into_repr().write(writer)
    }
}
impl<C: Fp256Parameters> FromBytes for NewFp256<C> {
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger256::read(reader).and_then(|b| match NewFp256::from_repr(b) {
            Some(f) => Ok(f),
            None => Err(crate::error("FromBytes::read failed")),
        })
    }
}
    // + core::iter::Sum<Self>
    // + for<'a> core::iter::Sum<&'a Self>
    // + core::iter::Product<Self>
    // + for<'a> core::iter::Product<&'a Self>

impl<C: Fp256Parameters> Field for NewFp256<C> {
    type BasePrimeField = Self;
    fn extension_degree() -> u64 {
        1
    }
    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        if elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        Some(elems[0])
    }
    #[inline]
    fn double(&self) -> Self {
        let mut temp = *self;
        temp.double_in_place();
        temp
    }
    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        self.0.mul2();
        self.reduce();
        self
    }
    #[inline]
    fn characteristic() -> &'static [u64] {
        todo!()
        // P::MODULUS.as_ref()
    }
    #[inline]
    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        if F::BIT_SIZE > 8 {
            return None;
        } else {
            let mut result_bytes = [0u8; 4 * 8 + 1];
            result_bytes
                .iter_mut()
                .zip(bytes)
                .for_each(|(result, input)| {
                    *result = *input;
                });
            let last_limb_mask = (u64::MAX >> C::REPR_SHAVE_BITS).to_le_bytes();
            let mut last_bytes_mask = [0u8; 9];
            last_bytes_mask[..8].copy_from_slice(&last_limb_mask);
            let output_byte_size = buffer_byte_size(C::MODULUS_BITS as usize + F::BIT_SIZE);
            let flag_location = output_byte_size - 1;
            let flag_location_in_last_limb = flag_location - (8 * (4 - 1));
            let last_bytes = &mut result_bytes[8 * (4 - 1)..];
            let flags_mask = u8::MAX.checked_shl(8 - (F::BIT_SIZE as u32)).unwrap_or(0);
            let mut flags: u8 = 0;
            for (i, (b, m)) in last_bytes.iter_mut().zip(&last_bytes_mask).enumerate() {
                if i == flag_location_in_last_limb {
                    flags = *b & flags_mask;
                }
                *b &= m;
            }
            Self::deserialize(&result_bytes[..(4 * 8)])
                .ok()
                .and_then(|f| F::from_u8(flags).map(|flag| (f, flag)))
        }
    }
    #[inline]
    fn square(&self) -> Self {
        let mut temp = self.clone();
        temp.square_in_place();
        temp
    }
    #[inline]
    #[allow(unused_braces, clippy::absurd_extreme_comparisons)]
    fn square_in_place(&mut self) -> &mut Self {
        let this = self.clone();
        self.mul_assign(&this);
        self
    }
    #[inline]
    fn inverse(&self) -> Option<Self> {
        use crate::FpParameters;

        if self.is_zero() {
            None
        } else {
            let one = BigInteger256::from(1);
            let mut u = self.0;
            let mut v = C::MODULUS;
            let mut b = Self(C::R2, PhantomData);
            let mut c = Self::zero();
            while u != one && v != one {
                while u.is_even() {
                    u.div2();
                    if b.0.is_even() {
                        b.0.div2();
                    } else {
                        b.0.add_nocarry(&C::MODULUS);
                        b.0.div2();
                    }
                }
                while v.is_even() {
                    v.div2();
                    if c.0.is_even() {
                        c.0.div2();
                    } else {
                        c.0.add_nocarry(&C::MODULUS);
                        c.0.div2();
                    }
                }
                if v < u {
                    u.sub_noborrow(&v);
                    b.sub_assign(&c);
                } else {
                    v.sub_noborrow(&u);
                    c.sub_assign(&b);
                }
            }
            if u == one {
                Some(b)
            } else {
                Some(c)
            }
        }
    }
    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        if let Some(inverse) = self.inverse() {
            *self = inverse;
            Some(self)
        } else {
            None
        }
    }
    /// The Frobenius map has no effect in a prime field.
    #[inline]
    fn frobenius_map(&mut self, _: usize) {}
}

impl<C: Fp256Parameters> ark_std::rand::distributions::Distribution<NewFp256<C>>
    for ark_std::rand::distributions::Standard
{
    #[inline]
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> NewFp256<C> {
        loop {
            if !(C::REPR_SHAVE_BITS <= 64) {
                panic!("assertion failed: P::REPR_SHAVE_BITS <= 64")
            }
            let mask = if C::REPR_SHAVE_BITS == 64 {
                0
            } else {
                core::u64::MAX >> C::REPR_SHAVE_BITS
            };

            let mut tmp: [u64; 4] = rng.sample(ark_std::rand::distributions::Standard);
            tmp.as_mut().last_mut().map(|val| *val &= mask);

                 let is_fp = match C::T.0[0] {
                     0x192d30ed => true,
                     0xc46eb21 => false,
                     _ => panic!(),
                 };

                 const FP_MODULUS: [u64; 4] = [
                     0x992d30ed00000001,
                     0x224698fc094cf91b,
                     0x0,
                     0x4000000000000000,
                 ];
                 const FQ_MODULUS: [u64; 4] = [
                     0x8c46eb2100000001,
                     0x224698fc0994a8dd,
                     0x0,
                     0x4000000000000000,
                 ];

                 let (modulus, inv) = if is_fp {
                     (FP_MODULUS, 11037532056220336127)
                 } else {
                     (FQ_MODULUS, 10108024940646105087)
                 };

                 let is_valid = || {
                     for (random, modulus) in tmp.iter().copied().zip(modulus).rev() {
                         if random > modulus {
                             return false;
                         } else if random < modulus {
                             return true;
                         }
                     }
                     false
                 };

                 if !is_valid() {
                     continue;
                 }

                 let mut r = tmp;
                 // Montgomery Reduction
                 for i in 0..4 {
                     let k = r[i].wrapping_mul(inv);
                     let mut carry = 0;
                     mac_with_carry!(r[i], k, modulus[0] as _, &mut carry);
                     for j in 1..4 {
                         r[(j + i) % 4] =
                             mac_with_carry!(r[(j + i) % 4], k, modulus[j], &mut carry);
                     }
                     r[i % 4] = carry;
                 }
                 tmp = r;

                 return NewFp256::<C>::from_repr(BigInteger256::from_64x4(tmp)).unwrap();
        }
    }
}

pub struct NewFpParameters;

impl<C: Fp256Parameters> zeroize::DefaultIsZeroes for NewFp256<C> {}

impl<C: Fp256Parameters> FftField for NewFp256<C> {
    type FftParams = C;
    fn two_adic_root_of_unity() -> Self {
        NewFp256::<C>(C::TWO_ADIC_ROOT_OF_UNITY, PhantomData)
    }
    fn large_subgroup_root_of_unity() -> Option<Self> {
        Some(NewFp256::<C>(C::LARGE_SUBGROUP_ROOT_OF_UNITY?, PhantomData))
    }
    fn multiplicative_generator() -> Self {
        NewFp256::<C>(C::GENERATOR, PhantomData)
    }
}

impl<C: Fp256Parameters> SquareRootField for NewFp256<C> {
    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        use crate::fields::LegendreSymbol::*;

        let modulus_minus_one_div_two = C::MODULUS_MINUS_ONE_DIV_TWO.to_64x4();
        let s = self.pow(modulus_minus_one_div_two);
        if s.is_zero() {
            Zero
        } else if s.is_one() {
            QuadraticResidue
        } else {
            QuadraticNonResidue
        }
    }
    #[inline]
    fn sqrt(&self) -> Option<Self> {
        {
            let t_minus_one_div_two = C::T_MINUS_ONE_DIV_TWO.to_64x4();

            if self.is_zero() {
                return Some(Self::zero());
            }
            let mut z = Self::qnr_to_t();
            let mut w = self.pow(t_minus_one_div_two);
            let mut x = w * self;
            let mut b = x * &w;
            let mut v = C::TWO_ADICITY as usize;
            while !b.is_one() {
                let mut k = 0usize;
                let mut b2k = b;
                while !b2k.is_one() {
                    b2k.square_in_place();
                    k += 1;
                }
                if k == (C::TWO_ADICITY as usize) {
                    return None;
                }
                let j = v - k;
                w = z;
                for _ in 1..j {
                    w.square_in_place();
                }
                z = w.square();
                b *= &z;
                x *= &w;
                v = k;
            }
            if x.square() == *self {
                return Some(x);
            } else {
                #[cfg(debug_assertions)]
                {
                    use crate::fields::LegendreSymbol::*;
                    if self.legendre() != QuadraticNonResidue {
                        panic!(
                            "Input has a square root per its legendre symbol, but it was not found",
                        )
                    }
                }
                None
            }
        }
    }
    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        (*self).sqrt().map(|sqrt| {
            *self = sqrt;
            self
        })
    }
}


pub trait Fp256Parameters: crate::FpParameters<BigInt = BigInteger256> + ark_std::fmt::Debug + Clone + Copy + Default + Eq + PartialEq + PartialOrd + Ord + core::hash::Hash + 'static + Send + Sync + Sized {}

// use num_traits::{One, Zero};


// /// Represents an element of the prime field F_p, where `p == P::MODULUS`.
// /// This type can represent elements in any field of size at most
// ///256
// /// bits.
// // #[derivative(
// //     Default(bound = ""),
// //     Hash(bound = ""),
// //     Clone(bound = ""),
// //     Copy(bound = ""),
// //     PartialEq(bound = ""),
// //     Eq(bound = "")
// // )]
// pub struct Fp256<P>(
//     pub BigInteger256,
//     // #[derivative(Debug = "ignore")]
//     #[doc(hidden)]
//     pub PhantomData<P>,
// );
// #[allow(unused_qualifications)]
// impl<P> ::core::clone::Clone for Fp256<P> {
//     fn clone(&self) -> Self {
//         match *self {
//             Fp256(ref __arg_0, ref __arg_1) => Fp256((*__arg_0).clone(), (*__arg_1).clone()),
//         }
//     }
// }
// #[allow(unused_qualifications)]
// impl<P> ::core::marker::Copy for Fp256<P> {}
// #[allow(unused_qualifications)]
// impl<P> ::core::default::Default for Fp256<P> {
//     fn default() -> Self {
//         Fp256(
//             ::core::default::Default::default(),
//             ::core::default::Default::default(),
//         )
//     }
// }
// #[allow(unused_qualifications)]
// impl<P> ::core::cmp::Eq for Fp256<P> {}
// #[allow(unused_qualifications)]
// impl<P> ::core::hash::Hash for Fp256<P> {
//     fn hash<__HP>(&self, __state: &mut __HP)
//     where
//         __HP: ::core::hash::Hasher,
//     {
//         match *self {
//             Fp256(ref __arg_0, ref __arg_1) => {
//                 ::core::hash::Hash::hash(&(*__arg_0), __state);
//                 ::core::hash::Hash::hash(&(*__arg_1), __state);
//             },
//         }
//     }
// }
// #[allow(unused_qualifications)]
// #[allow(clippy::unneeded_field_pattern)]
// impl<P> ::core::cmp::PartialEq for Fp256<P> {
//     fn eq(&self, other: &Self) -> bool {
//         true && match *self {
//             Fp256(ref __self_0, ref __self_1) => match *other {
//                 Fp256(ref __other_0, ref __other_1) => {
//                     true && &(*__self_0) == &(*__other_0) && &(*__self_1) == &(*__other_1)
//                 },
//             },
//         }
//     }
// }
// impl<P: Fp256Parameters> ark_std::fmt::Debug for Fp256<P> {
//     fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
//         use crate::ark_std::string::ToString;
//         let r = self.into_repr();
//         let bigint: num_bigint::BigUint = r.into();
//         let s = bigint.to_string();
//         let name = match P::INV {
//             11037532056220336127 => "Fp",
//             10108024940646105087 => "Fq",
//             _ => "Field",
//         };
//         f.write_fmt(format_args!("{0}({1})", name, s))
//     }
// }
// impl<P> Fp256<P> {
//     #[inline]
//     pub const fn new(element: BigInteger256) -> Self {
//         Self(element, PhantomData)
//     }
//     const fn const_is_zero(&self) -> bool {
//         let mut is_zero = true;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 is_zero &= (self.0).0[i] == 0;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 is_zero &= (self.0).0[i] == 0;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 is_zero &= (self.0).0[i] == 0;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 is_zero &= (self.0).0[i] == 0;
//             }
//         }
//         is_zero
//     }
//     const fn const_neg(self, modulus: BigInteger256) -> Self {
//         if !self.const_is_zero() {
//             Self::new(Self::sub_noborrow(&modulus, &self.0))
//         } else {
//             self
//         }
//     }
//     /// Interpret a string of decimal numbers as a prime field element.
//     /// Does not accept unnecessary leading zeroes or a blank string.
//     /// For *internal* use only; please use the `field_new` macro instead
//     /// of this method
//     #[doc(hidden)]
//     pub const fn const_from_str(
//         limbs: &[u64],
//         is_positive: bool,
//         r2: BigInteger256,
//         modulus: BigInteger256,
//         inv: u64,
//     ) -> Self {
//         let mut repr = BigInteger256([0; 4]);
//         let mut i = 0;
//         while i < limbs.len() {
//             repr.0[i] = limbs[i];
//             i += 1;
//         }
//         let res = Self::const_from_repr(repr, r2, modulus, inv);
//         if is_positive {
//             res
//         } else {
//             res.const_neg(modulus)
//         }
//     }
//     #[inline]
//     pub(crate) const fn const_from_repr(
//         repr: BigInteger256,
//         r2: BigInteger256,
//         modulus: BigInteger256,
//         inv: u64,
//     ) -> Self {
//         let mut r = Self::new(repr);
//         if r.const_is_zero() {
//             r
//         } else {
//             r = r.const_mul(&Fp256(r2, PhantomData), modulus, inv);
//             r
//         }
//     }
//     const fn mul_without_reduce(mut self, other: &Self, modulus: BigInteger256, inv: u64) -> Self {
//         let mut r = [0u64; 4 * 2];
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 let mut carry = 0;
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 0usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 let mut carry = 0;
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 0usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 let mut carry = 0;
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 0usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 let mut carry = 0;
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 0usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = carry;
//             }
//         }
//         let mut _carry2 = 0;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 let k = r[i].wrapping_mul(inv);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 let k = r[i].wrapping_mul(inv);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 let k = r[i].wrapping_mul(inv);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 let k = r[i].wrapping_mul(inv);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + (k as u128 * modulus.0[j] as u128)
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//         }
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 (self.0).0[i] = r[4 + i];
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 (self.0).0[i] = r[4 + i];
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 (self.0).0[i] = r[4 + i];
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 (self.0).0[i] = r[4 + i];
//             }
//         }
//         self
//     }
//     const fn const_mul(mut self, other: &Self, modulus: BigInteger256, inv: u64) -> Self {
//         self = self.mul_without_reduce(other, modulus, inv);
//         self.const_reduce(modulus)
//     }
//     const fn const_is_valid(&self, modulus: BigInteger256) -> bool {
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
//                     return true;
//                 } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
//                     return false;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
//                     return true;
//                 } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
//                     return false;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
//                     return true;
//                 } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
//                     return false;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
//                     return true;
//                 } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
//                     return false;
//                 }
//             }
//         }
//         false
//     }
//     #[inline]
//     const fn const_reduce(mut self, modulus: BigInteger256) -> Self {
//         if !self.const_is_valid(modulus) {
//             self.0 = Self::sub_noborrow(&self.0, &modulus);
//         }
//         self
//     }
//     #[allow(unused_assignments)]
//     const fn sub_noborrow(a: &BigInteger256, b: &BigInteger256) -> BigInteger256 {
//         let mut a = *a;
//         let mut borrow = 0;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 a.0[i] = {
//                     let tmp =
//                         (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
//                     borrow = if tmp >> 64 == 0 { 1 } else { 0 };
//                     tmp as u64
//                 };
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 a.0[i] = {
//                     let tmp =
//                         (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
//                     borrow = if tmp >> 64 == 0 { 1 } else { 0 };
//                     tmp as u64
//                 };
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 a.0[i] = {
//                     let tmp =
//                         (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
//                     borrow = if tmp >> 64 == 0 { 1 } else { 0 };
//                     tmp as u64
//                 };
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 a.0[i] = {
//                     let tmp =
//                         (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
//                     borrow = if tmp >> 64 == 0 { 1 } else { 0 };
//                     tmp as u64
//                 };
//             }
//         }
//         a
//     }
// }

// impl<P: Fp256Parameters> Fp256<P> {
//     #[inline(always)]
//     pub(crate) fn is_valid(&self) -> bool {
//         self.0 < P::MODULUS
//     }
//     #[inline]
//     fn reduce(&mut self) {
//         if !self.is_valid() {
//             self.0.sub_noborrow(&P::MODULUS);
//         }
//     }
// }
// impl<P: Fp256Parameters> Zero for Fp256<P> {
//     #[inline]
//     fn zero() -> Self {
//         Fp256::<P>(BigInteger256::from(0), PhantomData)
//     }
//     #[inline]
//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }
// impl<P: Fp256Parameters> One for Fp256<P> {
//     #[inline]
//     fn one() -> Self {
//         Fp256::<P>(P::R, PhantomData)
//     }
//     #[inline]
//     fn is_one(&self) -> bool {
//         self.0 == P::R
//     }
// }

// impl<P: Fp256Parameters> Field for Fp256<P> {
//     type BasePrimeField = Self;
//     fn extension_degree() -> u64 {
//         1
//     }
//     fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
//         if elems.len() != (Self::extension_degree() as usize) {
//             return None;
//         }
//         Some(elems[0])
//     }
//     #[inline]
//     fn double(&self) -> Self {
//         let mut temp = *self;
//         temp.double_in_place();
//         temp
//     }
//     #[inline]
//     fn double_in_place(&mut self) -> &mut Self {
//         self.0.mul2();
//         self.reduce();
//         self
//     }
//     #[inline]
//     fn characteristic() -> &'static [u64] {
//         P::MODULUS.as_ref()
//     }
//     #[inline]
//     fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
//         if F::BIT_SIZE > 8 {
//             return None;
//         } else {
//             let mut result_bytes = [0u8; 4 * 8 + 1];
//             result_bytes
//                 .iter_mut()
//                 .zip(bytes)
//                 .for_each(|(result, input)| {
//                     *result = *input;
//                 });
//             let last_limb_mask = (u64::MAX >> P::REPR_SHAVE_BITS).to_le_bytes();
//             let mut last_bytes_mask = [0u8; 9];
//             last_bytes_mask[..8].copy_from_slice(&last_limb_mask);
//             let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
//             let flag_location = output_byte_size - 1;
//             let flag_location_in_last_limb = flag_location - (8 * (4 - 1));
//             let last_bytes = &mut result_bytes[8 * (4 - 1)..];
//             let flags_mask = u8::MAX.checked_shl(8 - (F::BIT_SIZE as u32)).unwrap_or(0);
//             let mut flags: u8 = 0;
//             for (i, (b, m)) in last_bytes.iter_mut().zip(&last_bytes_mask).enumerate() {
//                 if i == flag_location_in_last_limb {
//                     flags = *b & flags_mask;
//                 }
//                 *b &= m;
//             }
//             Self::deserialize(&result_bytes[..(4 * 8)])
//                 .ok()
//                 .and_then(|f| F::from_u8(flags).map(|flag| (f, flag)))
//         }
//     }
//     #[inline]
//     fn square(&self) -> Self {
//         let mut temp = self.clone();
//         temp.square_in_place();
//         temp
//     }
//     #[inline]
//     #[allow(unused_braces, clippy::absurd_extreme_comparisons)]
//     fn square_in_place(&mut self) -> &mut Self {
//         if 4 == 1 {
//             *self = *self * *self;
//             return self;
//         }
//         let mut r = [0u64; 4 * 2];
//         let mut carry = 0;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 if i < 4 - 1 {
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 0usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                     }
//                     r[4 + i] = carry;
//                     carry = 0;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 if i < 4 - 1 {
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 0usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                     }
//                     r[4 + i] = carry;
//                     carry = 0;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 if i < 4 - 1 {
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 0usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                     }
//                     r[4 + i] = carry;
//                     carry = 0;
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 if i < 4 - 1 {
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 0usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             if j > i {
//                                 r[i + j] = {
//                                     let tmp = (r[i + j] as u128)
//                                         + crate::biginteger::arithmetic::u64_mul_u64(
//                                             (self.0).0[i],
//                                             (self.0).0[j],
//                                         )
//                                         + (carry as u128);
//                                     carry = (tmp >> 64) as u64;
//                                     tmp as u64
//                                 };
//                             }
//                         }
//                     }
//                     r[4 + i] = carry;
//                     carry = 0;
//                 }
//             }
//         }
//         r[4 * 2 - 1] = r[4 * 2 - 2] >> 63;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 let subtractor = (2 * (4 - 1usize))
//                     .checked_sub(i + 1)
//                     .map(|index| r[index])
//                     .unwrap_or(0);
//                 r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 let subtractor = (2 * (4 - 1usize))
//                     .checked_sub(i + 1)
//                     .map(|index| r[index])
//                     .unwrap_or(0);
//                 r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 let subtractor = (2 * (4 - 1usize))
//                     .checked_sub(i + 1)
//                     .map(|index| r[index])
//                     .unwrap_or(0);
//                 r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 let subtractor = (2 * (4 - 1usize))
//                     .checked_sub(i + 1)
//                     .map(|index| r[index])
//                     .unwrap_or(0);
//                 r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
//             }
//         }
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 r[4 + 1 - i] = (r[4 + 1 - i] << 1) | (r[4 - i] >> 63);
//             }
//         }
//         r[1] <<= 1;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 r[2 * i] = {
//                     let tmp = (r[2 * i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 #[allow(unused_assignments)]
//                 {
//                     r[2 * i + 1] = {
//                         let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
//                         carry = (tmp >> 64) as u64;
//                         tmp as u64
//                     };
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 r[2 * i] = {
//                     let tmp = (r[2 * i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 #[allow(unused_assignments)]
//                 {
//                     r[2 * i + 1] = {
//                         let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
//                         carry = (tmp >> 64) as u64;
//                         tmp as u64
//                     };
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 r[2 * i] = {
//                     let tmp = (r[2 * i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 #[allow(unused_assignments)]
//                 {
//                     r[2 * i + 1] = {
//                         let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
//                         carry = (tmp >> 64) as u64;
//                         tmp as u64
//                     };
//                 }
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 r[2 * i] = {
//                     let tmp = (r[2 * i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 #[allow(unused_assignments)]
//                 {
//                     r[2 * i + 1] = {
//                         let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
//                         carry = (tmp >> 64) as u64;
//                         tmp as u64
//                     };
//                 }
//             }
//         }
//         let mut _carry2 = 0;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[j + i] = {
//                             let tmp = (r[j + i] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[4 + i] = {
//                     let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 _carry2 = carry;
//             }
//         }
//         (self.0).0.copy_from_slice(&r[4..]);
//         self.reduce();
//         self
//     }
//     #[inline]
//     fn inverse(&self) -> Option<Self> {
//         if self.is_zero() {
//             None
//         } else {
//             let one = BigInteger256::from(1);
//             let mut u = self.0;
//             let mut v = P::MODULUS;
//             let mut b = Fp256::<P>(P::R2, PhantomData);
//             let mut c = Self::zero();
//             while u != one && v != one {
//                 while u.is_even() {
//                     u.div2();
//                     if b.0.is_even() {
//                         b.0.div2();
//                     } else {
//                         b.0.add_nocarry(&P::MODULUS);
//                         b.0.div2();
//                     }
//                 }
//                 while v.is_even() {
//                     v.div2();
//                     if c.0.is_even() {
//                         c.0.div2();
//                     } else {
//                         c.0.add_nocarry(&P::MODULUS);
//                         c.0.div2();
//                     }
//                 }
//                 if v < u {
//                     u.sub_noborrow(&v);
//                     b.sub_assign(&c);
//                 } else {
//                     v.sub_noborrow(&u);
//                     c.sub_assign(&b);
//                 }
//             }
//             if u == one {
//                 Some(b)
//             } else {
//                 Some(c)
//             }
//         }
//     }
//     fn inverse_in_place(&mut self) -> Option<&mut Self> {
//         if let Some(inverse) = self.inverse() {
//             *self = inverse;
//             Some(self)
//         } else {
//             None
//         }
//     }
//     /// The Frobenius map has no effect in a prime field.
//     #[inline]
//     fn frobenius_map(&mut self, _: usize) {}
// }



// impl<P: Fp256Parameters> PrimeField for Fp256<P> {
//     type Params = P;
//     type BigInt = BigInteger256;
//     #[inline]
//     fn from_repr(r: BigInteger256) -> Option<Self> {
//         let mut r = Fp256(r, PhantomData);
//         if r.is_zero() {
//             Some(r)
//         } else if r.is_valid() {
//             r *= &Fp256(P::R2, PhantomData);
//             Some(r)
//         } else {
//             None
//         }
//     }
//     #[inline]
//     #[allow(clippy::modulo_one)]
//     fn into_repr(&self) -> BigInteger256 {
//         let mut tmp = self.0;
//         let mut r = tmp.0;
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 0usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[i % 4] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[i % 4] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[i % 4] = carry;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 let k = r[i].wrapping_mul(P::INV);
//                 let mut carry = 0;
//                 {
//                     let tmp = (r[i] as u128)
//                         + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
//                         + (carry as u128);
//                     carry = (tmp >> 64) as u64;
//                     tmp as u64
//                 };
//                 {
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 1usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 2usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                     {
//                         #[allow(non_upper_case_globals)]
//                         const j: usize = 3usize;
//                         r[(j + i) % 4] = {
//                             let tmp = (r[(j + i) % 4] as u128)
//                                 + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
//                                 + (carry as u128);
//                             carry = (tmp >> 64) as u64;
//                             tmp as u64
//                         };
//                     }
//                 }
//                 r[i % 4] = carry;
//             }
//         }
//         tmp.0 = r;
//         tmp
//     }
// }
// impl<P: Fp256Parameters> FftField for Fp256<P> {
//     type FftParams = P;
//     #[inline]
//     fn two_adic_root_of_unity() -> Self {
//         Fp256::<P>(P::TWO_ADIC_ROOT_OF_UNITY, PhantomData)
//     }
//     #[inline]
//     fn large_subgroup_root_of_unity() -> Option<Self> {
//         Some(Fp256::<P>(P::LARGE_SUBGROUP_ROOT_OF_UNITY?, PhantomData))
//     }
//     #[inline]
//     fn multiplicative_generator() -> Self {
//         Fp256::<P>(P::GENERATOR, PhantomData)
//     }
// }
// impl<P: Fp256Parameters> SquareRootField for Fp256<P> {
//     #[inline]
//     fn legendre(&self) -> LegendreSymbol {
//         use crate::fields::LegendreSymbol::*;
//         let s = self.pow(P::MODULUS_MINUS_ONE_DIV_TWO);
//         if s.is_zero() {
//             Zero
//         } else if s.is_one() {
//             QuadraticResidue
//         } else {
//             QuadraticNonResidue
//         }
//     }
//     #[inline]
//     fn sqrt(&self) -> Option<Self> {
//         {
//             if self.is_zero() {
//                 return Some(Self::zero());
//             }
//             let mut z = Self::qnr_to_t();
//             let mut w = self.pow(P::T_MINUS_ONE_DIV_TWO);
//             let mut x = w * self;
//             let mut b = x * &w;
//             let mut v = P::TWO_ADICITY as usize;
//             while !b.is_one() {
//                 let mut k = 0usize;
//                 let mut b2k = b;
//                 while !b2k.is_one() {
//                     b2k.square_in_place();
//                     k += 1;
//                 }
//                 if k == (P::TWO_ADICITY as usize) {
//                     return None;
//                 }
//                 let j = v - k;
//                 w = z;
//                 for _ in 1..j {
//                     w.square_in_place();
//                 }
//                 z = w.square();
//                 b *= &z;
//                 x *= &w;
//                 v = k;
//             }
//             if x.square() == *self {
//                 return Some(x);
//             } else {
//                 #[cfg(debug_assertions)]
//                 {
//                     use crate::fields::LegendreSymbol::*;
//                     if self.legendre() != QuadraticNonResidue {
//                         panic!(
//                             "Input has a square root per its legendre symbol, but it was not found",
//                         )
//                     }
//                 }
//                 None
//             }
//         }
//     }
//     fn sqrt_in_place(&mut self) -> Option<&mut Self> {
//         (*self).sqrt().map(|sqrt| {
//             *self = sqrt;
//             self
//         })
//     }
// }

// /// Note that this implementation of `Ord` compares field elements viewing
// /// them as integers in the range 0, 1, ..., P::MODULUS - 1. However, other
// /// implementations of `PrimeField` might choose a different ordering, and
// /// as such, users should use this `Ord` for applications where
// /// any ordering suffices (like in a BTreeMap), and not in applications
// /// where a particular ordering is required.
// impl<P: Fp256Parameters> Ord for Fp256<P> {
//     #[inline(always)]
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.into_repr().cmp(&other.into_repr())
//     }
// }
// /// Note that this implementation of `PartialOrd` compares field elements viewing
// /// them as integers in the range 0, 1, ..., `P::MODULUS` - 1. However, other
// /// implementations of `PrimeField` might choose a different ordering, and
// /// as such, users should use this `PartialOrd` for applications where
// /// any ordering suffices (like in a BTreeMap), and not in applications
// /// where a particular ordering is required.
// impl<P: Fp256Parameters> PartialOrd for Fp256<P> {
//     #[inline(always)]
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }
// impl<P: Fp256Parameters> From<u128> for Fp256<P> {
//     fn from(other: u128) -> Self {
//         let mut default_int = P::BigInt::default();
//         if 4 == 1 {
//             default_int.0[0] = (other % u128::from(P::MODULUS.0[0])) as u64;
//         } else {
//             let upper = (other >> 64) as u64;
//             let lower = ((other << 64) >> 64) as u64;
//             let limbs = [lower, upper];
//             for (cur, other) in default_int.0.iter_mut().zip(&limbs) {
//                 *cur = *other;
//             }
//         }
//         Self::from_repr(default_int).unwrap()
//     }
// }
// impl<P: Fp256Parameters> From<i128> for Fp256<P> {
//     fn from(other: i128) -> Self {
//         let abs = Self::from(other.unsigned_abs());
//         if other.is_positive() {
//             abs
//         } else {
//             -abs
//         }
//     }
// }
// impl<P: Fp256Parameters> From<u64> for Fp256<P> {
//     fn from(other: u64) -> Self {
//         if 4 == 1 {
//             Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
//         } else {
//             Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
//         }
//     }
// }
// impl<P: Fp256Parameters> From<i64> for Fp256<P> {
//     fn from(other: i64) -> Self {
//         let abs = Self::from(other.unsigned_abs());
//         if other.is_positive() {
//             abs
//         } else {
//             -abs
//         }
//     }
// }
// impl<P: Fp256Parameters> From<u32> for Fp256<P> {
//     fn from(other: u32) -> Self {
//         if 4 == 1 {
//             Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
//         } else {
//             Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
//         }
//     }
// }
// impl<P: Fp256Parameters> From<i32> for Fp256<P> {
//     fn from(other: i32) -> Self {
//         let abs = Self::from(other.unsigned_abs());
//         if other.is_positive() {
//             abs
//         } else {
//             -abs
//         }
//     }
// }
// impl<P: Fp256Parameters> From<u16> for Fp256<P> {
//     fn from(other: u16) -> Self {
//         if 4 == 1 {
//             Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
//         } else {
//             Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
//         }
//     }
// }
// impl<P: Fp256Parameters> From<i16> for Fp256<P> {
//     fn from(other: i16) -> Self {
//         let abs = Self::from(other.unsigned_abs());
//         if other.is_positive() {
//             abs
//         } else {
//             -abs
//         }
//     }
// }
// impl<P: Fp256Parameters> From<u8> for Fp256<P> {
//     fn from(other: u8) -> Self {
//         if 4 == 1 {
//             Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
//         } else {
//             Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
//         }
//     }
// }
// impl<P: Fp256Parameters> From<i8> for Fp256<P> {
//     fn from(other: i8) -> Self {
//         let abs = Self::from(other.unsigned_abs());
//         if other.is_positive() {
//             abs
//         } else {
//             -abs
//         }
//     }
// }
// impl<P: Fp256Parameters> From<bool> for Fp256<P> {
//     fn from(other: bool) -> Self {
//         if 4 == 1 {
//             Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
//         } else {
//             Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
//         }
//     }
// }
// impl<P: Fp256Parameters> ark_std::rand::distributions::Distribution<Fp256<P>>
//     for ark_std::rand::distributions::Standard
// {
//     #[inline]
//     fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp256<P> {
//         loop {
//             let mut tmp = Fp256(
//                 rng.sample(ark_std::rand::distributions::Standard),
//                 PhantomData,
//             );
//             if !(P::REPR_SHAVE_BITS <= 64) {
//                 panic!("assertion failed: P::REPR_SHAVE_BITS <= 64")
//             }
//             let mask = if P::REPR_SHAVE_BITS == 64 {
//                 0
//             } else {
//                 core::u64::MAX >> P::REPR_SHAVE_BITS
//             };
//             tmp.0.as_mut().last_mut().map(|val| *val &= mask);
//             if tmp.is_valid() {
//                 return tmp;
//             }
//         }
//     }
// }
// impl<P: Fp256Parameters> CanonicalSerializeWithFlags for Fp256<P> {
//     fn serialize_with_flags<W: ark_std::io::Write, F: Flags>(
//         &self,
//         mut writer: W,
//         flags: F,
//     ) -> Result<(), SerializationError> {
//         if F::BIT_SIZE > 8 {
//             return Err(SerializationError::NotEnoughSpace);
//         }
//         let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
//         let mut bytes = [0u8; 4 * 8 + 1];
//         self.write(&mut bytes[..4 * 8])?;
//         bytes[output_byte_size - 1] |= flags.u8_bitmask();
//         writer.write_all(&bytes[..output_byte_size])?;
//         Ok(())
//     }
//     fn serialized_size_with_flags<F: Flags>(&self) -> usize {
//         buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE)
//     }
// }
// impl<P: Fp256Parameters> CanonicalSerialize for Fp256<P> {
//     #[inline]
//     fn serialize<W: ark_std::io::Write>(&self, writer: W) -> Result<(), SerializationError> {
//         self.serialize_with_flags(writer, EmptyFlags)
//     }
//     #[inline]
//     fn serialized_size(&self) -> usize {
//         self.serialized_size_with_flags::<EmptyFlags>()
//     }
// }
// impl<P: Fp256Parameters> CanonicalDeserializeWithFlags for Fp256<P> {
//     fn deserialize_with_flags<R: ark_std::io::Read, F: Flags>(
//         mut reader: R,
//     ) -> Result<(Self, F), SerializationError> {
//         if F::BIT_SIZE > 8 {
//             return Err(SerializationError::NotEnoughSpace);
//         }
//         let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
//         let mut masked_bytes = [0; 4 * 8 + 1];
//         reader.read_exact(&mut masked_bytes[..output_byte_size])?;
//         let flags = F::from_u8_remove_flags(&mut masked_bytes[output_byte_size - 1])
//             .ok_or(SerializationError::UnexpectedFlags)?;
//         Ok((Self::read(&masked_bytes[..])?, flags))
//     }
// }
// impl<P: Fp256Parameters> CanonicalDeserialize for Fp256<P> {
//     fn deserialize<R: ark_std::io::Read>(reader: R) -> Result<Self, SerializationError> {
//         Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
//     }
// }
// impl<P: Fp256Parameters> ToBytes for Fp256<P> {
//     #[inline]
//     fn write<W: Write>(&self, writer: W) -> IoResult<()> {
//         self.into_repr().write(writer)
//     }
// }
// impl<P: Fp256Parameters> FromBytes for Fp256<P> {
//     #[inline]
//     fn read<R: Read>(reader: R) -> IoResult<Self> {
//         BigInteger256::read(reader).and_then(|b| match Fp256::from_repr(b) {
//             Some(f) => Ok(f),
//             None => Err(crate::error("FromBytes::read failed")),
//         })
//     }
// }
// impl<P: Fp256Parameters> FromStr for Fp256<P> {
//     type Err = ();
//     /// Interpret a string of numbers as a (congruent) prime field element.
//     /// Does not accept unnecessary leading zeroes or a blank string.
//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         if s.is_empty() {
//             return Err(());
//         }
//         if s == "0" {
//             return Ok(Self::zero());
//         }
//         let mut res = Self::zero();
//         use core::convert::TryFrom;
//         let ten = Self::try_from(<Self as PrimeField>::BigInt::from(10)).unwrap();
//         let mut first_digit = true;
//         for c in s.chars() {
//             match c.to_digit(10) {
//                 Some(c) => {
//                     if first_digit {
//                         if c == 0 {
//                             return Err(());
//                         }
//                         first_digit = false;
//                     }
//                     res.mul_assign(&ten);
//                     let digit = Self::from(u64::from(c));
//                     res.add_assign(&digit);
//                 },
//                 None => {
//                     return Err(());
//                 },
//             }
//         }
//         if !res.is_valid() {
//             Err(())
//         } else {
//             Ok(res)
//         }
//     }
// }
// /// Outputs a string containing the value of `self`, chunked up into
// /// 64-bit limbs.
// impl<P: Fp256Parameters> Display for Fp256<P> {
//     #[inline]
//     fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
//         f.write_fmt(format_args!("Fp256 \"({0})\"", self.into_repr()))
//     }
// }
// impl<P: Fp256Parameters> Neg for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     #[must_use]
//     fn neg(self) -> Self {
//         if !self.is_zero() {
//             let mut tmp = P::MODULUS;
//             tmp.sub_noborrow(&self.0);
//             Fp256::<P>(tmp, PhantomData)
//         } else {
//             self
//         }
//     }
// }
// impl<'a, P: Fp256Parameters> Add<&'a Fp256<P>> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn add(mut self, other: &Self) -> Self {
//         self.add_assign(other);
//         self
//     }
// }
// impl<'a, P: Fp256Parameters> Sub<&'a Fp256<P>> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn sub(mut self, other: &Self) -> Self {
//         self.sub_assign(other);
//         self
//     }
// }
// impl<'a, P: Fp256Parameters> Mul<&'a Fp256<P>> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn mul(mut self, other: &Self) -> Self {
//         self.mul_assign(other);
//         self
//     }
// }
// impl<'a, P: Fp256Parameters> Div<&'a Fp256<P>> for Fp256<P> {
//     type Output = Self;
//     /// Returns `self * other.inverse()` if `other.inverse()` is `Some`, and
//     /// panics otherwise.
//     #[inline]
//     fn div(mut self, other: &Self) -> Self {
//         self.mul_assign(&other.inverse().unwrap());
//         self
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::Add<Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn add(self, other: Self) -> Self {
//         let mut result = self;
//         result.add_assign(&other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::Add<&'a mut Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn add(self, other: &'a mut Self) -> Self {
//         let mut result = self;
//         result.add_assign(&*other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::Sub<Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn sub(self, other: Self) -> Self {
//         let mut result = self;
//         result.sub_assign(&other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::Sub<&'a mut Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn sub(self, other: &'a mut Self) -> Self {
//         let mut result = self;
//         result.sub_assign(&*other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::iter::Sum<Self> for Fp256<P> {
//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(Self::zero(), core::ops::Add::add)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::iter::Sum<&'a Self> for Fp256<P> {
//     fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
//         iter.fold(Self::zero(), core::ops::Add::add)
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::AddAssign<Self> for Fp256<P> {
//     fn add_assign(&mut self, other: Self) {
//         self.add_assign(&other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::SubAssign<Self> for Fp256<P> {
//     fn sub_assign(&mut self, other: Self) {
//         self.sub_assign(&other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::AddAssign<&'a mut Self> for Fp256<P> {
//     fn add_assign(&mut self, other: &'a mut Self) {
//         self.add_assign(&*other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::SubAssign<&'a mut Self> for Fp256<P> {
//     fn sub_assign(&mut self, other: &'a mut Self) {
//         self.sub_assign(&*other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::Mul<Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn mul(self, other: Self) -> Self {
//         let mut result = self;
//         result.mul_assign(&other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::Div<Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn div(self, other: Self) -> Self {
//         let mut result = self;
//         result.div_assign(&other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::Mul<&'a mut Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn mul(self, other: &'a mut Self) -> Self {
//         let mut result = self;
//         result.mul_assign(&*other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::Div<&'a mut Self> for Fp256<P> {
//     type Output = Self;
//     #[inline]
//     fn div(self, other: &'a mut Self) -> Self {
//         let mut result = self;
//         result.div_assign(&*other);
//         result
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::iter::Product<Self> for Fp256<P> {
//     fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(Self::one(), core::ops::Mul::mul)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::iter::Product<&'a Self> for Fp256<P> {
//     fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
//         iter.fold(Self::one(), Mul::mul)
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::MulAssign<Self> for Fp256<P> {
//     fn mul_assign(&mut self, other: Self) {
//         self.mul_assign(&other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::DivAssign<&'a mut Self> for Fp256<P> {
//     fn div_assign(&mut self, other: &'a mut Self) {
//         self.div_assign(&*other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<'a, P: Fp256Parameters> core::ops::MulAssign<&'a mut Self> for Fp256<P> {
//     fn mul_assign(&mut self, other: &'a mut Self) {
//         self.mul_assign(&*other)
//     }
// }
// #[allow(unused_qualifications)]
// impl<P: Fp256Parameters> core::ops::DivAssign<Self> for Fp256<P> {
//     fn div_assign(&mut self, other: Self) {
//         self.div_assign(&other)
//     }
// }
// impl<'a, P: Fp256Parameters> AddAssign<&'a Self> for Fp256<P> {
//     #[inline]
//     fn add_assign(&mut self, other: &Self) {
//         self.0.add_nocarry(&other.0);
//         self.reduce();
//     }
// }
// impl<'a, P: Fp256Parameters> SubAssign<&'a Self> for Fp256<P> {
//     #[inline]
//     fn sub_assign(&mut self, other: &Self) {
//         if other.0 > self.0 {
//             self.0.add_nocarry(&P::MODULUS);
//         }
//         self.0.sub_noborrow(&other.0);
//     }
// }
// impl<'a, P: Fp256Parameters> MulAssign<&'a Self> for Fp256<P> {
//     #[inline]
//     fn mul_assign(&mut self, other: &Self) {
//         let first_bit_set = P::MODULUS.0[4 - 1] >> 63 != 0;
//         #[allow(unused_mut)]
//         let mut all_bits_set = P::MODULUS.0[4 - 1] == !0 - (1 << 63);
//         {
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 1usize;
//                 all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 2usize;
//                 all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
//             }
//             {
//                 #[allow(non_upper_case_globals)]
//                 const i: usize = 3usize;
//                 all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
//             }
//         }
//         let _no_carry: bool = !(first_bit_set || all_bits_set);
//         if _no_carry {
//             let mut r = [0u64; 4];
//             let mut carry1 = 0u64;
//             let mut carry2 = 0u64;
//             {
//                 {
//                     #[allow(non_upper_case_globals)]
//                     const i: usize = 0usize;
//                     r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
//                     let k = r[0].wrapping_mul(P::INV);
//                     fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                     }
//                     r[4 - 1] = carry1 + carry2;
//                 }
//                 {
//                     #[allow(non_upper_case_globals)]
//                     const i: usize = 1usize;
//                     r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
//                     let k = r[0].wrapping_mul(P::INV);
//                     fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                     }
//                     r[4 - 1] = carry1 + carry2;
//                 }
//                 {
//                     #[allow(non_upper_case_globals)]
//                     const i: usize = 2usize;
//                     r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
//                     let k = r[0].wrapping_mul(P::INV);
//                     fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                     }
//                     r[4 - 1] = carry1 + carry2;
//                 }
//                 {
//                     #[allow(non_upper_case_globals)]
//                     const i: usize = 3usize;
//                     r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
//                     let k = r[0].wrapping_mul(P::INV);
//                     fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
//                     {
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 1usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 2usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                         {
//                             #[allow(non_upper_case_globals)]
//                             const j: usize = 3usize;
//                             r[j] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         (self.0).0[j],
//                                         (other.0).0[i],
//                                     )
//                                     + (carry1 as u128);
//                                 carry1 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                             r[j - 1] = {
//                                 let tmp = (r[j] as u128)
//                                     + crate::biginteger::arithmetic::u64_mul_u64(
//                                         k,
//                                         P::MODULUS.0[j],
//                                     )
//                                     + (carry2 as u128);
//                                 carry2 = (tmp >> 64) as u64;
//                                 tmp as u64
//                             };
//                         }
//                     }
//                     r[4 - 1] = carry1 + carry2;
//                 }
//             }
//             (self.0).0 = r;
//             self.reduce();
//         } else {
//             *self = self.mul_without_reduce(other, P::MODULUS, P::INV);
//             self.reduce();
//         }
//     }
// }
// /// Computes `self *= other.inverse()` if `other.inverse()` is `Some`, and
// /// panics otherwise.
// impl<'a, P: Fp256Parameters> DivAssign<&'a Self> for Fp256<P> {
//     #[inline]
//     fn div_assign(&mut self, other: &Self) {
//         self.mul_assign(&other.inverse().unwrap());
//     }
// }
// impl<P: Fp256Parameters> zeroize::Zeroize for Fp256<P> {
//     fn zeroize(&mut self) {
//         self.0.zeroize();
//     }
// }
// impl<P: Fp256Parameters> From<num_bigint::BigUint> for Fp256<P> {
//     #[inline]
//     fn from(val: num_bigint::BigUint) -> Fp256<P> {
//         Fp256::<P>::from_le_bytes_mod_order(&val.to_bytes_le())
//     }
// }
// impl<P: Fp256Parameters> Into<num_bigint::BigUint> for Fp256<P> {
//     #[inline]
//     fn into(self) -> num_bigint::BigUint {
//         self.into_repr().into()
//     }
// }
