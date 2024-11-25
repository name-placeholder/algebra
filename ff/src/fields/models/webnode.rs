use crate::{
    biginteger::{
        arithmetic as fa, BigInteger as _BigInteger, BigInteger256,
    },
    bytes::{FromBytes, ToBytes},
    fields::{FftField, Field, FpParameters, LegendreSymbol, PrimeField, SquareRootField},
};
use ark_serialize::*;
use ark_std::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Display, Formatter, Result as FmtResult},
    io::{Read, Result as IoResult, Write},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

impl<P: Fp256Parameters> Into<BigInteger256> for Fp256<P> {
    fn into(self) -> BigInteger256 {
        self.into_repr()
    }
}

impl<P: Fp256Parameters> core::convert::TryFrom<BigInteger256> for Fp256<P> {
    type Error = crate::fields::arithmetic::InvalidBigInt;

    /// Converts `Self::BigInteger` into `Self`
    ///
    /// This method returns an error if `int` is larger than `P::MODULUS`.
    fn try_from(int: BigInteger256) -> Result<Self, Self::Error> {
        Self::from_repr(int).ok_or(crate::fields::arithmetic::InvalidBigInt)
    }
}


use num_traits::{One, Zero};
pub trait Fp256Parameters: FpParameters<BigInt = BigInteger256> {}
/// Represents an element of the prime field F_p, where `p == P::MODULUS`.
/// This type can represent elements in any field of size at most
///256
/// bits.
// #[derivative(
//     Default(bound = ""),
//     Hash(bound = ""),
//     Clone(bound = ""),
//     Copy(bound = ""),
//     PartialEq(bound = ""),
//     Eq(bound = "")
// )]
pub struct Fp256<P>(
    pub BigInteger256,
    // #[derivative(Debug = "ignore")]
    #[doc(hidden)]
    pub PhantomData<P>,
);
#[allow(unused_qualifications)]
impl<P> ::core::clone::Clone for Fp256<P> {
    fn clone(&self) -> Self {
        match *self {
            Fp256(ref __arg_0, ref __arg_1) => Fp256((*__arg_0).clone(), (*__arg_1).clone()),
        }
    }
}
#[allow(unused_qualifications)]
impl<P> ::core::marker::Copy for Fp256<P> {}
#[allow(unused_qualifications)]
impl<P> ::core::default::Default for Fp256<P> {
    fn default() -> Self {
        Fp256(
            ::core::default::Default::default(),
            ::core::default::Default::default(),
        )
    }
}
#[allow(unused_qualifications)]
impl<P> ::core::cmp::Eq for Fp256<P> {}
#[allow(unused_qualifications)]
impl<P> ::core::hash::Hash for Fp256<P> {
    fn hash<__HP>(&self, __state: &mut __HP)
    where
        __HP: ::core::hash::Hasher,
    {
        match *self {
            Fp256(ref __arg_0, ref __arg_1) => {
                ::core::hash::Hash::hash(&(*__arg_0), __state);
                ::core::hash::Hash::hash(&(*__arg_1), __state);
            },
        }
    }
}
#[allow(unused_qualifications)]
#[allow(clippy::unneeded_field_pattern)]
impl<P> ::core::cmp::PartialEq for Fp256<P> {
    fn eq(&self, other: &Self) -> bool {
        true && match *self {
            Fp256(ref __self_0, ref __self_1) => match *other {
                Fp256(ref __other_0, ref __other_1) => {
                    true && &(*__self_0) == &(*__other_0) && &(*__self_1) == &(*__other_1)
                },
            },
        }
    }
}
impl<P: Fp256Parameters> ark_std::fmt::Debug for Fp256<P> {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        use crate::ark_std::string::ToString;
        let r = self.into_repr();
        let bigint: num_bigint::BigUint = r.into();
        let s = bigint.to_string();
        let name = match P::INV {
            11037532056220336127 => "Fp",
            10108024940646105087 => "Fq",
            _ => "Field",
        };
        f.write_fmt(format_args!("{0}({1})", name, s))
    }
}
impl<P> Fp256<P> {
    #[inline]
    pub const fn new(element: BigInteger256) -> Self {
        Self(element, PhantomData)
    }
    const fn const_is_zero(&self) -> bool {
        let mut is_zero = true;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                is_zero &= (self.0).0[i] == 0;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                is_zero &= (self.0).0[i] == 0;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                is_zero &= (self.0).0[i] == 0;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                is_zero &= (self.0).0[i] == 0;
            }
        }
        is_zero
    }
    const fn const_neg(self, modulus: BigInteger256) -> Self {
        if !self.const_is_zero() {
            Self::new(Self::sub_noborrow(&modulus, &self.0))
        } else {
            self
        }
    }
    /// Interpret a string of decimal numbers as a prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    /// For *internal* use only; please use the `field_new` macro instead
    /// of this method
    #[doc(hidden)]
    pub const fn const_from_str(
        limbs: &[u64],
        is_positive: bool,
        r2: BigInteger256,
        modulus: BigInteger256,
        inv: u64,
    ) -> Self {
        let mut repr = BigInteger256([0; 4]);
        let mut i = 0;
        while i < limbs.len() {
            repr.0[i] = limbs[i];
            i += 1;
        }
        let res = Self::const_from_repr(repr, r2, modulus, inv);
        if is_positive {
            res
        } else {
            res.const_neg(modulus)
        }
    }
    #[inline]
    pub(crate) const fn const_from_repr(
        repr: BigInteger256,
        r2: BigInteger256,
        modulus: BigInteger256,
        inv: u64,
    ) -> Self {
        let mut r = Self::new(repr);
        if r.const_is_zero() {
            r
        } else {
            r = r.const_mul(&Fp256(r2, PhantomData), modulus, inv);
            r
        }
    }
    const fn mul_without_reduce(mut self, other: &Self, modulus: BigInteger256, inv: u64) -> Self {
        let mut r = [0u64; 4 * 2];
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                let mut carry = 0;
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 0usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                let mut carry = 0;
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 0usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                let mut carry = 0;
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 0usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                let mut carry = 0;
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 0usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + ((self.0).0[i] as u128 * (other.0).0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = carry;
            }
        }
        let mut _carry2 = 0;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                let k = r[i].wrapping_mul(inv);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                let k = r[i].wrapping_mul(inv);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                let k = r[i].wrapping_mul(inv);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                let k = r[i].wrapping_mul(inv);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128) + (k as u128 * modulus.0[0] as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + (k as u128 * modulus.0[j] as u128)
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
        }
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                (self.0).0[i] = r[4 + i];
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                (self.0).0[i] = r[4 + i];
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                (self.0).0[i] = r[4 + i];
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                (self.0).0[i] = r[4 + i];
            }
        }
        self
    }
    const fn const_mul(mut self, other: &Self, modulus: BigInteger256, inv: u64) -> Self {
        self = self.mul_without_reduce(other, modulus, inv);
        self.const_reduce(modulus)
    }
    const fn const_is_valid(&self, modulus: BigInteger256) -> bool {
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
                    return true;
                } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
                    return false;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
                    return true;
                } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
                    return false;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
                    return true;
                } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
                    return false;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                if (self.0).0[4 - i - 1] < modulus.0[4 - i - 1] {
                    return true;
                } else if (self.0).0[4 - i - 1] > modulus.0[4 - i - 1] {
                    return false;
                }
            }
        }
        false
    }
    #[inline]
    const fn const_reduce(mut self, modulus: BigInteger256) -> Self {
        if !self.const_is_valid(modulus) {
            self.0 = Self::sub_noborrow(&self.0, &modulus);
        }
        self
    }
    #[allow(unused_assignments)]
    const fn sub_noborrow(a: &BigInteger256, b: &BigInteger256) -> BigInteger256 {
        let mut a = *a;
        let mut borrow = 0;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                a.0[i] = {
                    let tmp =
                        (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
                    borrow = if tmp >> 64 == 0 { 1 } else { 0 };
                    tmp as u64
                };
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                a.0[i] = {
                    let tmp =
                        (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
                    borrow = if tmp >> 64 == 0 { 1 } else { 0 };
                    tmp as u64
                };
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                a.0[i] = {
                    let tmp =
                        (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
                    borrow = if tmp >> 64 == 0 { 1 } else { 0 };
                    tmp as u64
                };
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                a.0[i] = {
                    let tmp =
                        (1u128 << 64) + (a.0[i] as u128) - (b.0[i] as u128) - (borrow as u128);
                    borrow = if tmp >> 64 == 0 { 1 } else { 0 };
                    tmp as u64
                };
            }
        }
        a
    }
}
impl<P: Fp256Parameters> Fp256<P> {
    #[inline(always)]
    pub(crate) fn is_valid(&self) -> bool {
        self.0 < P::MODULUS
    }
    #[inline]
    fn reduce(&mut self) {
        if !self.is_valid() {
            self.0.sub_noborrow(&P::MODULUS);
        }
    }
}
impl<P: Fp256Parameters> Zero for Fp256<P> {
    #[inline]
    fn zero() -> Self {
        Fp256::<P>(BigInteger256::from(0), PhantomData)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<P: Fp256Parameters> One for Fp256<P> {
    #[inline]
    fn one() -> Self {
        Fp256::<P>(P::R, PhantomData)
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.0 == P::R
    }
}
impl<P: Fp256Parameters> Field for Fp256<P> {
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
        P::MODULUS.as_ref()
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
            let last_limb_mask = (u64::MAX >> P::REPR_SHAVE_BITS).to_le_bytes();
            let mut last_bytes_mask = [0u8; 9];
            last_bytes_mask[..8].copy_from_slice(&last_limb_mask);
            let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
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
        if 4 == 1 {
            *self = *self * *self;
            return self;
        }
        let mut r = [0u64; 4 * 2];
        let mut carry = 0;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                if i < 4 - 1 {
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 0usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                    }
                    r[4 + i] = carry;
                    carry = 0;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                if i < 4 - 1 {
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 0usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                    }
                    r[4 + i] = carry;
                    carry = 0;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                if i < 4 - 1 {
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 0usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                    }
                    r[4 + i] = carry;
                    carry = 0;
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                if i < 4 - 1 {
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 0usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            if j > i {
                                r[i + j] = {
                                    let tmp = (r[i + j] as u128)
                                        + crate::biginteger::arithmetic::u64_mul_u64(
                                            (self.0).0[i],
                                            (self.0).0[j],
                                        )
                                        + (carry as u128);
                                    carry = (tmp >> 64) as u64;
                                    tmp as u64
                                };
                            }
                        }
                    }
                    r[4 + i] = carry;
                    carry = 0;
                }
            }
        }
        r[4 * 2 - 1] = r[4 * 2 - 2] >> 63;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                let subtractor = (2 * (4 - 1usize))
                    .checked_sub(i + 1)
                    .map(|index| r[index])
                    .unwrap_or(0);
                r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                let subtractor = (2 * (4 - 1usize))
                    .checked_sub(i + 1)
                    .map(|index| r[index])
                    .unwrap_or(0);
                r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                let subtractor = (2 * (4 - 1usize))
                    .checked_sub(i + 1)
                    .map(|index| r[index])
                    .unwrap_or(0);
                r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                let subtractor = (2 * (4 - 1usize))
                    .checked_sub(i + 1)
                    .map(|index| r[index])
                    .unwrap_or(0);
                r[2 * (4 - 1) - i] = (r[2 * (4 - 1) - i] << 1) | (subtractor >> 63);
            }
        }
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                r[4 + 1 - i] = (r[4 + 1 - i] << 1) | (r[4 - i] >> 63);
            }
        }
        r[1] <<= 1;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                r[2 * i] = {
                    let tmp = (r[2 * i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                #[allow(unused_assignments)]
                {
                    r[2 * i + 1] = {
                        let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
                        carry = (tmp >> 64) as u64;
                        tmp as u64
                    };
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                r[2 * i] = {
                    let tmp = (r[2 * i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                #[allow(unused_assignments)]
                {
                    r[2 * i + 1] = {
                        let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
                        carry = (tmp >> 64) as u64;
                        tmp as u64
                    };
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                r[2 * i] = {
                    let tmp = (r[2 * i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                #[allow(unused_assignments)]
                {
                    r[2 * i + 1] = {
                        let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
                        carry = (tmp >> 64) as u64;
                        tmp as u64
                    };
                }
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                r[2 * i] = {
                    let tmp = (r[2 * i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64((self.0).0[i], (self.0).0[i])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                #[allow(unused_assignments)]
                {
                    r[2 * i + 1] = {
                        let tmp = (r[2 * i + 1] as u128) + (0 as u128) + (carry as u128);
                        carry = (tmp >> 64) as u64;
                        tmp as u64
                    };
                }
            }
        }
        let mut _carry2 = 0;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[j + i] = {
                            let tmp = (r[j + i] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[4 + i] = {
                    let tmp = (r[4 + i] as u128) + (_carry2 as u128) + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                _carry2 = carry;
            }
        }
        (self.0).0.copy_from_slice(&r[4..]);
        self.reduce();
        self
    }
    #[inline]
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let one = BigInteger256::from(1);
            let mut u = self.0;
            let mut v = P::MODULUS;
            let mut b = Fp256::<P>(P::R2, PhantomData);
            let mut c = Self::zero();
            while u != one && v != one {
                while u.is_even() {
                    u.div2();
                    if b.0.is_even() {
                        b.0.div2();
                    } else {
                        b.0.add_nocarry(&P::MODULUS);
                        b.0.div2();
                    }
                }
                while v.is_even() {
                    v.div2();
                    if c.0.is_even() {
                        c.0.div2();
                    } else {
                        c.0.add_nocarry(&P::MODULUS);
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



impl<P: Fp256Parameters> PrimeField for Fp256<P> {
    type Params = P;
    type BigInt = BigInteger256;
    #[inline]
    fn from_repr(r: BigInteger256) -> Option<Self> {
        let mut r = Fp256(r, PhantomData);
        if r.is_zero() {
            Some(r)
        } else if r.is_valid() {
            r *= &Fp256(P::R2, PhantomData);
            Some(r)
        } else {
            None
        }
    }
    #[inline]
    #[allow(clippy::modulo_one)]
    fn into_repr(&self) -> BigInteger256 {
        let mut tmp = self.0;
        let mut r = tmp.0;
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 0usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[i % 4] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[i % 4] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[i % 4] = carry;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                let k = r[i].wrapping_mul(P::INV);
                let mut carry = 0;
                {
                    let tmp = (r[i] as u128)
                        + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[0])
                        + (carry as u128);
                    carry = (tmp >> 64) as u64;
                    tmp as u64
                };
                {
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 1usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 2usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                    {
                        #[allow(non_upper_case_globals)]
                        const j: usize = 3usize;
                        r[(j + i) % 4] = {
                            let tmp = (r[(j + i) % 4] as u128)
                                + crate::biginteger::arithmetic::u64_mul_u64(k, P::MODULUS.0[j])
                                + (carry as u128);
                            carry = (tmp >> 64) as u64;
                            tmp as u64
                        };
                    }
                }
                r[i % 4] = carry;
            }
        }
        tmp.0 = r;
        tmp
    }
}
impl<P: Fp256Parameters> FftField for Fp256<P> {
    type FftParams = P;
    #[inline]
    fn two_adic_root_of_unity() -> Self {
        Fp256::<P>(P::TWO_ADIC_ROOT_OF_UNITY, PhantomData)
    }
    #[inline]
    fn large_subgroup_root_of_unity() -> Option<Self> {
        Some(Fp256::<P>(P::LARGE_SUBGROUP_ROOT_OF_UNITY?, PhantomData))
    }
    #[inline]
    fn multiplicative_generator() -> Self {
        Fp256::<P>(P::GENERATOR, PhantomData)
    }
}
impl<P: Fp256Parameters> SquareRootField for Fp256<P> {
    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        use crate::fields::LegendreSymbol::*;
        let s = self.pow(P::MODULUS_MINUS_ONE_DIV_TWO);
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
            if self.is_zero() {
                return Some(Self::zero());
            }
            let mut z = Self::qnr_to_t();
            let mut w = self.pow(P::T_MINUS_ONE_DIV_TWO);
            let mut x = w * self;
            let mut b = x * &w;
            let mut v = P::TWO_ADICITY as usize;
            while !b.is_one() {
                let mut k = 0usize;
                let mut b2k = b;
                while !b2k.is_one() {
                    b2k.square_in_place();
                    k += 1;
                }
                if k == (P::TWO_ADICITY as usize) {
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
/// Note that this implementation of `Ord` compares field elements viewing
/// them as integers in the range 0, 1, ..., P::MODULUS - 1. However, other
/// implementations of `PrimeField` might choose a different ordering, and
/// as such, users should use this `Ord` for applications where
/// any ordering suffices (like in a BTreeMap), and not in applications
/// where a particular ordering is required.
impl<P: Fp256Parameters> Ord for Fp256<P> {
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
impl<P: Fp256Parameters> PartialOrd for Fp256<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<P: Fp256Parameters> From<u128> for Fp256<P> {
    fn from(other: u128) -> Self {
        let mut default_int = P::BigInt::default();
        if 4 == 1 {
            default_int.0[0] = (other % u128::from(P::MODULUS.0[0])) as u64;
        } else {
            let upper = (other >> 64) as u64;
            let lower = ((other << 64) >> 64) as u64;
            let limbs = [lower, upper];
            for (cur, other) in default_int.0.iter_mut().zip(&limbs) {
                *cur = *other;
            }
        }
        Self::from_repr(default_int).unwrap()
    }
}
impl<P: Fp256Parameters> From<i128> for Fp256<P> {
    fn from(other: i128) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<P: Fp256Parameters> From<u64> for Fp256<P> {
    fn from(other: u64) -> Self {
        if 4 == 1 {
            Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
        } else {
            Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
        }
    }
}
impl<P: Fp256Parameters> From<i64> for Fp256<P> {
    fn from(other: i64) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<P: Fp256Parameters> From<u32> for Fp256<P> {
    fn from(other: u32) -> Self {
        if 4 == 1 {
            Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
        } else {
            Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
        }
    }
}
impl<P: Fp256Parameters> From<i32> for Fp256<P> {
    fn from(other: i32) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<P: Fp256Parameters> From<u16> for Fp256<P> {
    fn from(other: u16) -> Self {
        if 4 == 1 {
            Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
        } else {
            Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
        }
    }
}
impl<P: Fp256Parameters> From<i16> for Fp256<P> {
    fn from(other: i16) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<P: Fp256Parameters> From<u8> for Fp256<P> {
    fn from(other: u8) -> Self {
        if 4 == 1 {
            Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
        } else {
            Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
        }
    }
}
impl<P: Fp256Parameters> From<i8> for Fp256<P> {
    fn from(other: i8) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}
impl<P: Fp256Parameters> From<bool> for Fp256<P> {
    fn from(other: bool) -> Self {
        if 4 == 1 {
            Self::from_repr(P::BigInt::from(u64::from(other) % P::MODULUS.0[0])).unwrap()
        } else {
            Self::from_repr(P::BigInt::from(u64::from(other))).unwrap()
        }
    }
}
impl<P: Fp256Parameters> ark_std::rand::distributions::Distribution<Fp256<P>>
    for ark_std::rand::distributions::Standard
{
    #[inline]
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp256<P> {
        loop {
            let mut tmp = Fp256(
                rng.sample(ark_std::rand::distributions::Standard),
                PhantomData,
            );
            if !(P::REPR_SHAVE_BITS <= 64) {
                panic!("assertion failed: P::REPR_SHAVE_BITS <= 64")
            }
            let mask = if P::REPR_SHAVE_BITS == 64 {
                0
            } else {
                core::u64::MAX >> P::REPR_SHAVE_BITS
            };
            tmp.0.as_mut().last_mut().map(|val| *val &= mask);
            if tmp.is_valid() {
                return tmp;
            }
        }
    }
}
impl<P: Fp256Parameters> CanonicalSerializeWithFlags for Fp256<P> {
    fn serialize_with_flags<W: ark_std::io::Write, F: Flags>(
        &self,
        mut writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
        let mut bytes = [0u8; 4 * 8 + 1];
        self.write(&mut bytes[..4 * 8])?;
        bytes[output_byte_size - 1] |= flags.u8_bitmask();
        writer.write_all(&bytes[..output_byte_size])?;
        Ok(())
    }
    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE)
    }
}
impl<P: Fp256Parameters> CanonicalSerialize for Fp256<P> {
    #[inline]
    fn serialize<W: ark_std::io::Write>(&self, writer: W) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }
    #[inline]
    fn serialized_size(&self) -> usize {
        self.serialized_size_with_flags::<EmptyFlags>()
    }
}
impl<P: Fp256Parameters> CanonicalDeserializeWithFlags for Fp256<P> {
    fn deserialize_with_flags<R: ark_std::io::Read, F: Flags>(
        mut reader: R,
    ) -> Result<(Self, F), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let output_byte_size = buffer_byte_size(P::MODULUS_BITS as usize + F::BIT_SIZE);
        let mut masked_bytes = [0; 4 * 8 + 1];
        reader.read_exact(&mut masked_bytes[..output_byte_size])?;
        let flags = F::from_u8_remove_flags(&mut masked_bytes[output_byte_size - 1])
            .ok_or(SerializationError::UnexpectedFlags)?;
        Ok((Self::read(&masked_bytes[..])?, flags))
    }
}
impl<P: Fp256Parameters> CanonicalDeserialize for Fp256<P> {
    fn deserialize<R: ark_std::io::Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}
impl<P: Fp256Parameters> ToBytes for Fp256<P> {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.into_repr().write(writer)
    }
}
impl<P: Fp256Parameters> FromBytes for Fp256<P> {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger256::read(reader).and_then(|b| match Fp256::from_repr(b) {
            Some(f) => Ok(f),
            None => Err(crate::error("FromBytes::read failed")),
        })
    }
}
impl<P: Fp256Parameters> FromStr for Fp256<P> {
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
/// Outputs a string containing the value of `self`, chunked up into
/// 64-bit limbs.
impl<P: Fp256Parameters> Display for Fp256<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("Fp256 \"({0})\"", self.into_repr()))
    }
}
impl<P: Fp256Parameters> Neg for Fp256<P> {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        if !self.is_zero() {
            let mut tmp = P::MODULUS;
            tmp.sub_noborrow(&self.0);
            Fp256::<P>(tmp, PhantomData)
        } else {
            self
        }
    }
}
impl<'a, P: Fp256Parameters> Add<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}
impl<'a, P: Fp256Parameters> Sub<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}
impl<'a, P: Fp256Parameters> Mul<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}
impl<'a, P: Fp256Parameters> Div<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;
    /// Returns `self * other.inverse()` if `other.inverse()` is `Some`, and
    /// panics otherwise.
    #[inline]
    fn div(mut self, other: &Self) -> Self {
        self.mul_assign(&other.inverse().unwrap());
        self
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::Add<Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut result = self;
        result.add_assign(&other);
        result
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::Add<&'a mut Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn add(self, other: &'a mut Self) -> Self {
        let mut result = self;
        result.add_assign(&*other);
        result
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::Sub<Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut result = self;
        result.sub_assign(&other);
        result
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::Sub<&'a mut Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn sub(self, other: &'a mut Self) -> Self {
        let mut result = self;
        result.sub_assign(&*other);
        result
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::iter::Sum<Self> for Fp256<P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::iter::Sum<&'a Self> for Fp256<P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::AddAssign<Self> for Fp256<P> {
    fn add_assign(&mut self, other: Self) {
        self.add_assign(&other)
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::SubAssign<Self> for Fp256<P> {
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::AddAssign<&'a mut Self> for Fp256<P> {
    fn add_assign(&mut self, other: &'a mut Self) {
        self.add_assign(&*other)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::SubAssign<&'a mut Self> for Fp256<P> {
    fn sub_assign(&mut self, other: &'a mut Self) {
        self.sub_assign(&*other)
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::Mul<Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut result = self;
        result.mul_assign(&other);
        result
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::Div<Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        let mut result = self;
        result.div_assign(&other);
        result
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::Mul<&'a mut Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn mul(self, other: &'a mut Self) -> Self {
        let mut result = self;
        result.mul_assign(&*other);
        result
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::Div<&'a mut Self> for Fp256<P> {
    type Output = Self;
    #[inline]
    fn div(self, other: &'a mut Self) -> Self {
        let mut result = self;
        result.div_assign(&*other);
        result
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::iter::Product<Self> for Fp256<P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), core::ops::Mul::mul)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::iter::Product<&'a Self> for Fp256<P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::MulAssign<Self> for Fp256<P> {
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign(&other)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::DivAssign<&'a mut Self> for Fp256<P> {
    fn div_assign(&mut self, other: &'a mut Self) {
        self.div_assign(&*other)
    }
}
#[allow(unused_qualifications)]
impl<'a, P: Fp256Parameters> core::ops::MulAssign<&'a mut Self> for Fp256<P> {
    fn mul_assign(&mut self, other: &'a mut Self) {
        self.mul_assign(&*other)
    }
}
#[allow(unused_qualifications)]
impl<P: Fp256Parameters> core::ops::DivAssign<Self> for Fp256<P> {
    fn div_assign(&mut self, other: Self) {
        self.div_assign(&other)
    }
}
impl<'a, P: Fp256Parameters> AddAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.0.add_nocarry(&other.0);
        self.reduce();
    }
}
impl<'a, P: Fp256Parameters> SubAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        if other.0 > self.0 {
            self.0.add_nocarry(&P::MODULUS);
        }
        self.0.sub_noborrow(&other.0);
    }
}
impl<'a, P: Fp256Parameters> MulAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        let first_bit_set = P::MODULUS.0[4 - 1] >> 63 != 0;
        #[allow(unused_mut)]
        let mut all_bits_set = P::MODULUS.0[4 - 1] == !0 - (1 << 63);
        {
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 1usize;
                all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 2usize;
                all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
            }
            {
                #[allow(non_upper_case_globals)]
                const i: usize = 3usize;
                all_bits_set &= P::MODULUS.0[4 - i - 1] == !0u64;
            }
        }
        let _no_carry: bool = !(first_bit_set || all_bits_set);
        if _no_carry {
            let mut r = [0u64; 4];
            let mut carry1 = 0u64;
            let mut carry2 = 0u64;
            {
                {
                    #[allow(non_upper_case_globals)]
                    const i: usize = 0usize;
                    r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
                    let k = r[0].wrapping_mul(P::INV);
                    fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                    }
                    r[4 - 1] = carry1 + carry2;
                }
                {
                    #[allow(non_upper_case_globals)]
                    const i: usize = 1usize;
                    r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
                    let k = r[0].wrapping_mul(P::INV);
                    fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                    }
                    r[4 - 1] = carry1 + carry2;
                }
                {
                    #[allow(non_upper_case_globals)]
                    const i: usize = 2usize;
                    r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
                    let k = r[0].wrapping_mul(P::INV);
                    fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                    }
                    r[4 - 1] = carry1 + carry2;
                }
                {
                    #[allow(non_upper_case_globals)]
                    const i: usize = 3usize;
                    r[0] = fa::mac(r[0], (self.0).0[0], (other.0).0[i], &mut carry1);
                    let k = r[0].wrapping_mul(P::INV);
                    fa::mac_discard(r[0], k, P::MODULUS.0[0], &mut carry2);
                    {
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 1usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 2usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                        {
                            #[allow(non_upper_case_globals)]
                            const j: usize = 3usize;
                            r[j] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        (self.0).0[j],
                                        (other.0).0[i],
                                    )
                                    + (carry1 as u128);
                                carry1 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                            r[j - 1] = {
                                let tmp = (r[j] as u128)
                                    + crate::biginteger::arithmetic::u64_mul_u64(
                                        k,
                                        P::MODULUS.0[j],
                                    )
                                    + (carry2 as u128);
                                carry2 = (tmp >> 64) as u64;
                                tmp as u64
                            };
                        }
                    }
                    r[4 - 1] = carry1 + carry2;
                }
            }
            (self.0).0 = r;
            self.reduce();
        } else {
            *self = self.mul_without_reduce(other, P::MODULUS, P::INV);
            self.reduce();
        }
    }
}
/// Computes `self *= other.inverse()` if `other.inverse()` is `Some`, and
/// panics otherwise.
impl<'a, P: Fp256Parameters> DivAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}
impl<P: Fp256Parameters> zeroize::Zeroize for Fp256<P> {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}
impl<P: Fp256Parameters> From<num_bigint::BigUint> for Fp256<P> {
    #[inline]
    fn from(val: num_bigint::BigUint) -> Fp256<P> {
        Fp256::<P>::from_le_bytes_mod_order(&val.to_bytes_le())
    }
}
impl<P: Fp256Parameters> Into<num_bigint::BigUint> for Fp256<P> {
    #[inline]
    fn into(self) -> num_bigint::BigUint {
        self.into_repr().into()
    }
}
