//! Robust Linear Output Regulation Problem (RLORP) functionality/solutions

use ndarray::{concatenate, s, Array1, Array2, Axis, LinalgScalar};
use ndarray_linalg::Lapack;

use crate::{control_theory::lqr, dynamics::Dynamics, lorp::LorpDynamics, LtiDynamics};

/// Implement linear, time-invariant RLORP dynamics for convenience
///
/// $$ \dot{x} = (A + \Delta A) x + (B + \Delta B) u + (E + \Delta E) v $$
/// $$ e = (C + \Delta C) x + (D + \Delta D) u + (F + \Delta F) v $$
/// $$ \dot{v} = S v
pub struct RlorpDynamics<T: LinalgScalar> {
    pub lorp_dynamics: LorpDynamics<T>,
    pub delta_a: Array2<T>,
    pub delta_b: Array2<T>,
    pub delta_c: Array2<T>,
    pub delta_d: Array2<T>,
    pub delta_e: Array2<T>,
    pub delta_f: Array2<T>,
    a_tilde: Array2<T>,
    b_tilde: Array2<T>,
    c_tilde: Array2<T>,
    d_tilde: Array2<T>,
}
impl<T: LinalgScalar> RlorpDynamics<T> {
    pub fn new(
        lorp_dynamics: LorpDynamics<T>,
        delta_a: Array2<T>,
        delta_b: Array2<T>,
        delta_c: Array2<T>,
        delta_d: Array2<T>,
        delta_e: Array2<T>,
        delta_f: Array2<T>,
    ) -> Self {
        let a_tilde = concatenate![
            Axis(0),
            concatenate![
                Axis(1),
                (&lorp_dynamics.a_mat + &delta_a),
                (&lorp_dynamics.e_mat + &delta_e)
            ],
            concatenate![
                Axis(1),
                Array2::zeros((lorp_dynamics.s_mat.nrows(), lorp_dynamics.a_mat.ncols())),
                lorp_dynamics.s_mat
            ]
        ];
        let b_tilde = concatenate![
            Axis(0),
            (&lorp_dynamics.b_mat + &delta_b),
            Array2::zeros((lorp_dynamics.s_mat.nrows(), lorp_dynamics.b_mat.ncols()))
        ];
        let c_tilde = concatenate![
            Axis(1),
            (&lorp_dynamics.c_mat + &delta_c),
            (&lorp_dynamics.f_mat + &delta_f)
        ];
        let d_tilde = &lorp_dynamics.d_mat + &delta_d;
        RlorpDynamics {
            lorp_dynamics,
            delta_a,
            delta_b,
            delta_c,
            delta_d,
            delta_e,
            delta_f,
            a_tilde,
            b_tilde,
            c_tilde,
            d_tilde,
        }
    }

    fn a_tilde(&self) -> &Array2<T> {
        &self.a_tilde
    }

    fn b_tilde(&self) -> &Array2<T> {
        &self.b_tilde
    }

    fn c_tilde(&self) -> &Array2<T> {
        &self.c_tilde
    }

    fn d_tilde(&self) -> &Array2<T> {
        &self.d_tilde
    }
}
impl<T: LinalgScalar> Dynamics<T> for RlorpDynamics<T> {
    fn n_input(&self) -> usize {
        self.b_tilde.ncols()
    }
    fn n_state(&self) -> usize {
        self.a_tilde.ncols()
    }
    fn n_output(&self) -> usize {
        self.c_tilde.nrows()
    }
    fn dynamics(&self, _t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
        self.a_tilde.dot(x) + self.b_tilde.dot(u)
    }
    fn output(&self, _t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
        self.c_tilde.dot(x) + self.d_tilde.dot(u)
    }
}

/// Create a block diagonal matrix.
fn block_diag<T: LinalgScalar>(blocks: &[Array2<T>]) -> Array2<T> {
    let rows: Vec<_> = blocks.iter().map(|v| v.nrows()).collect();
    let cols: Vec<_> = blocks.iter().map(|v| v.ncols()).collect();
    let total_rows = rows.iter().sum();
    let total_cols = cols.iter().sum();

    let mut out_arr = Array2::zeros((total_rows, total_cols));

    let mut row_start = 0;
    let mut col_start = 0;
    for (i, block) in blocks.iter().enumerate() {
        let nrow = rows[i];
        let ncol = cols[i];
        out_arr
            .slice_mut(s![row_start..row_start + nrow, col_start..col_start + ncol])
            .assign(block);
        row_start += nrow;
        col_start += ncol;
    }

    out_arr
}

/// Create a p-copy internal model from the given minimum polynomial
fn p_copy_internal_model<T: LinalgScalar + Lapack>(
    minimal_polynomial: &[T],
    n_p: usize,
) -> (Array2<T>, Array2<T>) {
    let n = minimal_polynomial.len() - 1;
    let beta = concatenate![
        Axis(0),
        concatenate![Axis(1), Array2::zeros((n - 1, 1)), Array2::eye(n - 1)],
        -Array2::from_shape_vec(
            (1, n),
            minimal_polynomial.iter().copied().skip(1).rev().collect()
        )
        .unwrap()
    ];
    let sigma = Array2::from_shape_fn(
        (n, 1),
        |(i, _j)| if i < n - 1 { T::zero() } else { T::one() },
    );

    let v1_mat = block_diag(&std::iter::repeat(beta).take(n_p).collect::<Vec<_>>());
    let v2_mat = block_diag(&std::iter::repeat(sigma).take(n_p).collect::<Vec<_>>());
    (v1_mat, v2_mat)
}

/// Create the dynamic state feedback controller solving the RLORP
///
/// Creates an augmented dynamic system and the feedback input function based on
/// the augmented system.
/// The feedback has the form $u = K_1 x + K_2 z$ with $\dot{z} = G_1 z + G_2 e$
/// with $K_1$, $K_2$, $G_1$, and $G_2$ being the design parameters.
///
/// The $G_1$, $G_2$ are constructed to incorporate an internal model of $S$,
/// While $K_1$ and $K_2$ are constructed to stabilize the system
/// $$
///     \dot{x} = \begin{bmatrix} A & 0 \\ G_2C & G_1 \end{bmatrix}
///     + \begin{bmatrix} B \\ G_2D \end{bmatrix} u
/// $$
/// $$
///     u = \begin{bmatrix} K_1 & K_2 \end{bmatrix} x
/// $$
pub fn dynamic_state_feedback_controller_rlorp<
    T: LinalgScalar + Lapack + ndarray::ScalarOperand + std::cmp::PartialOrd,
>(
    rlorp_dynamics: &RlorpDynamics<T>,
    min_poly_s: &[T],
    q_mat: Option<&Array2<T>>,
    r_mat: Option<&Array2<T>>,
) -> (LtiDynamics<T>, impl Fn(T, &Array1<T>) -> Array1<T>) {
    let (g1_mat, g2_mat) =
        p_copy_internal_model(min_poly_s, rlorp_dynamics.lorp_dynamics.c_mat.nrows());

    let a_bar = concatenate![
        Axis(0),
        concatenate![
            Axis(1),
            rlorp_dynamics.lorp_dynamics.a_mat,
            Array2::zeros((rlorp_dynamics.lorp_dynamics.a_mat.nrows(), g1_mat.ncols()))
        ],
        concatenate![
            Axis(1),
            g2_mat.dot(&rlorp_dynamics.lorp_dynamics.c_mat),
            g1_mat
        ]
    ];
    let b_bar = concatenate![
        Axis(0),
        rlorp_dynamics.lorp_dynamics.b_mat,
        g2_mat.dot(&rlorp_dynamics.lorp_dynamics.d_mat)
    ];
    let k_mat = -lqr(
        &a_bar,
        &b_bar,
        q_mat,
        r_mat,
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap();


    let k_mat_aug = concatenate![
        Axis(1),
        k_mat.slice(s![.., 0..rlorp_dynamics.lorp_dynamics.a_mat.ncols()]),
        Array2::zeros((k_mat.nrows(), rlorp_dynamics.lorp_dynamics.s_mat.ncols())),
        k_mat.slice(s![.., rlorp_dynamics.lorp_dynamics.a_mat.ncols()..])
    ];
    println!("{k_mat_aug}");
    (
        augmented_system_dynamic_state_feedback(rlorp_dynamics, &g1_mat, &g2_mat),
        move |_t, x| k_mat_aug.dot(x),
    )
}

pub fn augmented_system_dynamic_state_feedback<T: LinalgScalar>(rlorp_dynamics: &RlorpDynamics<T>, g1_mat: &Array2<T>, g2_mat: &Array2<T>) -> LtiDynamics<T> {
    let a_aug = concatenate![
        Axis(0),
        concatenate![
            Axis(1),
            *rlorp_dynamics.a_tilde(),
            Array2::zeros((rlorp_dynamics.n_state(), g1_mat.ncols()))
        ],
        concatenate![
            Axis(1),
            concatenate![
                Axis(1),
                g2_mat.dot(&(&rlorp_dynamics.lorp_dynamics.c_mat + &rlorp_dynamics.delta_c)),
                g2_mat.dot(&(&rlorp_dynamics.lorp_dynamics.f_mat + &rlorp_dynamics.delta_f))
            ],
            *g1_mat
        ]
    ];
    let b_aug = concatenate![
        Axis(0),
        *rlorp_dynamics.b_tilde(),
        g2_mat.dot(&(&rlorp_dynamics.lorp_dynamics.d_mat + &rlorp_dynamics.delta_d))
    ];
    let c_aug = concatenate![
        Axis(1),
        *rlorp_dynamics.c_tilde(),
        Array2::zeros((rlorp_dynamics.c_tilde().nrows(), g1_mat.ncols()))
    ];
    let d_aug = rlorp_dynamics.d_tilde().clone();
    LtiDynamics {a_mat: a_aug, b_mat: b_aug, c_mat: c_aug, d_mat: d_aug}
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_dynamic_state_feedback_controller_rlorp() {
        let a_mat = array![
            [-0.0158, 0.02633, -9.81, 0.],
            [-0.1531, -1.03, 0., 120.5],
            [0., 0., 0., 1.],
            [0.0005274, -0.01652, 0., -1.466]
        ];
        let b_mat = array![[0.0006056, 0.], [0., -9.496], [0., 0.], [0., -5.565]];
        let c_mat = array![[1., 0., 0., 0.]];
        let d_mat = array![[0., 0.]];
        let e_mat = array![
            [0., 0., 0.],
            [0., 0., -9.496],
            [0., 0., 0.],
            [0., 0., -5.565]
        ];
        let f_mat = array![[-1., 0., 0.]];
        let s_mat = array![[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]];
        let delta_a = Array2::zeros(a_mat.raw_dim());
        let delta_b = Array2::zeros(b_mat.raw_dim());
        let delta_c = Array2::zeros(c_mat.raw_dim());
        let delta_d = Array2::zeros(d_mat.raw_dim());
        let delta_e = Array2::zeros(e_mat.raw_dim());
        let delta_f = Array2::zeros(f_mat.raw_dim());
        let lorp_dynamics = LorpDynamics::new(a_mat, b_mat, c_mat, d_mat, e_mat, f_mat, s_mat);
        let rlorp_dynamics = RlorpDynamics::new(
            lorp_dynamics,
            delta_a,
            delta_b,
            delta_c,
            delta_d,
            delta_e,
            delta_f,
        );

        let min_poly_s = [1., 0., 1., 0.];

        let (rlorp_dyn, rlorp_ctrl) =
            dynamic_state_feedback_controller_rlorp(&rlorp_dynamics, &min_poly_s, None, None);

        assert!(rlorp_dyn.a_mat.abs_diff_eq(
            &array![
                [-1.58e-2, 2.633e-2, -9.81, 0., 0., 0., 0., 0., 0., 0.],
                [-1.531e-1, -1.03, 0., 1.205e2, 0., 0., -9.496, 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                [5.274e-4, -1.652e-2, 0., -1.466, 0., 0., -5.565, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., -1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., -1., 0., 0., 0., -1., 0.]
            ],
            1e-12
        ));
        assert!(rlorp_dyn.b_mat.abs_diff_eq(
            &array![
                [6.056e-4, 0.],
                [0., -9.496],
                [0., 0.],
                [0., -5.565],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.]
            ],
            1e-12
        ));
        assert!(rlorp_dyn.c_mat.abs_diff_eq(&array![[1., 0., 0., 0., -1., 0., 0., 0., 0., 0.]], 1e-12));
        assert!(rlorp_dyn.d_mat.abs_diff_eq(&array![[0., 0.]], 1e-12));

        let eye10 = Array2::eye(10);
        assert!(rlorp_ctrl(0., &eye10.slice(s![0, ..]).to_owned()).abs_diff_eq(&array![-3.513697633313342e-2, -6.335195984931541], 1e-6));
        assert!(rlorp_ctrl(0., &eye10.slice(s![1, ..]).to_owned()).abs_diff_eq(&array![-1.95060633838762e-3, 2.541285622352293e-1], 1e-7));
        assert!(rlorp_ctrl(0., &eye10.slice(s![2, ..]).to_owned()).abs_diff_eq(&array![2.646935642526326e-1, 9.260870385043025e1], 1e-5));
        assert!(rlorp_ctrl(0., &eye10.slice(s![3, ..]).to_owned()).abs_diff_eq(&array![4.017889034646774e-3, 6.243495025478761], 1e-7));
        assert!(rlorp_ctrl(0., &eye10.slice(s![4, ..]).to_owned()).abs_diff_eq(&array![0., 0.], 1e-12));
        assert!(rlorp_ctrl(0., &eye10.slice(s![5, ..]).to_owned()).abs_diff_eq(&array![0., 0.], 1e-12));
        assert!(rlorp_ctrl(0., &eye10.slice(s![6, ..]).to_owned()).abs_diff_eq(&array![0., 0.], 1e-12));
        assert!(rlorp_ctrl(0., &eye10.slice(s![7, ..]).to_owned()).abs_diff_eq(&array![-5.717006212533678e-3, -9.999836577864565e-1], 1e-7));
        assert!(rlorp_ctrl(0., &eye10.slice(s![8, ..]).to_owned()).abs_diff_eq(&array![1.191185788711509e-2, 1.717901328143854], 1e-7));
        assert!(rlorp_ctrl(0., &eye10.slice(s![9, ..]).to_owned()).abs_diff_eq(&array![-1.270628456329595e-2, -1.220492805876535], 1e-7));
    }
}
