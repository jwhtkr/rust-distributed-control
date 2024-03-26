//! Linear Output Regulation Problem solutions/functionality.

use ndarray::{
    concatenate, linalg::kron, s, Array1, Array2, ArrayBase, Axis, Data, Ix2, LinalgScalar,
    ScalarOperand, ShapeBuilder,
};
use ndarray_linalg::{error::LinalgError, Lapack, Solve};

use crate::{control_theory::lqr, dynamics::Dynamics, LtiDynamics};

/// Implement linear, time-invariant LORP dynamics for convenience
///
/// $$ \dot{x} = A x + B u + E v $$
/// $$ e = C x + D u + F v $$
/// $$ \dot{v} = S v
pub struct LorpDynamics<T: LinalgScalar> {
    pub a_mat: Array2<T>,
    pub b_mat: Array2<T>,
    pub c_mat: Array2<T>,
    pub d_mat: Array2<T>,
    pub e_mat: Array2<T>,
    pub f_mat: Array2<T>,
    pub s_mat: Array2<T>,
    a_tilde: Array2<T>,
    b_tilde: Array2<T>,
    c_tilde: Array2<T>,
    d_tilde: Array2<T>,
}
impl<T: LinalgScalar> LorpDynamics<T> {
    pub fn new(
        a_mat: Array2<T>,
        b_mat: Array2<T>,
        c_mat: Array2<T>,
        d_mat: Array2<T>,
        e_mat: Array2<T>,
        f_mat: Array2<T>,
        s_mat: Array2<T>,
    ) -> Self {
        let a_tilde = concatenate![
            Axis(0),
            concatenate![Axis(1), a_mat, e_mat],
            concatenate![
                Axis(1),
                Array2::zeros((s_mat.nrows(), a_mat.ncols())),
                s_mat
            ]
        ];
        let b_tilde = concatenate![
            Axis(0),
            b_mat,
            Array2::zeros((s_mat.nrows(), b_mat.ncols()))
        ];
        let c_tilde = concatenate![Axis(1), c_mat, f_mat];
        let d_tilde = d_mat.clone();
        LorpDynamics {
            a_mat,
            b_mat,
            c_mat,
            d_mat,
            e_mat,
            f_mat,
            s_mat,
            a_tilde,
            b_tilde,
            c_tilde,
            d_tilde,
        }
    }
}
impl<T: LinalgScalar> Dynamics<T> for LorpDynamics<T> {
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

/// The regulator equations for a set of matrices $A, B, E, C, D, F, S$.
///
/// The regulator equations correspond to the equations
/// $$ \dot{x} = A x + B u + E v $$
/// $$ e = C x + B u + F v $$
/// with the state $x\in\mathbb{R}^n$, the input $u\in\mathbb{R}^m$,
/// the measurement error $e\in\mathbb{R}^p$, and an exosystem
/// $v\in\mathbb{R}^q$ such that $\dot{v}=Sv$ for $S\in\mathbb{R}^{q\times q}$
/// with $\operatorname{spec}(S)\subseteq CRHP$.
pub struct RegulatorEquations<'a, T: LinalgScalar> {
    pub lorp_dyn: &'a LorpDynamics<T>,
    x_mat: Array2<T>,
    u_mat: Array2<T>,
}

impl<'a, T> RegulatorEquations<'a, T>
where
    T: LinalgScalar + Lapack,
{
    pub fn new(lorp_dyn: &'a LorpDynamics<T>) -> Self {
        // TODO: Add solvability check here.
        let (x_mat, u_mat) = RegulatorEquations::solve(lorp_dyn);

        RegulatorEquations {
            lorp_dyn,
            x_mat,
            u_mat,
        }
    }

    pub fn x_mat(&self) -> &Array2<T> {
        &self.x_mat
    }

    pub fn u_mat(&self) -> &Array2<T> {
        &self.u_mat
    }

    /// Solve the represented regulator equations and store in $X$ and $U$
    fn solve(lorp_dyn: &LorpDynamics<T>) -> (Array2<T>, Array2<T>) {
        let lme_a = lorp_dyn.s_mat.view();
        let lme_b = concatenate![
            Axis(0),
            concatenate![Axis(1), lorp_dyn.a_mat, lorp_dyn.b_mat],
            concatenate![Axis(1), lorp_dyn.c_mat, lorp_dyn.d_mat]
        ];
        let lme_c = concatenate![Axis(0), lorp_dyn.e_mat, lorp_dyn.f_mat];
        let lme_m = concatenate![
            Axis(0),
            concatenate![
                Axis(1),
                Array2::eye(lorp_dyn.a_mat.ncols()),
                Array2::zeros(lorp_dyn.b_mat.raw_dim())
            ],
            Array2::zeros((
                lorp_dyn.n_output(),
                lorp_dyn.a_mat.ncols() + lorp_dyn.b_mat.ncols()
            ))
        ];

        let x_u_mat = solve_lme(
            &lme_a,
            &lme_b.view(),
            &lme_c.view(),
            Some(&lme_m),
            Default::default(),
        )
        .unwrap();
        (
            x_u_mat.slice(s![0..lorp_dyn.a_mat.ncols(), ..]).to_owned(),
            x_u_mat.slice(s![lorp_dyn.a_mat.ncols().., ..]).to_owned(),
        )
    }
}

/// Solve a linear matrix equation, $MXA - BXQ = C$, for the matrix $X$.
///
/// This is solved using the vectorization and its relationship to the kronecker
/// product as the solution to the matrix equation
/// $$
/// (A^\text{T} \otimes M - Q^\text{T} \otimes B) \operatorname{vec}(X) = \operatorname{vec}(C).
/// $$
///
/// Note that the LME is not checked for the existence of a solution before
/// solving the matrix equation; this is the duty of the caller.
///
/// Also note that when $M$ and $Q$ are None, or default, they are assumed to be
/// appropriately sized identity matrices, and the resulting LME is a Sylvester
/// equation. If $B=-A^\text{T}$ in addition to $M$ and $Q$ identity matrices,
/// with $C$ negative definite, then the resulting LME is a Lyapunov equation.
pub fn solve_lme<T: LinalgScalar + Lapack, S: Data<Elem = T>>(
    a_mat: &ArrayBase<S, Ix2>,
    b_mat: &ArrayBase<S, Ix2>,
    c_mat: &ArrayBase<S, Ix2>,
    m_mat: Option<&Array2<T>>,
    q_mat: Option<&Array2<T>>,
) -> Result<Array2<T>, LinalgError> {
    let mut _m_mat;
    let m_mat = match m_mat {
        Some(mat) => mat,
        None => {
            _m_mat = Array2::zeros(b_mat.raw_dim());
            _m_mat.diag_mut().fill(T::one());
            &_m_mat
        }
    };
    let mut _q_mat;
    let q_mat = match q_mat {
        Some(mat) => mat,
        None => {
            _q_mat = Array2::zeros(a_mat.raw_dim());
            _q_mat.diag_mut().fill(T::one());
            &_q_mat
        }
    };
    let term_1 = kron(&a_mat.t(), m_mat);
    let term_2 = kron(&q_mat.t(), b_mat);
    let a_bar: Array2<T> = term_1 - term_2;
    let c_mat_t = c_mat.t();
    let c_bar = c_mat_t
        .as_standard_layout()
        .into_shape(b_mat.nrows() * a_mat.ncols())?;
    let vec_res = Solve::solve(&a_bar, &c_bar)?;
    let res = Array2::from_shape_vec(
        (b_mat.nrows(), a_mat.ncols()).f(),
        vec_res.as_slice().unwrap().to_vec(),
    )?;
    Ok(res)
}

/// Create the static state feedback control law that solves the LORP
///
/// For the given $A, B, E, C, D, F, S$, the control law is of the following
/// form
/// $$ u = K_1 x + K_2 v.$$
/// This control law solves the LORP when: $K_1$ is selected such that
/// $A - B K_1$ is Hurwitz, and $K_2 = U - K_1 X$ with $X,U$ the solution to the
/// corresponding regulator equations ([`RegulatorEquations`]).
pub fn static_state_feedback<T>(lorp_dyn: &LorpDynamics<T>) -> impl Fn(T, &Array1<T>) -> Array1<T>
where
    T: LinalgScalar + Lapack + ScalarOperand + std::cmp::PartialOrd,
{
    let k_mat = static_state_feedback_k(lorp_dyn);

    move |_t, x| k_mat.dot(x)
}

fn static_state_feedback_k<T>(lorp_dyn: &LorpDynamics<T>) -> Array2<T>
where
    T: LinalgScalar + Lapack + ScalarOperand + std::cmp::PartialOrd,
{
    let reg_eqs = RegulatorEquations::new(lorp_dyn);

    let k1_mat = lqr(
        &lorp_dyn.a_mat,
        &lorp_dyn.b_mat,
        &Array2::eye(lorp_dyn.a_mat.ncols()),
        &Array2::eye(lorp_dyn.b_mat.ncols()),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap();
    let k2_mat = reg_eqs.u_mat() - k1_mat.dot(reg_eqs.x_mat());
    concatenate![Axis(1), k1_mat, k2_mat]
}

/// Create the dynamic output feedback control law that solves the LORP
///
/// For the given $A, B, E, C, D, F, C_m, D_m, F_m, S$ the control law is of the
/// following form
/// $$ u = K z$$
/// $$ \dot{z} = G_1 z + G_2 y_m.$$
/// Note that $y_m = C_m x + D_m u + F_m v$.
/// This control law solves the LORP when 1) $K=[K_1 K_2]$ (corresponding to the
/// dimensionality of $x$ and $v$ respectively) such that $A - B K_1$ is
/// Hurwitz and $K_2 = U - K_1 X$ with ($X,U$) the solution to the corresponding
/// regulator equations, 2) $G_2$ is selected such that
/// $$\begin{bmatrix}
///     A & E \\ 0 & S
/// \end{bmatrix}
/// - G2 \begin{bmatrix} C_m & F_m \end{bmatrix}
/// $$
/// is Hurwitz, and 3) $G_1$ is calculated as
/// $$
/// G_1 = \begin{bmatrix} A & E \\ 0 & S \end{bmatrix}
///     + \begin{bmatrix} B \\ 0 \end{bmatrix} K
///     - G_2\left(
///         \begin{bmatrix} C_m & F_m \end{bmatrix}
///         + DK
/// \right)
/// $$
///
/// Note that because the controller itself has state, this function creates new
/// dynamics that augment the state with the controller state/dynamics in
/// addition to creating the control law.
pub fn dynamic_measurement_output_feedback<T>(
    lorp_dyn: &LorpDynamics<T>,
    cm_mat: &Array2<T>,
    dm_mat: &Array2<T>,
    fm_mat: &Array2<T>,
) -> (LtiDynamics<T>, impl Fn(T, &Array1<T>) -> Array1<T>)
where
    T: LinalgScalar + Lapack + ScalarOperand + std::cmp::PartialOrd,
{
    let k_mat = static_state_feedback_k(lorp_dyn);
    let a_bar = concatenate![
        Axis(0),
        concatenate![Axis(1), lorp_dyn.a_mat, lorp_dyn.e_mat],
        concatenate![
            Axis(1),
            Array2::zeros((lorp_dyn.s_mat.nrows(), lorp_dyn.a_mat.ncols())),
            lorp_dyn.s_mat
        ]
    ];
    let b_bar = concatenate![
        Axis(0),
        lorp_dyn.b_mat,
        Array2::zeros((lorp_dyn.s_mat.nrows(), lorp_dyn.b_mat.ncols()))
    ];
    let c_bar = concatenate![Axis(1), *cm_mat, *fm_mat];

    let g2_mat = lqr(
        &a_bar.t().to_owned(),
        &c_bar.t().to_owned(),
        &Array2::eye(a_bar.ncols()),
        &Array2::eye(c_bar.nrows()),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap()
    .t()
    .to_owned();
    let g1_mat = a_bar + b_bar.dot(&k_mat) - g2_mat.dot(&(&c_bar + &dm_mat.dot(&k_mat)));

    let a_tilde = concatenate![
        Axis(0),
        concatenate![
            Axis(1),
            lorp_dyn.a_mat,
            lorp_dyn.e_mat,
            Array2::zeros((lorp_dyn.a_mat.nrows(), lorp_dyn.n_state()))
        ],
        concatenate![
            Axis(1),
            Array2::zeros((lorp_dyn.s_mat.nrows(), lorp_dyn.a_mat.ncols())),
            lorp_dyn.s_mat,
            Array2::zeros((lorp_dyn.s_mat.nrows(), lorp_dyn.n_state()))
        ],
        concatenate![Axis(1), g2_mat.dot(cm_mat), g2_mat.dot(fm_mat), g1_mat]
    ];
    let b_tilde = concatenate![
        Axis(0),
        lorp_dyn.b_mat,
        Array2::zeros((lorp_dyn.s_mat.nrows(), lorp_dyn.b_mat.ncols())),
        g2_mat.dot(dm_mat)
    ];
    let c_tilde = concatenate![
        Axis(1),
        lorp_dyn.c_mat,
        lorp_dyn.f_mat,
        Array2::zeros((lorp_dyn.n_output(), lorp_dyn.n_state()))
    ];
    let d_tilde = lorp_dyn.d_mat.clone();

    (
        LtiDynamics {
            a_mat: a_tilde,
            b_mat: b_tilde,
            c_mat: c_tilde,
            d_mat: d_tilde,
        },
        move |_t, x| k_mat.dot(x),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_solve_lme_lyapunov() {
        let a_mat = array![[-6., 1., 0.], [-11., 0., 1.], [-6., 0., 0.]];
        let b_mat = -a_mat.clone().reversed_axes();
        let c_mat = -Array2::eye(3);

        let p_mat = solve_lme(
            &a_mat,
            &b_mat,
            &c_mat,
            Default::default(),
            Default::default(),
        )
        .unwrap();
        assert!(p_mat.abs_diff_eq(
            &array![
                [1.7, -0.5, -0.7],
                [-0.5, 0.7, -0.5],
                [-0.7, -0.5, 1.5333333333333333]
            ],
            1e-12
        ));
    }

    #[test]
    fn test_solve_lme_reg_eqns() {
        let eps = 0.2;
        let den = 1. - eps * eps;
        let lme_a = array![[0., 3.], [-3., 0.]];
        let lme_b = array![
            [0., 1., 0., 0., 0.],
            [-1. / den, 0., 0., 0., -eps / den],
            [0., 0., 0., 1., 0.],
            [eps / den, 0., 0., 0., 1. / den],
            [1., 0., 0., 0., 0.]
        ];
        let lme_c = array![
            [0., 0.],
            [1. / den, 0.],
            [0., 0.],
            [-eps / den, 0.],
            [0., 0.]
        ];
        let lme_m = array![
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.]
        ];
        let sol = solve_lme(&lme_a, &lme_b, &lme_c, Some(&lme_m), Default::default()).unwrap();

        assert!(sol.abs_diff_eq(
            &array![
                [0., 0.],
                [0., 0.],
                [-0.555555555555556, 0.],
                [0., -1.666666666666667],
                [4.999999999999999, 0.]
            ],
            1e-12
        ));
    }

    #[test]
    fn test_regulator_equations() {
        let a_mat = array![[-0.02, -0.098, 0.014], [0., 0., 1.], [1., 0., -0.4]];
        let b_mat = array![[0.098], [0.], [-6.3]];
        let c_mat = array![[1., 0., 0.]];
        let d_mat = Array2::zeros((1, 1));
        let e_mat = Array2::zeros((3, 1));
        let f_mat = array![[-1.]];
        let s_mat = Array2::zeros((1, 1));

        let lorp = LorpDynamics::new(a_mat, b_mat, c_mat, d_mat, e_mat, f_mat, s_mat);

        let reg_eqs = RegulatorEquations::new(&lorp);

        assert!(reg_eqs
            .x_mat
            .abs_diff_eq(&array![[1.], [-0.045351473922902], [0.]], 1e-12));
        assert!(reg_eqs
            .u_mat
            .abs_diff_eq(&array![[0.158730158730159]], 1e-12));
    }

    #[test]
    fn test_static_state_feedback() {
        let eye4 = Array2::eye(4);
        let a_mat = array![[-0.02, -0.098, 0.014], [0., 0., 1.], [1., 0., -0.4]];
        let b_mat = array![[0.098], [0.], [-6.3]];
        let c_mat = array![[1., 0., 0.]];
        let d_mat = Array2::zeros((1, 1));
        let e_mat = Array2::zeros((3, 1));
        let f_mat = array![[-1.]];
        let s_mat = Array2::zeros((1, 1));

        let lorp = LorpDynamics::new(a_mat, b_mat, c_mat, d_mat, e_mat, f_mat, s_mat);

        let u = static_state_feedback(&lorp);

        assert!(
            u(0., &eye4.slice(s![.., 0]).to_owned()).abs_diff_eq(&array![0.804473968919774], 1e-7)
        );
        assert!(
            u(0., &eye4.slice(s![.., 1]).to_owned()).abs_diff_eq(&array![-1.109782934927153], 1e-8)
        );
        assert!(
            u(0., &eye4.slice(s![.., 2]).to_owned()).abs_diff_eq(&array![-1.085526859977024], 1e-8)
        );
        assert!(
            u(0., &eye4.slice(s![.., 3]).to_owned()).abs_diff_eq(&array![-0.696074102023046], 1e-7)
        );
    }

    #[test]
    fn test_dynamic_measurement_output_feedback() {
        let eps = 0.2;
        let den = 1. - eps * eps;
        let a_mat = array![
            [0., 1., 0., 0.],
            [-1. / den, 0., 0., 0.,],
            [0., 0., 0., 1.],
            [eps / den, 0., 0., 0.]
        ];
        let b_mat = array![[0.], [-eps / den], [0.], [1. / den]];
        let c_mat = array![[1., 0., 0., 0.]];
        let d_mat = array![[0.]];
        let e_mat = array![[0., 0.], [1. / den, 0.], [0., 0.], [-eps / den, 0.]];
        let f_mat = array![[0., 0.]];
        let cm_mat = array![[1., 0., 0., 0.], [0., 0., 1., 0.]];
        let dm_mat = array![[0.], [0.]];
        let fm_mat = array![[0., 0.], [0., 0.]];
        let s_mat = array![[0., 3.], [-3., 0.]];

        let lorp_dyn = LorpDynamics::new(a_mat, b_mat, c_mat, d_mat, e_mat, f_mat, s_mat);

        let (feedback_dyn, feedback_ctrl) =
            dynamic_measurement_output_feedback(&lorp_dyn, &cm_mat, &dm_mat, &fm_mat);

        assert!(feedback_dyn.a_mat.abs_diff_eq(
            &array![
                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [
                    -1.041666666666667,
                    0.,
                    0.,
                    0.,
                    1.041666666666667,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.
                ],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [
                    0.208333333333333,
                    0.,
                    0.,
                    0.,
                    -0.208333333333333,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.
                ],
                [0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., -3., 0., 0., 0., 0., 0., 0., 0.],
                [
                    1.552374448254172,
                    0.,
                    -0.027632790508885,
                    0.,
                    0.,
                    0.,
                    -1.552374448254172,
                    1.,
                    0.027632790508885,
                    0.,
                    0.,
                    0.
                ],
                [
                    0.705314999351882,
                    0.,
                    -0.143025601236463,
                    0.,
                    0.,
                    0.,
                    -1.491557332345385,
                    -0.016691440433567,
                    -0.065307732096870,
                    -0.389221909034889,
                    -0.115740740740740,
                    -0.648703181724815
                ],
                [
                    -0.027632790508885,
                    0.,
                    1.749071054158124,
                    0.,
                    0.,
                    0.,
                    0.027632790508885,
                    0.,
                    -1.749071054158124,
                    1.,
                    0.,
                    0.
                ],
                [
                    0.051797449291800,
                    0.,
                    1.030006561802559,
                    0.,
                    0.,
                    0.,
                    -1.120585784324283,
                    0.083457202167834,
                    0.011660104864107,
                    1.946109545174446,
                    5.578703703703703,
                    3.243515908624077
                ],
                [
                    -1.162189394067212,
                    0.,
                    0.206545089667662,
                    0.,
                    0.,
                    0.,
                    1.162189394067212,
                    0.,
                    -0.206545089667662,
                    0.,
                    0.,
                    3.
                ],
                [
                    -0.761259936656251,
                    0.,
                    0.164736902648343,
                    0.,
                    0.,
                    0.,
                    0.761259936656251,
                    0.,
                    -0.164736902648343,
                    0.,
                    -3.,
                    0.
                ],
            ],
            1e-6
        ));
        assert!(feedback_dyn.b_mat.abs_diff_eq(
            &array![
                [0.],
                [-0.208333333333333],
                [0.],
                [1.041666666666667],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.],
                [0.]
            ],
            1e-8
        ));
        assert!(feedback_dyn.c_mat.abs_diff_eq(
            &array![[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]],
            1e-8
        ));
        assert!(feedback_dyn.d_mat.abs_diff_eq(&array![[0.]], 1e-8));

        let eye6 = Array2::eye(6);
        assert!(feedback_ctrl(0., &eye6.slice(s![0, ..]).to_owned())
            .abs_diff_eq(&array![-1.226036801631183], 1e-6));
        assert!(feedback_ctrl(0., &eye6.slice(s![1, ..]).to_owned())
            .abs_diff_eq(&array![0.080118914081120], 1e-7));
        assert!(feedback_ctrl(0., &eye6.slice(s![2, ..]).to_owned()).abs_diff_eq(&array![1.], 1e-8));
        assert!(feedback_ctrl(0., &eye6.slice(s![3, ..]).to_owned())
            .abs_diff_eq(&array![1.868265163367468], 1e-7));
        assert!(feedback_ctrl(0., &eye6.slice(s![4, ..]).to_owned())
            .abs_diff_eq(&array![5.555555555555554], 1e-8));
        assert!(feedback_ctrl(0., &eye6.slice(s![5, ..]).to_owned())
            .abs_diff_eq(&array![3.113775272279113], 1e-7));
    }
}
