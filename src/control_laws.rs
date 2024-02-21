//! Define common/useful distributed control laws for easy use
//! as feedback control for distributed systems.

use std::ops::Mul;

use ndarray::{linalg::kron, s, Array1, Array2, LinalgScalar, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Lapack, SVD};

/// Create the consensus feedback control law for homogenous single-integrators.
///
/// Formed from the negative of the graph Laplacian and the number of states of
/// each agent. This is equivalent to
/// $$
///     u(t, x) = -(L \otimes I_{n}) x
/// $$
/// where $n$ is `n_states` and $\otimes$ denotes the Kronecker product.
///
/// # Examples
/// ```
/// use ndarray::{array, Array2};
/// use distributed_control as dc;
/// use dc::{EulerIntegration, HomMas, integrator::Integrator, LtiDynamics};
///
/// let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
/// let control = dc::control_laws::single_integrator_consensus(&-laplacian, 2);
/// let x0 = array![-1., 0., 0., 0., 2., 2.];
/// let single_integrator_dynamics = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
/// let single_integrator_mas = HomMas::new(&single_integrator_dynamics, 3);
/// let step_state = EulerIntegration::step(0.0, 1.0, &x0, &single_integrator_mas, &control);
///
/// assert_eq!(step_state, array![0., 0., 1., 2., 0., 0.]);
/// ```
pub fn single_integrator_consensus<T: LinalgScalar>(
    neg_laplacian: &Array2<T>,
    n_states: usize,
) -> impl Fn(T, &Array1<T>) -> Array1<T> {
    let feedback_mat = kron(&neg_laplacian, &Array2::eye(n_states));
    move |_t: T, x: &Array1<T>| -> Array1<T> { feedback_mat.dot(x) }
}

/// Create the control for single integrator forced consensus.
///
/// Formed from the negative of the communication graph Laplacian matrix, $L$,
/// the pinning gains (from the augmented "leader-follower" graph), $k$, and the
/// reference signal, $r$.
/// Equivalent to:
/// $$
///     u(t) = -\left( (L + \operatorname{diag}(k)) \otimes I_n \right)x + k \otimes r
/// $$
/// where $\otimes$ denotes that Kronecker product and $n$ is the size of the
/// state.
pub fn single_integrator_forced_consensus<T: LinalgScalar>(
    neg_laplacian: &Array2<T>,
    pinning_gains: &Array1<T>,
    reference: impl Fn(T) -> Array1<T>,
    n_states: usize,
) -> impl Fn(T, &Array1<T>) -> Array1<T> {
    let pinning_gains = pinning_gains.clone();
    let neg_l_plus_k = neg_laplacian - Array2::from_diag(&pinning_gains);
    let state_feedback = kron(&neg_l_plus_k, &Array2::eye(n_states));
    move |t: T, x: &Array1<T>| -> Array1<T> {
        let r_vec = reference(t);
        let pinned_ref = kron(
            &pinning_gains.slice(s![.., ndarray::NewAxis]),
            &r_vec.insert_axis(ndarray::Axis(1)),
        )
        .remove_axis(ndarray::Axis(1));
        state_feedback.dot(x) + pinned_ref
    }
}

/// Create the leaderless synchronizing controller for a homogenous MAS
///
/// Formed from the negative of the graph Laplacian, a coupling gain (which
/// helps stabilize the communication dynamics), and the feedback gain matrix
/// (which stabilizes the dynamics). Equivalent to
/// $$
///     u(t,x) = - c (L \otimes K) x
/// $$
/// where $c$ is the coupling gain, $K$ the feedback gain matrix, $L$ is the
/// Laplacian, and $\otimes$ denotes the Kronecker product.
pub fn homogenous_leaderless_synchronization<T: LinalgScalar + ScalarOperand>(
    neg_laplacian: &Array2<T>,
    coupling_gain: T,
    feedback_gain: &Array2<T>,
) -> impl Fn(T, &Array1<T>) -> Array1<T> {
    let feedback_mat = kron(&neg_laplacian.mul(coupling_gain), feedback_gain);
    move |_t: T, x: &Array1<T>| -> Array1<T> { feedback_mat.dot(x) }
}

/// Determine the coupling gain that corresponds to a graph Laplacian matrix.
///
/// Computed as
/// $$
///     c = \frac{1}{2\operatorname{Re}(\lambda_2(L))}
/// $$
pub fn coupling_gain<T: LinalgScalar + Lapack<Real = T> + std::cmp::PartialOrd>(
    laplacian: &Array2<T>,
) -> Result<T, LinalgError> {
    let (_, eig, _) = SVD::svd(laplacian, false, false)?;
    let nonzero = eig
        .iter()
        .map(|&v| v.re())
        .filter(|&v| T::from_f64(1e-12).unwrap() < v);
    let min_nonzero = nonzero
        .reduce(|v1, v2| if v1 <= v2 { v1 } else { v2 })
        .unwrap();
    Ok(T::one() / T::from_f64(2.0).unwrap() / min_nonzero)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{
        control_theory::{care_iterative, k_from_p},
        integrator::{EulerIntegration, Integrator},
        HomMas, LtiDynamics,
    };

    #[test]
    fn test_single_integrator_feedback() {
        let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
        let control = single_integrator_consensus(&-&laplacian, 2);
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let single_integrator_2 = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
        let single_integrator_3_2 = HomMas::new(&single_integrator_2, 3);
        let step_state = EulerIntegration::step(0.0, 1.0, &x0, &single_integrator_3_2, &control);

        let k = kron(&-&laplacian, &Array2::eye(2));
        assert_eq!(step_state, k.dot(&x0) + &x0);
    }

    #[test]
    fn test_hom_leaderless_synch() {
        let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
        let a_mat = array![[0., 1.], [0., 0.]];
        let b_mat = array![[0.], [1.]];
        let dynamics = LtiDynamics::new(a_mat.clone(), b_mat.clone());
        let mas = HomMas::new(&dynamics, 3);
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let q_mat = Array2::eye(2);
        let r_mat = Array2::eye(1);
        let k_mat = k_from_p(
            &b_mat,
            &r_mat,
            &care_iterative(
                &a_mat,
                &b_mat,
                &q_mat,
                &r_mat,
                Default::default(),
                Default::default(),
                Default::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let c = coupling_gain(&laplacian).unwrap();
        let control = homogenous_leaderless_synchronization(&-laplacian, c, &k_mat);
        let state_step = EulerIntegration::step(0.0, 1.0, &x0, &mas, &control);

        assert!(state_step.abs_diff_eq(
            &array![-1., 0.5, 0., 2.232050807568878, 4., -0.732050807568878],
            1e-7
        ))
    }
}
