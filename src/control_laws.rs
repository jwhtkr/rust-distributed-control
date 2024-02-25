//! Define common/useful distributed control laws for easy use
//! as feedback control for distributed systems.

use std::ops::Mul;

use ndarray::{
    linalg::kron, s, Array1, Array2, ArrayBase, Data, Ix1, Ix2, LinalgScalar, ScalarOperand,
};
use ndarray_linalg::{error::LinalgError, EigVals, Lapack, Scalar};

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
/// let offsets = array![0., 0., 0., 0., 0., 0.];
/// let control = dc::control_laws::single_integrator_consensus(
///     &-laplacian, &offsets, 2
/// );
/// let x0 = array![-1., 0., 0., 0., 2., 2.];
/// let single_integrator_dynamics = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
/// let single_integrator_mas = HomMas::new(&single_integrator_dynamics, 3);
/// let step_state = EulerIntegration::step(0.0, 1.0, &x0, &single_integrator_mas, &control);
///
/// assert_eq!(step_state, array![0., 0., 1., 2., 0., 0.]);
/// ```
pub fn single_integrator_consensus<'a, T: LinalgScalar, S: Data<Elem = T>>(
    neg_laplacian: &ArrayBase<S, Ix2>,
    offsets: &'a ArrayBase<S, Ix1>,
    n_states: usize,
) -> impl Fn(T, &ArrayBase<S, Ix1>) -> Array1<T> + 'a {
    let feedback_mat = kron(neg_laplacian, &Array2::eye(n_states));
    move |_t: T, x: &ArrayBase<S, Ix1>| -> Array1<T> { feedback_mat.dot(&(x - offsets)) }
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
/// # Examples
/// ```
/// use ndarray::{array, s, Array1, Array2};
/// use distributed_control as dc;
/// use dc::{control_laws::single_integrator_forced_consensus, integrator::Integrator, *};
///
/// let laplacian = array![[0., 0., 0.], [-1., 1., 0.], [0., -1., 1.]];
/// let offsets = array![0., 0., 0., 0., 0., 0.];
/// let pinning_gains = array![1., 0., 0.];
/// let reference = |_t| array![1., -2.];
/// let control = single_integrator_forced_consensus(
///     &-&laplacian, &offsets, &pinning_gains, reference, 2
/// );
/// let x0 = array![-1., 0., 0., 0., 2., 2.];
/// let agent_dyn_2d = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
/// let mas_dyn_3_agents = HomMas::new(&agent_dyn_2d, 3);
/// let times = Array1::linspace(0.0, 20.0, 201).into_iter().collect();
/// let states = EulerIntegration::simulate(&times, &x0, &mas_dyn_3_agents, &control);
///
/// println!("{}", states.slice(s![.., -1]));
/// assert!(
///     states.slice(s![.., -1]).abs_diff_eq(
///         &array![1., -2., 1., -2., 1., -2.], 1e-6
///     )
/// )
/// ```
pub fn single_integrator_forced_consensus<
    'a,
    T: LinalgScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
>(
    neg_laplacian: &ArrayBase<S1, Ix2>,
    offsets: &'a ArrayBase<S1, Ix1>,
    pinning_gains: &'a ArrayBase<S1, Ix1>,
    reference: impl Fn(T) -> ArrayBase<S2, Ix1> + 'a,
    n_states: usize,
) -> impl Fn(T, &ArrayBase<S1, Ix1>) -> Array1<T> + 'a {
    let neg_l_plus_k = neg_laplacian - Array2::from_diag(pinning_gains);
    let state_feedback = kron(&neg_l_plus_k, &Array2::eye(n_states));
    move |t: T, x: &ArrayBase<S1, Ix1>| -> Array1<T> {
        let r_vec = reference(t);
        let pinned_ref = kron(
            &pinning_gains.slice(s![.., ndarray::NewAxis]),
            &r_vec.insert_axis(ndarray::Axis(1)),
        )
        .remove_axis(ndarray::Axis(1));
        state_feedback.dot(&(x - offsets)) + pinned_ref
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
pub fn homogeneous_leaderless_synchronization<
    'a,
    T: LinalgScalar + ScalarOperand,
    S: Data<Elem = T>,
>(
    neg_laplacian: &ArrayBase<S, Ix2>,
    offsets: &'a ArrayBase<S, Ix1>,
    coupling_gain: T,
    feedback_gain: &ArrayBase<S, Ix2>,
) -> impl Fn(T, &ArrayBase<S, Ix1>) -> Array1<T> + 'a {
    let feedback_mat = kron(&neg_laplacian.mul(coupling_gain), feedback_gain);
    move |_t: T, x: &ArrayBase<S, Ix1>| feedback_mat.dot(&(x - offsets))
}

/// Create the leader-follower synchronizing controller for a homogenous MAS
///
/// If $L$ is the graph Laplacian, $h$ is a vector of pinning gains for each
/// agent, $c$ is a coupling gain, and $K$ is a feedback gain matrix, with the
/// leader reference given as a function of time in $r(t)$, then this control
/// law is:
/// $$
///     u(t,x) = -c ((L + H) \otimes K) x + c (h \otimes Kr(t))
/// $$
///
/// # Examples
///
/// ```
/// use ndarray::{array, s, Array1, Array2};
///
/// use distributed_control as dc;
/// use dc::{
///     control_laws::{coupling_gain, homogeneous_leader_follower_synchronization},
///     control_theory::{care_iterative, k_from_p},
///     integrator::Integrator,
///     *,
/// };
///
/// let laplacian = array![[0., 0., 0.], [-1., 1., 0.], [0., -1., 1.]];
/// let a_mat = array![[0., 1.], [0., 0.]];
/// let b_mat = array![[0.], [1.]];
/// let dynamics = LtiDynamics::new(a_mat.clone(), b_mat.clone());
/// let mas = HomMas::new(&dynamics, 3);
/// let x0 = array![-1., 0., 0., 0., 2., 2.];
/// let q_mat = Array2::eye(2);
/// let r_mat = Array2::eye(1);
/// let k_mat = k_from_p(
///     &b_mat,
///     &r_mat,
///     &care_iterative(
///         &a_mat,
///         &b_mat,
///         &q_mat,
///         &r_mat,
///         Default::default(),
///         Default::default(),
///         Default::default(),
///     )
///     .unwrap(),
/// )
/// .unwrap();
/// let c = coupling_gain(&laplacian).unwrap();
/// let pinning_gains = array![1., 0., 0.];
/// let times = Array1::linspace(0.0, 30.0, 301).into_iter().collect();
/// let r0 = array![-4., 0.];
/// let reference_states =
///     EulerIntegration::simulate(&times, &r0, &dynamics, &|_t, _x| array![0.]);
/// let reference = |t| {
///     reference_states.column(
///         times
///             .iter()
///             .enumerate()
///             .find_map(|(i, &v)| if v == t { Some(i) } else { None })
///             .unwrap(),
///     )
/// };
/// let offsets = array![0., 0., 0., 0., 0., 0.];
/// let control = homogeneous_leader_follower_synchronization(
///     &-laplacian,
///     &offsets,
///     c,
///     &k_mat,
///     &pinning_gains,
///     reference,
/// );
/// let states = EulerIntegration::simulate(&times, &x0, &mas, &control);
///
/// assert!(states.slice(s![.., -1]).abs_diff_eq(
///     &array![-4., 0., -4., 0., -4., 0.],
///     0.1
/// ));
///```
pub fn homogeneous_leader_follower_synchronization<
    'a,
    T: LinalgScalar + ScalarOperand,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
>(
    neg_laplacian: &ArrayBase<S1, Ix2>,
    offsets: &'a ArrayBase<S1, Ix1>,
    coupling_gain: T,
    feedback_gain: &ArrayBase<S1, Ix2>,
    pinning_gains: &ArrayBase<S1, Ix1>,
    reference: impl Fn(T) -> ArrayBase<S2, Ix1> + 'a,
) -> impl Fn(T, &ArrayBase<S1, Ix1>) -> Array1<T> + 'a {
    let pinning_gains = Array2::from_diag(pinning_gains);
    let feedback_mat = kron(&(neg_laplacian - &pinning_gains), feedback_gain).mul(coupling_gain);
    let reference_mat = kron(&pinning_gains, feedback_gain).mul(coupling_gain);
    move |t, x| {
        let reference = Array1::from_iter(
            std::iter::repeat(reference(t).iter())
                .take(pinning_gains.nrows())
                .flatten()
                .cloned(),
        );
        feedback_mat.dot(&(x - offsets)) + reference_mat.dot(&reference)
    }
}

/// Determine the coupling gain that corresponds to a graph Laplacian matrix.
///
/// Computed as
/// $$
///     c = \frac{1}{2\operatorname{Re}(\lambda_2(L))}
/// $$
pub fn coupling_gain<T: LinalgScalar + Lapack + std::cmp::PartialOrd, S: Data<Elem = T>>(
    laplacian: &ArrayBase<S, Ix2>,
) -> Result<T, LinalgError> {
    // let (_, eig, _) = SVD::svd(laplacian, false, false)?;
    let eig = laplacian.eigvals()?;
    let nonzero = eig
        .iter()
        .map(|&v| v.re())
        .filter(|&v| T::from_f64(1e-12).unwrap() < T::from_real(v));
    let min_nonzero = nonzero.min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    Ok(T::one() / T::from_f64(2.0).unwrap() / T::from_real(min_nonzero))
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
        let offsets = array![0., 0., 0., 0., 0., 0.];
        let control = single_integrator_consensus(&-&laplacian, &offsets, 2);
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let single_integrator_2 = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
        let single_integrator_3_2 = HomMas::new(&single_integrator_2, 3);
        let step_state = EulerIntegration::step(0.0, 1.0, &x0, &single_integrator_3_2, &control);

        let k = kron(&-&laplacian, &Array2::eye(2));
        assert_eq!(step_state, k.dot(&x0) + &x0);
    }

    #[test]
    fn test_single_int_offset() {
        let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
        let offsets = array![0., 0., 1., 0., 0., 1.];
        let control = single_integrator_consensus(&-&laplacian, &offsets, 2);
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let single_integrator_2 = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
        let single_integrator_3_2 = HomMas::new(&single_integrator_2, 3);
        let times = Array1::linspace(0., 20., 201).into_iter().collect();
        let states = EulerIntegration::simulate(&times, &x0, &single_integrator_3_2, &control);

        assert!(states
            .slice(s![.., -1])
            .abs_diff_eq(&array![0., 1. / 3., 1., 1. / 3., 0., 4. / 3.], 1e-8));
    }

    #[test]
    fn test_forced_single_int() {
        let laplacian = array![[0., 0., 0.], [-1., 1., 0.], [0., -1., 1.]];
        let offsets = array![0., 0., 0., 0., 0., 0.];
        let pinning_gains = array![1., 0., 0.];
        let reference = |_t| array![1., -2.];
        let control = single_integrator_forced_consensus(
            &-&laplacian,
            &offsets,
            &pinning_gains,
            reference,
            2,
        );
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let single_integrator_2 = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
        let single_integrator_3_2 = HomMas::new(&single_integrator_2, 3);
        let times = Array1::linspace(0.0, 20.0, 201).into_iter().collect();
        let states = EulerIntegration::simulate(&times, &x0, &single_integrator_3_2, &control);

        assert!(states
            .slice(s![.., -1])
            .abs_diff_eq(&array![1., -2., 1., -2., 1., -2.], 1e-6))
    }

    #[test]
    fn test_forced_single_int_offset() {
        let laplacian = array![[0., 0., 0.], [-1., 1., 0.], [0., -1., 1.]];
        let offsets = array![0., 0., 1., 0., 0., 1.];
        let pinning_gains = array![1., 0., 0.];
        let reference = |_t| array![1., -2.];
        let control = single_integrator_forced_consensus(
            &-&laplacian,
            &offsets,
            &pinning_gains,
            reference,
            2,
        );
        let x0 = array![-1., 0., 0., 0., 2., 2.];
        let single_integrator_2 = LtiDynamics::new(Array2::zeros((2, 2)), Array2::eye(2));
        let single_integrator_3_2 = HomMas::new(&single_integrator_2, 3);
        let times = Array1::linspace(0.0, 20.0, 201).into_iter().collect();
        let states = EulerIntegration::simulate(&times, &x0, &single_integrator_3_2, &control);

        assert!(states
            .slice(s![.., -1])
            .abs_diff_eq(&array![1., -2., 2., -2., 1., -1.], 1e-6))
    }

    #[test]
    fn test_hom_leaderless_sync() {
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
        let offsets = array![0., 0., 0., 0., 0., 0.];
        let control = homogeneous_leaderless_synchronization(&-laplacian, &offsets, c, &k_mat);
        let state_step = EulerIntegration::step(0.0, 1.0, &x0, &mas, &control);

        assert!(state_step.abs_diff_eq(
            &array![-1., 0.5, 0., 2.232050807568878, 4., -0.732050807568878],
            1e-7
        ))
    }

    #[test]
    fn test_hom_leader_follower_sync() {
        let laplacian = array![[0., 0., 0.], [-1., 1., 0.], [0., -1., 1.]];
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
        let pinning_gains = array![1., 0., 0.];
        let times = Array1::linspace(0.0, 60.0, 301).into_iter().collect();
        let r0 = array![-4., 0.];
        let reference_states =
            EulerIntegration::simulate(&times, &r0, &dynamics, &|_t, _x| array![0.]);
        let reference = |t| {
            reference_states.column(
                times
                    .iter()
                    .enumerate()
                    .find_map(|(i, &v)| if v == t { Some(i) } else { None })
                    .unwrap(),
            )
        };
        let offsets = array![0., 0., 0., 0., 0., 0.];
        let control = homogeneous_leader_follower_synchronization(
            &-laplacian,
            &offsets,
            c,
            &k_mat,
            &pinning_gains,
            reference,
        );
        let states = EulerIntegration::simulate(&times, &x0, &mas, &control);

        assert!(states
            .slice(s![.., -1])
            .abs_diff_eq(&array![-4., 0., -4., 0., -4., 0.], 1e-7))
    }
}
