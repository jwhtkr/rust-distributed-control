//! Define common/useful distributed control laws for easy use
//! as feedback control for distributed systems.

use ndarray::{linalg::kron, Array1, Array2, LinalgScalar};

/// Create the consensus feedback control law for homogenous single-integrators.
///
/// Formed from the negative of the graph laplacian and the number of states of
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
    return move |_t: T, x: &Array1<T>| -> Array1<T> { feedback_mat.dot(x) };
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{
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
}
