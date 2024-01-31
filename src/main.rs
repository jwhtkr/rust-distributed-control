use dc::integrator::Integrator;
use distributed_control as dc;
use ndarray::{array, Array1};

fn main() {
    let lin_dyn = dc::LtiDynamics::new(array![[0.]], array![[1.]]);
    let mas = dc::HomMas::new(&lin_dyn, 3);
    let initial_states = array![-1., 0., 1.];
    let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
    let u = |_t: f64, x: &Array1<f64>| -laplacian.dot(x);

    let states = dc::EulerIntegration::simulate(
        (0..100).map(|i| i as f64 * 0.1).collect(),
        &initial_states,
        &mas,
        &u,
    );
    println!("{states}");
    println!("{}", states.column(states.ncols() - 1));
}
