//! This module contains common control-theoretic functions or computations.

use ndarray::{s, Array2, LinalgScalar};
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, Scalar, SVD};

/// Determine the rank of a matrix (using the SVD).
///
/// Uses a similar method to the numpy implementation: see [numpy][numpy-rank]
///
/// [numpy-rank]: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html
///
/// # Examples
/// ```
/// use ndarray::array;
/// use distributed_control::control_theory::rank;
///
/// let mat = array![[1., 2., 1.], [0., 1., 0.], [2., 5., 2.]];
///
/// assert_eq!(rank(&mat, Default::default()).expect("Error in rank"), 2);
/// ```
pub fn rank<T: Lapack>(mat: &Array2<T>, eps: Option<f64>) -> Result<usize, LinalgError> {
    let (_, singular_values, _) = SVD::svd(mat, false, false)?;
    let sv_max = singular_values
        .iter()
        .map(|&v| v.abs().re())
        .max_by(|&left, &right| left.partial_cmp(&right).unwrap())
        .unwrap();
    let max_dim = mat.nrows().max(mat.ncols());
    let eps = eps.or(Some(f64::EPSILON)).unwrap();
    let tol = sv_max * T::real(max_dim) * T::real(eps);
    Ok(singular_values
        .iter()
        .map(|&v| if v.abs().re() < tol { 0_usize } else { 1 })
        .sum())
}

/// Calculate a matrix power (integer only)
fn pow<T: LinalgScalar>(mat: &Array2<T>, exponent: u32) -> Array2<T> {
    assert_eq!(mat.nrows(), mat.ncols());
    match exponent {
        0 => Array2::<T>::eye(mat.ncols()),
        1 => mat.clone(),
        exponent => {
            let mut result = mat.clone();
            for _i in 2..=exponent {
                result = result.dot(mat);
            }
            result
        }
    }
}

/// Determine if the pair of matrices ($A$, $B$) is controllable.
///
/// The pair of matrices ($A$, $B$), where $A \in \mathbb{R}^{n\times n}$ and
/// $B \in \mathbb{R}^{n\times m}$, are controllable if and only if the rank of
/// $\begin{bmatrix}B & AB & A^2B & \dots & A^{n-1}B\end{bmatrix}$ is equal to $n$, otherwise, it is not.
///
/// Note that this function can also be used to check observability due to the
/// duality of controllability and observability. A pair ($A$, $C$) (
/// $A\in\mathbb{R}^{n\times n}$, $C\in\mathbb{R}^{p\times n}$) is observable
/// if and only if the pair ($A^\text{T}$, $C^\text{T}$) is controllable.
///
/// # Examples
/// ```
/// use ndarray::array;
///
/// let a_mat = array![[0., 1.], [0., 0.]];
/// let b_mat = array![[0.], [1.]];
///
/// assert!(controllable(&a_mat, &b_mat));
/// ```
pub fn controllable<T: LinalgScalar + ndarray_linalg::Lapack>(
    a_mat: &Array2<T>,
    b_mat: &Array2<T>,
) -> bool {
    let controllability = controllability_matrix(a_mat, b_mat);
    rank(&controllability, Default::default())
        .expect("The rank of the controllability matrix could not be determined.")
        == controllability.nrows()
}

/// Calculate the controllability matrix for the matrix pair ($A$, $B$).
///
/// The matrix pair are such that $A\in\mathbb{R}^{n\times n}$ and
/// $B\in\mathbb{R}^{n\times m}$.
/// Constructs the controllability matrix as
/// $\begin{bmatrix}B & AB & A^2B & \dots & A^{n-1}B\end{bmatrix}$.
pub fn controllability_matrix<T: LinalgScalar>(a_mat: &Array2<T>, b_mat: &Array2<T>) -> Array2<T> {
    let n = a_mat.ncols();
    let mut controllability = Array2::zeros((b_mat.nrows(), n * b_mat.ncols()));
    for i in 0..n {
        let col_start = i * b_mat.ncols();
        let col_end = col_start + b_mat.ncols();
        controllability
            .slice_mut(s![.., col_start..col_end])
            .assign(&pow(a_mat, i.try_into().unwrap()).dot(b_mat));
    }
    controllability
}

/// Solve the Continuous Algebraic Riccati Equation (CARE) iteratively.
///
/// With $Q\succcurlyeq 0$ and $R\succcurlyeq 0$ the CARE is
/// $PA + A^\text{T}P - PBR^{-1}B^{\text{T}}P + Q = 0$ where $P=P^{\text{T}}$ is
/// the desired solution. The iterative method treats the equation as a matrix
/// differential equation
/// $\dot{P} = PA + A^\text{T}P - PBR^{-1}B^\text{T}P + Q$ and simulates the
/// system (with Euler integration) from the initial condition $P(0)=Q$ until
/// equilibrium is reached, i.e., $\dot{P}=0$ and $P(t_f)$ is the desired
/// solution $P$.
///
/// Note that this is the most simple and straightforward method of solving the
/// CARE and can have both poor computational performance, as well as numerical
/// instabilities in some cases. For many systems, however, it is a sufficient
/// method for computing the solution, and if computation time is not a major
/// concern, the stability can be addressed with a smaller value for `dt` and/or
/// a larger `iter_max`
///
/// # Examples
/// ```
/// use ndarray::{array, Array2}
///
/// let a_mat = array![[0., 1.], [0., 0.]];
/// let b_mat = array![[0., 0.], [1., 1.]];
/// let q_mat = Array2::eye(2);
/// let r_mat = Array2::eye(2);
/// let p_mat = care_iterative(
///     &a_mat,
///     &b_mat,
///     &q_mat,
///     &r_mat,
///     Default::default(), // default `dt` is 0.001
///     Default::default(), // default `tol` is 1E-10
///     Default::default(), // default `iter_max` is 100000
/// )
/// .unwrap();
/// assert!(p_mat.abs_diff_eq(
///     // "Truth" value obtained from Matlab's `icare` function.
///     &array![
///         [1.553773974030038, 0.707106781186548],
///         [0.707106781186548, 1.098684113467811]
///     ],
///     // Increased precision can be obtained with smaller `tol`
///     // and/or larger `iter_max`.
///     1.0e-7
/// ));
/// ```
pub fn care_iterative<T>(
    a_mat: &Array2<T>,
    b_mat: &Array2<T>,
    q_mat: &Array2<T>,
    r_mat: &Array2<T>,
    dt: Option<T>,
    tol: Option<T>,
    iter_max: Option<usize>,
) -> Result<Array2<T>, LinalgError>
where
    T: LinalgScalar + ndarray_linalg::Lapack + ndarray::ScalarOperand + std::cmp::PartialOrd,
{
    let dt = dt.or(T::from_f64(0.001)).unwrap();
    let tol = tol.or(T::from_f64(1.0e-10)).unwrap();
    let iter_max = iter_max.or(Some(100000)).unwrap();

    let mut p_mat = q_mat.clone();

    let a_transpose = a_mat.t();
    let b_transpose = b_mat.t();
    let r_inverse = Array2::inv(r_mat)?;

    for _i in 0..iter_max {
        let p_next = &p_mat
            + (p_mat.dot(a_mat) + a_transpose.dot(&p_mat)
                - p_mat
                    .dot(b_mat)
                    .dot(&r_inverse)
                    .dot(&b_transpose)
                    .dot(&p_mat)
                + q_mat)
                * dt;
        let diff = (&p_next - &p_mat)
            .iter()
            .map(|&v| -> T { v * v.conj() })
            .sum::<T>()
            .sqrt();
        if diff < tol {
            println!("Returning early ({_i}) with sufficiently small update");
            return Ok(p_next);
        }
        p_mat = p_next;
    }
    println!("Reached iteration maximum.");
    Ok(p_mat)
}

/// Determine the feedback matrix $K$ from $P$, the CARE solution.
///
/// Simply $K=R^{-1}B^\text{T}P$.
pub fn k_from_p<T: LinalgScalar + ndarray_linalg::Lapack>(
    b_mat: &Array2<T>,
    r_mat: &Array2<T>,
    p_mat: &Array2<T>,
) -> Result<Array2<T>, LinalgError> {
    Ok(Array2::inv(r_mat)?.dot(&b_mat.t()).dot(p_mat))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_rank() {
        let mat = array![[1., 2., 1.], [0., 1., 0.], [2., 5., 2.]];
        assert_eq!(rank(&mat, Default::default()).expect("Error in rank"), 2);
    }

    #[test]
    fn test_pow_0() {
        let mat = array![[1., 2.], [3., 4.]];
        assert_eq!(pow(&mat, 0), array![[1., 0.], [0., 1.]]);
    }

    #[test]
    fn test_pow_1() {
        let mat = array![[1., 2.], [3., 4.]];
        assert_eq!(pow(&mat, 1), mat);
    }

    #[test]
    fn test_pow_2() {
        let mat = array![[1., 2.], [3., 4.]];
        assert_eq!(pow(&mat, 2), array![[7., 10.], [15., 22.]]);
    }

    #[test]
    fn test_controllability_matrix() {
        let a_mat = array![[0., 1.], [0., 0.]];
        let b_mat = array![[0., 0.], [1., 1.]];
        let controllability = controllability_matrix(&a_mat, &b_mat);
        assert_eq!(controllability, array![[0., 0., 1., 1.], [1., 1., 0., 0.]]);
    }

    #[test]
    fn test_is_controllable() {
        let a_mat = array![[0., 1.], [0., 0.]];
        let b_mat = array![[0., 0.], [1., 1.]];
        assert!(controllable(&a_mat, &b_mat));
    }

    #[test]
    fn test_is_not_controllable() {
        let a_mat = array![[0., 1.], [0., 0.]];
        let b_mat = array![[1., 1.], [0., 0.]];
        assert!(!controllable(&a_mat, &b_mat));
    }

    #[test]
    fn test_care_iterative() {
        let a_mat = array![[0., 1.], [0., 0.]];
        let b_mat = array![[0., 0.], [1., 1.]];
        let q_mat = Array2::eye(2);
        let r_mat = Array2::eye(2);
        let p_mat = care_iterative(
            &a_mat,
            &b_mat,
            &q_mat,
            &r_mat,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
        assert!(p_mat.abs_diff_eq(
            &array![
                [1.553773974030038, 0.707106781186548],
                [0.707106781186548, 1.098684113467811]
            ],
            1.0e-7
        ));
    }
}