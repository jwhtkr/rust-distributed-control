//! Linear Output Regulation Problem solutions/functionality.

/// The regulator equations for a set of matrices A, B, E, C, D, F.
///
/// The regulator equations correspond to the equations
/// $$ \dot{x} = A x + B u + E v $$
/// $$ e = C x + B u + F v $$
/// with the state $x\in\mathbb{R}^n$, the input $u\in\mathbb{R}^m$,
/// the measurement error $e\in\mathbb{R}^p$, and an exosystem
/// $v\in\mathbb{R}^q$ such that $\dot{v}=Sv$ for $S\in\mathbb{R}^{q\times q}$
/// with $\operatorname{spec}(S)\subseteq CRHP$.
pub struct RegulatorEquations {}


/// Create the static state feedback control law that solves the LORP
///
/// For the given $A, B, E, C, D, F, S$, the control law is of the following
/// form
/// $$ u = K_1 x + K_2 v.$$
/// This control law solves the LORP when: $K_1$ is selected such that
/// $A - B K_1$ is Hurwitz, and $K_2 = U - K_1 X$ with $X,U$ the solution to the
/// corresponding regulator equations ([`RegulatorEquations`]).
pub fn static_state_feedback(){}


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
pub fn dynamic_measurement_output_feedback(){}
