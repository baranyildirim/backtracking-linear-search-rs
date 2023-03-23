use anyhow::Error;
use anyhow::Result;
use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::IterState;
use argmin::core::Problem;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::condition::StrongWolfeCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;

#[derive(Eq, PartialEq, Debug)]
struct UserDefinedProblem;

impl CostFunction for UserDefinedProblem {
    type Param = Vec<f32>;
    type Output = f32;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        Ok(10.0 * x[0].powi(2) + 0.001 * x[1].powi(2))
    }
}

impl Gradient for UserDefinedProblem {
    type Param = Vec<f32>;
    type Gradient = Vec<f32>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(vec![10.0 * x[0], 0.001 * x[1]])
    }
}

fn main() -> Result<()> {
    let condition = StrongWolfeCondition::<f32>::new(1e-4, 0.9)?;

    let linesearch =
        BacktrackingLineSearch::<Vec<f32>, Vec<f32>, StrongWolfeCondition<f32>, f32>::new(
            condition,
        );

    let descent = SteepestDescent::new(linesearch);

    // Run solver
    let result = Executor::new(UserDefinedProblem {}, descent)
        .configure(|state| state.param(vec![1.0f32, 1.0f32]))
        .run()?;
    println!("{}", result.to_string());
    Ok(())
}
