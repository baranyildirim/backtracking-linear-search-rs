import jax
import jax.numpy as jnp
import jaxopt
import time

@jax.jit
def f(x):
    return 10. * x[0]**2 + 0.001 * x[1]**2

x0 = jnp.array([1., 1.])
solver = jaxopt.GradientDescent(f, acceleration=False)

result = solver.run(x0)

start = time.time()
result = solver.run(x0)
end = time.time()

print((end - start)*1000)