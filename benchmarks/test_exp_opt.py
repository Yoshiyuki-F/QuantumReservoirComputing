import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

theta_samples = jax.random.uniform(jax.random.key(0), (10000,))

@jax.jit
def old_way(angles):
    c = jnp.cos(angles / 2.0)
    s = -1.0j * jnp.sin(angles / 2.0)
    zm = jnp.exp(-1.0j * angles / 2.0)
    zp = jnp.exp(1.0j * angles / 2.0)
    return c, s, zm, zp

@jax.jit
def new_way(angles):
    zp = jnp.exp(1.0j * (angles / 2.0))
    zm = jnp.conj(zp)
    c = jnp.real(zp)
    s = -1.0j * jnp.imag(zp)
    return c, s, zm, zp

old_way(theta_samples)
new_way(theta_samples)

c1, s1, zm1, zp1 = old_way(theta_samples)
c2, s2, zm2, zp2 = new_way(theta_samples)

t0 = time.time()
for _ in range(10000):
    c1, s1, zm1, zp1 = old_way(theta_samples)
jax.block_until_ready(c1)
print("Old way:", time.time() - t0)

t0 = time.time()
for _ in range(10000):
    c2, s2, zm2, zp2 = new_way(theta_samples)
jax.block_until_ready(c2)
print("New way:", time.time() - t0)

print("Max diff c:", float(jnp.max(jnp.abs(c1 - c2))))
print("Max diff s:", float(jnp.max(jnp.abs(s1 - s2))))
print("Max diff zm:", float(jnp.max(jnp.abs(zm1 - zm2))))
print("Max diff zp:", float(jnp.max(jnp.abs(zp1 - zp2))))
