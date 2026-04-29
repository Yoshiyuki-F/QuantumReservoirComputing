import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

key = jax.random.key(0)
fb_unitaries = jax.random.normal(key, (1000, 4, 4))

@jax.jit
def slice_with_jnp(fb):
    swap_idx = jnp.array([0, 2, 1, 3])
    return fb[:, swap_idx, :][:, :, swap_idx]

@jax.jit
def slice_with_list(fb):
    return fb[:, [0, 2, 1, 3], :][:, :, [0, 2, 1, 3]]

slice_with_jnp(fb_unitaries)
slice_with_list(fb_unitaries)

res1 = slice_with_jnp(fb_unitaries)
res2 = slice_with_list(fb_unitaries)

t0 = time.time()
for _ in range(10000):
    res1 = slice_with_jnp(fb_unitaries)
jax.block_until_ready(res1)
print("jnp array:", time.time() - t0)

t0 = time.time()
for _ in range(10000):
    res2 = slice_with_list(fb_unitaries)
jax.block_until_ready(res2)
print("list:", time.time() - t0)
