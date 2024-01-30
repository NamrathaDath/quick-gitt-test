import pybamm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

model = pybamm.BaseModel()

c = pybamm.Variable("Concentration", domain="particle")

#N = -pybamm.grad(c)  # define the flux
#dcdt = -pybamm.div(N)  # define the rhs equation

model.rhs = {c: pybamm.div(pybamm.grad(c))}  # add the equation to rhs dictionary

# initial conditions
c0 = pybamm.Scalar(0)
model.initial_conditions = {c: c0}

# boundary conditions
lbc = pybamm.Scalar(0)
rbc = pybamm.Scalar(1)
model.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}

model.variables = {"Concentration": c}

# define geometry
r = pybamm.SpatialVariable(
    "r", domain=["particle"], coord_sys="spherical polar"
)
#r = pybamm.SpatialVariable(
#    "r", domain=["particle"], coord_sys="cartesian"
#)
geometry = {
    "particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
}

# mesh and discretise
submesh_types = {"particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 500}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"particle": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 100000)
solution = solver.solve(model, t)

# post-process, so that the solution can be called at any time t or space r
# (using interpolation)
c = solution["Concentration"]

# plot
fig, ax = plt.subplots(1, 1, figsize=(13, 4))

t = solution.t
c_sand = 2 / np.sqrt(np.pi) * np.sqrt(t)


ax.plot(solution.t, c(solution.t, r=1), label="Numerical solution")
ax.plot(t, c_sand, label="Sand's solution", linestyle="--")
ax.set_xlabel("t")
ax.set_ylabel("Surface concentration")

ax.legend()

df = pd.DataFrame({"t_tilde": t, "c_tilde_surf": c(solution.t, r=1)})
df.to_csv("c_tilde_surf.csv", index=False)

plt.tight_layout()
plt.show()