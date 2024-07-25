
import peano4, exahype2
import os

from exahype2.solvers.aderdg.ADERDG import Polynomials

from Elastic import flux, eigenvalue, initial, boundary

min_level           = 1
max_depth           = 0
order               = 1
precision           = "double"
e_t                 = 2.0
plot_dt             = 0.1
polynomials         = "legendre"

size_1d = 3.0 #3.0**min_level
size    = [size_1d, size_1d, size_1d]   # thousand meters
offset  = [-size[0]/2., -size[1]/2., 0.0]  # thousand meters

unknowns            = {"v": 3, "sigma": 6}
auxiliary_variables = {"rho":1, "cp": 1, "cs": 1}

max_h               = 1.1 * min(size) / (3.0**min_level)
min_h               = max_h / (3.0**max_depth)

project = exahype2.Project( ["applications", "exahype2", "elastic"], ".",
  executable="LOH_level_"+str(min_level)+"_order_"+str(order)+"_"+polynomials+"_precision_"+str(precision) )

theSolver=exahype2.solvers.aderdg.GlobalAdaptiveTimeStep(
  name="elastic",  order=order,
  unknowns=unknowns, auxiliary_variables=auxiliary_variables,
  min_cell_h=min_h, max_cell_h=max_h, time_step_relaxation=0.9
)

theSolver.add_kernel_optimisations(
  is_linear=True, polynomials=(Polynomials.Gauss_Lobatto if polynomials=="lobatto" else Polynomials.Gauss_Legendre)
  ,solution_persistent_storage_precision=precision
  ,predictor_computation_precisions=[precision]
  ,corrector_computation_precision=precision
)

theSolver.set_implementation(
  initial_conditions    = initial(),
  boundary_conditions   = boundary(),
  eigenvalues   = eigenvalue(),
  flux          = flux(),
  point_source  = 1
)

theSolver.add_user_solver_includes("""
#include <fstream>
#include <vector>
""")

project.add_solver(theSolver)

tracer_particles = project.add_tracer(name="Tracer", attribute_count=12)

project.add_action_set_to_initialisation(
    exahype2.tracer.InsertParticlesByCoordinates(
        particle_set=tracer_particles,
        coordinates=[ [0.693, 0.000, 0.000], [7.348, 7.348, 0.000] ]
    )
)

project.add_action_set_to_timestepping(
    peano4.toolbox.particles.api.UpdateParallelState(particle_set=tracer_particles)
)

project.add_action_set_to_timestepping(
    exahype2.tracer.DiscontinuousGalerkinTracing(
        tracer_particles, theSolver,
        project_on_tracer_properties_kernel="::exahype2::dg::projectAllValuesOntoParticle"
    )
)

project.add_action_set_to_timestepping(
    exahype2.tracer.DumpTracerIntoDatabase(
        particle_set=tracer_particles, solver=theSolver,
        filename="/app/tracers/Cartesian_level_"+str(min_level)+"_order_"+str(order)+"_"+polynomials+"_precision_"+str(precision),
        data_delta_between_two_snapsots=1e16, time_delta_between_two_snapsots=0.000001,
        output_precision=10, clear_database_after_flush=False
    )
)

if not os.path.exists("tracers"):
    os.makedirs("tracers")

if plot_dt > 0.0:
  project.set_output_path("solutions")

project.set_global_simulation_parameters(
  dimensions            = 3,
  offset                = offset[0:3],
  size                  = size[0:3],
  min_end_time          = e_t,
  first_plot_time_stamp = 0.0,
  time_in_between_plots = plot_dt,
)

project.set_load_balancing( "toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates", "new ::exahype2::LoadBalancingConfiguration()" )
project.set_Peano4_installation( "../../../../", peano4.output.CompileMode.Release )
peano4_project = project.generate_Peano4_project("False")
peano4_project.output.makefile.add_h_file("elastic.h")
peano4_project.output.makefile.add_cpp_file("elastic.cpp")

peano4_project.build(make_clean_first=True)
