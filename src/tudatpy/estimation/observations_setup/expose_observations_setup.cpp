/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */
#if TUDATPY_ENABLE_DETAILED_PYBIND11_ERRORS
#define PYBIND11_DETAILED_ERROR_MESSAGES
#endif
#include "expose_observations_setup.h"

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "scalarTypes.h"

namespace py = pybind11;

namespace tudatpy
{
namespace estimation
{
namespace observations_setup
{

void expose_observations_setup( py::module& m )
{
    // ancillary_settings and observations_dependent_variables are exposed by
    // the caller (expose_estimation()) before this function runs, since they
    // need py::module_::import (not def_submodule) to attach to the
    // already-created submodules from expose_estimation_types(). Redoing
    // them here via def_submodule would create a second, distinct module
    // object and re-register their pybind11 types, which pybind11 rejects
    // ("an object with that name is already defined").

    auto observations_simulation_settings = m.def_submodule( "observations_simulation_settings" );
    observations_simulation_settings::expose_observations_simulation_settings( observations_simulation_settings );

    auto random_noise = m.def_submodule( "random_noise" );
    random_noise::expose_random_noise( random_noise );

    auto viability = m.def_submodule( "viability" );
    viability::expose_viability( viability );
}

}  // namespace observations_setup
}  // namespace estimation
}  // namespace tudatpy
