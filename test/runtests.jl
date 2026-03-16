module FieldEngineTestState
include(joinpath(@__DIR__, "test_state.jl"))
end

module FieldEngineTestFieldsEnergy
include(joinpath(@__DIR__, "test_fields_energy.jl"))
end

module FieldEngineTestSearch
include(joinpath(@__DIR__, "test_search.jl"))
end
