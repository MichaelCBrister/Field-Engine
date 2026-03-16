#=
FieldEngine.jl — Main module.

This is just the glue that loads all the sub-modules.
Each file handles one mathematical concept:
  state.jl   → the board as a matrix
  fields.jl  → potential field computation
  energy.jl  → position evaluation via field energy
  search.jl  → field-guided game tree search
=#

module FieldEngine

include("state.jl")
include("fields.jl")
include("energy.jl")
include("search.jl")

using .State
using .Fields
using .Energy
using .Search

export State, Fields, Energy, Search

end # module FieldEngine
