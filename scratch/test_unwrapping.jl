using SciMLBase, Symbolics

# Define a function wrapped in a FunctionWrapper (simulated)
function basic_f!(du, u, p, t)
    du[1] = -u[1]
end

# Wrap it in an ODEFunction (which is what SciML uses)
f = ODEFunction(basic_f!)

println("Testing unwrapping...")
@show f isa ODEFunction
@show f.f === basic_f!

# Try tracing with vars
vars = Symbolics.@variables x[1:1]
du_sym = similar(vars, Symbolics.Num)
f.f(du_sym, vars, nothing, 0.0)
@show du_sym
