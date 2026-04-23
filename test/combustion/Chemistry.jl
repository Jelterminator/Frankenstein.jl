###############################################################################
# Chemical Combustion Software - Mechanism Loader
#
# This script initializes the environment, imports necessary modules, and loads
# the GRI-Mech 3.0 chemical mechanism from a YAML file for use in combustion
# simulations.
#
# The chemical mechanism file can be obtained from:
# https://chemistry.cerfacs.fr/en/home/
#
# Author: J.S. van der Heide
# Date: 8-2-2025
###############################################################################

function load_mechanism_data(filename::String)::Dict
    """
    Load chemical mechanism data from a YAML file.

    # Arguments
    - `filename::String`: Path to the YAML file containing the chemical mechanism data.

    # Returns
    - `mechanism_data::Dict`: Parsed data from the YAML file.

    # Raises
    - `error`: If the file does not exist or cannot be loaded.
    """
    # Verify that the file exists
    if !isfile(filename)
        error("File $(filename) not found. Please ensure the file path is correct.")
    end

    # Initialize mechanism_data
    mechanism_data = Dict()

    # Load and parse the YAML file
    try
        mechanism_data = YAML.load_file(filename)
    catch e
        error("Failed to load $(filename): $(e)")
    end

    return mechanism_data
end

###############################################################################
# Chemical Combustion Software - Chemical Data Processing (AD-Compatible)
#
# This script processes the chemical data loaded from the YAML mechanism file.
# It extracts species information, reaction data, and thermodynamic properties,
# organizing them into AD-compatible, type-stable data structures for
# combustion simulations.
#
# Modified later because of AD type compatibility needs
#
# Modified again to add transport data processing for later k and D computation
#
# Author: J.S. van der Heide
# Date: 19-8-2025
###############################################################################

# Assuming 'mechanism_data' variable contains the loaded YAML data from the
# previous steps (as obtained from the 'load_mechanism_data' function).

# --- Abstract Type Definitions for AD-Compatibility ---

"""
Abstract supertype for all reaction rate parameter structures.
This allows for type-stable handling of different reaction kinetics.
"""
abstract type AbstractRateParameters end

"""
Abstract supertype for all Troe parameter structures.
"""
abstract type AbstractTroeParameters end


# --- Data Structures ---

"""
Represents the absence of rate parameters. This is a singleton type used as a
type-stable replacement for `nothing`.
"""
struct NoRateParameters <: AbstractRateParameters end

"""
Represents the absence of Troe parameters. This is a singleton type used as a
type-stable replacement for `nothing`.
"""
struct NoTroeParameters <: AbstractTroeParameters end

"""
    ThermoData(T_low::Float64, T_mid::Float64, T_high::Float64, coeffs_low::NTuple{7, Float64}, coeffs_high::NTuple{7, Float64})

Represents thermodynamic data using NASA polynomials for a species. (Unchanged)
"""
struct ThermoData
    T_low::Float64                  # Lower temperature limit
    T_mid::Float64                  # Midpoint temperature
    T_high::Float64                 # Upper temperature limit
    coeffs_low::NTuple{7, Float64}  # Coefficients for T_low ≤ T ≤ T_mid
    coeffs_high::NTuple{7, Float64} # Coefficients for T_mid < T ≤ T_high
end

"""
    TransData(M::Float64, d::Float64, eps::Float64)

Represents transport data using NASA polynomials for a species. 
"""
struct TransData
    M::Float64                  # Molar mass
    d::Float64                  # Lennard-Jones diameter (Å)
    eps::Float64                # Lennard-Jones well depth (K)
end

"""
    Species(name::String, composition::Dict{String, Int}, thermo::ThermoData)

Represents a chemical species with its name, atomic composition, and thermodynamic data. (Unchanged)
"""
struct Species
    name::String                   # Species name (e.g., "CH4")
    composition::Dict{String, Int} # Atomic composition (e.g., Dict("C" => 1, "H" => 4))
    thermo::ThermoData             # Thermodynamic data (NASA polynomials)
    trans::TransData               # Transport data
end

"""
    ReactionParameters(A::Float64, b::Float64, Ea::Float64)

Holds the Arrhenius parameters for a reaction. Now subtypes AbstractRateParameters.
"""
struct ReactionParameters <: AbstractRateParameters
    A::Float64     # Pre-exponential factor
    b::Float64     # Temperature exponent
    Ea::Float64    # Activation energy (in J/mol)
end

"""
    TroeParameters(A::Float64, T3::Float64, T1::Float64, T2::Float64)

Holds the Troe parameters for a reaction. Now subtypes AbstractTroeParameters.
"""
struct TroeParameters <: AbstractTroeParameters
    A::Float64
    T3::Float64
    T1::Float64
    T2::Float64
end

# Outer constructor accepting keyword arguments
TroeParameters(; A::Float64, T3::Float64, T1::Float64, T2::Float64) = TroeParameters(A, T3, T1, T2)

"""
    Reaction(equation::String, ...)

Represents a chemical reaction. The parameter fields now use abstract types
instead of Unions for type stability, making them compatible with AutoEnzyme.
"""
struct Reaction
    equation::String                   # Reaction equation
    reactant_indices::Vector{Int}      # Indices of reactant species
    reactant_stoich::Vector{Float64}   # Stoichiometric coefficients of reactants
    product_indices::Vector{Int}       # Indices of product species
    product_stoich::Vector{Float64}    # Stoichiometric coefficients of products
    rate_params::AbstractRateParameters      # For elementary reactions
    low_rate_params::AbstractRateParameters  # For falloff reactions
    high_rate_params::AbstractRateParameters # For falloff reactions
    troe_params::AbstractTroeParameters      # Troe parameters if applicable
    reaction_type::String              # Type of reaction (e.g., "elementary", "three-body", "falloff")
    efficiencies::Dict{Int, Float64}   # Third-body efficiencies (if applicable)
end

# Function to process chemical data from the mechanism
function process_chemical_data(mechanism_data::Dict)
    # This function's logic remains the same, but it calls the updated helpers.
    species_names = mechanism_data["phases"][1]["species"]
    species_index_map = Dict{String, Int}(name => i for (i, name) in enumerate(species_names))

    species_list = Vector{Species}(undef, length(species_names))
    for (i, name) in enumerate(species_names)
        species_data = find_species_data(name, mechanism_data["species"])
        if species_data === nothing
            error("Species data for $(name) not found in mechanism data.")
        end
        composition_data = get(species_data, "composition", Dict{String, Int}())
        composition = Dict{String, Int}(String(k) => Int(v) for (k, v) in composition_data)
        thermo_data = process_thermo_data(species_data["thermo"])
        trans_data = process_transport_data(composition, species_data["transport"])
            
        species_list[i] = Species(name, composition, thermo_data, trans_data)
    end

    reaction_list = Vector{Reaction}(undef, length(mechanism_data["reactions"]))
    for (i, reaction_data) in enumerate(mechanism_data["reactions"])
        reaction = process_reaction_data(reaction_data, species_index_map)
        reaction_list[i] = reaction
    end

    return species_list, reaction_list, species_index_map
end

# Helper functions for finding and processing species data (unchanged)
function find_species_data(name::String, species_data_list::Vector)
    for species_data in species_data_list
        if species_data["name"] == name
            return species_data
        end
    end
    return nothing
end

function process_thermo_data(thermo_info::Dict)
    T_ranges = thermo_info["temperature-ranges"]
    coeffs = thermo_info["data"]
    T_low = Float64(T_ranges[1])
    T_mid = Float64(T_ranges[2])
    T_high = Float64(T_ranges[3])
    coeffs_low = NTuple{7, Float64}(Float64.(coeffs[1]))
    coeffs_high = NTuple{7, Float64}(Float64.(coeffs[2]))
    return ThermoData(T_low, T_mid, T_high, coeffs_low, coeffs_high)
end

function process_transport_data(comp_dict::Dict, transport_info::Dict)::TransData
    
    # 1. Extract Molar Mass (M)
    # We need to calculate this from the composition
    M = 0.0
    for (element, count) in comp_dict
        # Add the atomic mass * number of atoms
        # You will need an atomic mass database for this.
        # Here is a small subset for your common elements:
        atomic_masses = Dict(
            "N" => 14.0067,
            "O" => 15.999,
            "H" => 1.00794,
            "C" => 12.0107,
            "Ar" => 39.948
        )
        M += atomic_masses[element] * count
    end
    
    # 2. Extract Transport parameters
    d = transport_info["diameter"]    # Lennard-Jones diameter (Å)
    eps = transport_info["well-depth"] # Lennard-Jones well-depth (K)
    
    # Create and return the TransData struct
    return TransData(M, d, eps)
end

# Helper function to process reaction data (UPDATED)
function process_reaction_data(reaction_data::Dict, species_index_map::Dict{String, Int})
    equation = reaction_data["equation"]
    reaction_type = get(reaction_data, "type", "elementary")

    # Assuming parse_reaction_equation and parse_stoichiometry exist and are correct
    reactants_str, products_str = parse_reaction_equation(equation)
    reactant_names, reactant_stoich = parse_stoichiometry(reactants_str)
    product_names, product_stoich = parse_stoichiometry(products_str)
    reactant_indices = [species_index_map[name] for name in reactant_names]
    product_indices = [species_index_map[name] for name in product_names]

    # Initialize parameters with their "null" type instances instead of `nothing`
    rate_params::AbstractRateParameters = NoRateParameters()
    low_rate_params::AbstractRateParameters = NoRateParameters()
    high_rate_params::AbstractRateParameters = NoRateParameters()
    troe_params::AbstractTroeParameters = NoTroeParameters()
    efficiencies = Dict{Int, Float64}()

    if haskey(reaction_data, "rate-constant")
        rate_constant = reaction_data["rate-constant"]
        rate_params = ReactionParameters(rate_constant["A"], rate_constant["b"], rate_constant["Ea"])
    end

    if haskey(reaction_data, "low-P-rate-constant")
        low_rate_constant = reaction_data["low-P-rate-constant"]
        low_rate_params = ReactionParameters(low_rate_constant["A"], low_rate_constant["b"], low_rate_constant["Ea"])
    end

    if haskey(reaction_data, "high-P-rate-constant")
        high_rate_constant = reaction_data["high-P-rate-constant"]
        high_rate_params = ReactionParameters(high_rate_constant["A"], high_rate_constant["b"], high_rate_constant["Ea"])
    end

    if haskey(reaction_data, "Troe")
        troe_data = reaction_data["Troe"]
        troe_params = TroeParameters(
            A=Float64(get(troe_data, "A", 0.0)),
            T3=Float64(get(troe_data, "T3", 0.0)),
            T1=Float64(get(troe_data, "T1", 0.0)),
            T2=Float64(get(troe_data, "T2", 0.0))
        )
    end
    
    if haskey(reaction_data, "efficiencies")
        for (species_name, eff) in reaction_data["efficiencies"]
            idx = species_index_map[species_name]
            efficiencies[idx] = Float64(eff)
        end
    end

    return Reaction(
        equation,
        reactant_indices,
        reactant_stoich,
        product_indices,
        product_stoich,
        rate_params,
        low_rate_params,
        high_rate_params,
        troe_params,
        reaction_type,
        efficiencies
    )
end

# Helper function to parse reaction equation
function parse_reaction_equation(equation::AbstractString)
    """
    Parses a chemical reaction equation string into reactants and products.

    # Arguments
    - `equation::AbstractString`: The reaction equation string.

    # Returns
    - `(reactants_str::String, products_str::String)`: Tuple of reactants and products strings.
    """
    # Remove third-body symbols '(+ M)', '+ M', 'M +', and 'M'
    equation = replace(equation, " (+ M)" => "")
    equation = replace(equation, " + M" => "")
    equation = strip(equation)

    # Determine the reaction direction
    if occursin("<=>", equation)
        splitter = "<=>"
    elseif occursin("=>", equation)
        splitter = "=>"
    else
        error("Unknown reaction format: $equation")
    end

    parts = split(equation, splitter)
    if length(parts) != 2
        error("Failed to parse reaction equation: $equation")
    end

    reactants_str = strip(parts[1])
    products_str = strip(parts[2])
    return reactants_str, products_str
end

# Helper function to parse stoichiometry
function parse_stoichiometry(species_str::AbstractString)
    """
    Parses a string of species and their stoichiometric coefficients.

    # Arguments
    - `species_str::AbstractString`: A string containing species and coefficients (e.g., "2 H2 + O2").

    # Returns
    - `species_names::Vector{String}`: List of species names.
    - `stoich_coeffs::Vector{Float64}`: Corresponding stoichiometric coefficients.
    """
    species_terms = split(species_str, "+")
    species_names = String[]
    stoich_coeffs = Float64[]
    for term in species_terms
        term = strip(term)
        # Skip if term is empty or "M"
        if isempty(term) || term == "M"
            continue
        end
        # Match terms like "2 H2" or "H2"
        m = match(r"^(\d*\.?\d+)?\s*(\S+)$", term)
        if m === nothing
            error("Failed to parse stoichiometric term: $term")
        end
        stoich_coeff_str = m.captures[1]
        species_name = m.captures[2]
        if species_name == "M"
            continue  # Skip "M" if it's still present
        end
        if stoich_coeff_str == "" || stoich_coeff_str == nothing
            coeff = 1.0
        else
            coeff = parse(Float64, stoich_coeff_str)
        end
        push!(species_names, species_name)
        push!(stoich_coeffs, coeff)
    end
    return species_names, stoich_coeffs
end

###############################################################################
# Chemical Combustion Software - Build Stoichiometric Matrix and Arrhenius Parameters
#
# This module processes the reactions from the mechanism data to construct:
# - The stoichiometric matrix (S)
# - The Arrhenius parameters for each reaction
# 
# The function returns the stoichiometric matrix and a vector of tuples containing
# the reaction parameters, avoiding the use of dictionaries.
#
# Author: J.S. van der Heide
# Date: 11-7-2025
###############################################################################

# Function to build the stoichiometric matrix
function build_stoichiometric_matrix(
    reaction_list::Vector{Reaction},
    n_species::Int
) :: Matrix{Float64}
    """
    Constructs the stoichiometric matrix S.

    # Arguments
    - `reaction_list::Vector{Reaction}`: List of Reaction structs.
    - `n_species::Int`: Total number of species.

    # Returns
    - `S::Matrix{Float64}`: Stoichiometric matrix of size (n_species, n_reactions).
    """
    n_reactions = length(reaction_list)
    S = zeros(Float64, n_species, n_reactions)

    for (r_idx, reaction) in enumerate(reaction_list)
        # Populate stoichiometric matrix S
        for (idx, coeff) in zip(reaction.reactant_indices, reaction.reactant_stoich)
            S[idx, r_idx] -= coeff  # Reactants have negative coefficients
        end
        for (idx, coeff) in zip(reaction.product_indices, reaction.product_stoich)
            S[idx, r_idx] += coeff  # Products have positive coefficients
        end
    end

    return S
end

# Abstract supertype for all our kinetic models
abstract type AbstractKinetics end

# Struct for elementary/three-body reactions
struct ElementaryKinetics <: AbstractKinetics
    A::Float64
    b::Float64
    Ea::Float64
    is_three_body::Bool
    is_reversible::Bool
    efficiencies::Vector{Tuple{Int, Float64}}
    reactant_indices::Vector{Int}
    reactant_stoich::Vector{Float64}
    product_indices::Vector{Int}
    product_stoich::Vector{Float64}
end

# Struct for falloff reactions
struct FalloffKinetics <: AbstractKinetics
    high_A::Float64; high_b::Float64; high_Ea::Float64
    low_A::Float64;  low_b::Float64;  low_Ea::Float64
    is_reversible::Bool
    troe_params::AbstractTroeParameters
    efficiencies::Vector{Tuple{Int, Float64}}
    reactant_indices::Vector{Int}
    reactant_stoich::Vector{Float64}
    product_indices::Vector{Int}
    product_stoich::Vector{Float64}
end

struct SplitKinetics
    # Data for elementary reactions
    elementary_kinetics::Vector{ElementaryKinetics}
    elementary_indices::Vector{Int} # Maps back to original reaction index

    # Data for falloff reactions
    falloff_kinetics::Vector{FalloffKinetics}
    falloff_indices::Vector{Int} # Maps back to original reaction index
end

function build_kinetics_list(reaction_list::Vector{Reaction}) :: SplitKinetics
    # Initialize empty vectors for each reaction type
    elem_kinetics = ElementaryKinetics[]
    elem_indices = Int[]
    
    falloff_kinetics = FalloffKinetics[]
    falloff_indices = Int[]

    for (r_idx, reaction) in enumerate(reaction_list)
        efficiencies_vec = [(idx, eff) for (idx, eff) in reaction.efficiencies]
        is_reversible = occursin("<=>", reaction.equation)

        # Instead of putting them in one abstract vector, we sort them into
        # separate, concrete vectors.
        if reaction.reaction_type == "elementary" || reaction.reaction_type == "three-body"
            params = reaction.rate_params::ReactionParameters
            k = ElementaryKinetics(
                params.A, params.b, params.Ea,
                reaction.reaction_type == "three-body",
                is_reversible,
                efficiencies_vec,
                reaction.reactant_indices, reaction.reactant_stoich,
                reaction.product_indices, reaction.product_stoich
            )
            push!(elem_kinetics, k)
            push!(elem_indices, r_idx) # Store the original reaction index

        elseif reaction.reaction_type == "falloff"
            high_params = reaction.high_rate_params::ReactionParameters
            low_params = reaction.low_rate_params::ReactionParameters
            k = FalloffKinetics(
                high_params.A, high_params.b, high_params.Ea,
                low_params.A, low_params.b, low_params.Ea,
                is_reversible,
                reaction.troe_params,
                efficiencies_vec,
                reaction.reactant_indices, reaction.reactant_stoich,
                reaction.product_indices, reaction.product_stoich
            )
            push!(falloff_kinetics, k)
            push!(falloff_indices, r_idx) # Store the original reaction index
        else
             error("Unsupported reaction type in reaction $(r_idx)")
        end
    end
    
    return SplitKinetics(elem_kinetics, elem_indices, falloff_kinetics, falloff_indices)
end

###############################################################################
# Chemical Combustion Software - Thermodynamic Properties Computation
#
# This module provides functions to compute thermodynamic properties such as
# specific heat capacity (Cp), standard enthalpy (h0), and entropy (s0) using
# NASA polynomials. It also includes the function `dT` to compute the rate of
# change of temperature during the reaction process.
#
# Modified for change to constant volume
#
# Author: J.S. van der Heide
# Date: 20-8-2025
###############################################################################

# Universal gas constant in 
const R_joule = 8.314462 # J/(mol·K)
const R_cal = 1.987204   # cal/(mol·K)

# Units are weird: the activation energy is in cal/mol and the thermodynamics are in J/mol

# Compute specific heat capacity Cp(T) using NASA polynomials
"""
    species_cp(T::Float64, thermo_data::ThermoData) -> Float64

Computes the specific heat capacity at constant pressure (Cp) for a species at temperature `T`.

# Arguments
- `T::Float64`: Temperature in Kelvin.
- `thermo_data::ThermoData`: Thermodynamic data for the species.

# Returns
- `Cp::Float64`: Specific heat capacity in J/(mol·K).

# Notes
- Uses NASA polynomials to compute dimensionless Cp/R, then multiplies by R to get Cp.
"""
function species_cp(T::Real, thermo_data::ThermoData)
    coeffs_low = thermo_data.coeffs_low
    coeffs_high = thermo_data.coeffs_high
    T_switch = thermo_data.T_mid

    # --- The "Calculate Both, Then Select" Pattern ---

    # 1. Calculate the result for the LOW-temperature branch
    Cp_R_low = coeffs_low[1] + coeffs_low[2]*T + coeffs_low[3]*T^2 + coeffs_low[4]*T^3 + coeffs_low[5]*T^4
    
    # 2. Calculate the result for the HIGH-temperature branch
    Cp_R_high = coeffs_high[1] + coeffs_high[2]*T + coeffs_high[3]*T^2 + coeffs_high[4]*T^3 + coeffs_high[5]*T^4

    # 3. Create a selector. This is 1.0 if T < T_switch, and 0.0 otherwise.
    #    This is differentiable almost everywhere and keeps the graph connected for AD.
    is_low = Float64(T < T_switch)
    
    # 4. Combine the results using the selector.
    #    If is_low is true (1.0), we get Cp_R_low.
    #    If is_low is false (0.0), we get Cp_R_high.
    Cp_R = is_low * Cp_R_low + (1 - is_low) * Cp_R_high
    
    return Cp_R * R_joule
end

# Compute standard enthalpy h0(T) using NASA polynomials
"""
    h0(T::Float64, thermo_data::ThermoData) -> Float64

Computes the standard enthalpy (h0) for a species at temperature `T`.

# Arguments
- `T::Float64`: Temperature in Kelvin.
- `thermo_data::ThermoData`: Thermodynamic data for the species.

# Returns
- `h::Float64`: Enthalpy in J/mol.

# Notes
- Uses NASA polynomials to compute dimensionless h/RT, then multiplies by R·T to get h.
"""
function h0(T::Real, thermo_data::ThermoData)
    coeffs_low = thermo_data.coeffs_low
    coeffs_high = thermo_data.coeffs_high
    T_switch = thermo_data.T_mid

    # Calculate both branches
    H_RT_low = coeffs_low[1] + coeffs_low[2]*(T/2) + coeffs_low[3]*(T^2)/3 + coeffs_low[4]*(T^3)/4 + coeffs_low[5]*(T^4)/5 + coeffs_low[6]/T
    H_RT_high = coeffs_high[1] + coeffs_high[2]*(T/2) + coeffs_high[3]*(T^2)/3 + coeffs_high[4]*(T^3)/4 + coeffs_high[5]*(T^4)/5 + coeffs_high[6]/T
    
    # Select and combine
    is_low = Float64(T < T_switch)

    H_RT = is_low * H_RT_low + (1 - is_low) * H_RT_high
    
    return H_RT * R_joule * T
end

# Compute standard entropy s0(T) using NASA polynomials
"""
    species_entropy(T::Float64, thermo_data::ThermoData) -> Float64

Computes the standard entropy (s0) for a species at temperature `T`.

# Arguments
- `T::Float64`: Temperature in Kelvin.
- `thermo_data::ThermoData`: Thermodynamic data for the species.

# Returns
- `s::Float64`: Entropy in J/(mol·K).

# Notes
- Uses NASA polynomials to compute dimensionless S/R, then multiplies by R to get s.
"""
function species_entropy(T::Real, thermo_data::ThermoData)
    coeffs_low = thermo_data.coeffs_low
    coeffs_high = thermo_data.coeffs_high
    T_switch = thermo_data.T_mid

    # Calculate both branches
    S_R_low = coeffs_low[1] * log(T) + coeffs_low[2]*T + coeffs_low[3]*(T^2)/2 + coeffs_low[4]*(T^3)/3 + coeffs_low[5]*(T^4)/4 + coeffs_low[7]
    S_R_high = coeffs_high[1] * log(T) + coeffs_high[2]*T + coeffs_high[3]*(T^2)/2 + coeffs_high[4]*(T^3)/3 + coeffs_high[5]*(T^4)/4 + coeffs_high[7]
    
    # Select and combine
    is_low = Float64(T < T_switch)
    S_R = is_low * S_R_low + (1 - is_low) * S_R_high
    
    return S_R * R_joule
end

###############################################################################
# Chemical Combustion Software - Reaction Rate Computation
#
# This module defines the `compute_reaction_rates` function (formerly `Arrhenius`)
# that calculates the reaction rates using the Arrhenius equation and handles
# different reaction types (elementary, three-body, falloff).
#
# It uses the updated data structures and integrates with the rest of the code.
#
# Modified later because of AD type compatibility needs
#
# Modified again for change to constant volume
#
# Author: J.S. van der Heide
# Date: 20-8-2025
###############################################################################

# Function to compute reaction rates
"""
    compute_reaction_rates(
        X::Vector{Float64},
        S::Array{Float64,2},
        kinetics_list::Vector{Any},
        species_list::Vector{Species},
        reaction_list::Vector{Reaction},
        species_index_map::Dict{String, Int}
    ) -> Vector{Float64}

Computes the reaction rates for all reactions given the current state.

# Arguments
- `X::Vector{Float64}`: State vector containing temperature and species concentrations.
  - `X[1]`: Temperature in Kelvin.
  - `X[2:end]`: Species concentrations in mol/m³.
- `S::Array{Float64,2}`: Stoichiometric matrix of size (n_species, n_reactions).
- `kinetics_list::Vector{Any}`: List of tuples containing kinetic parameters for each reaction.
- `species_list::Vector{Species}`: List of `Species` structs.
- `reaction_list::Vector{Reaction}`: List of `Reaction` structs.
- `species_index_map::Dict{String, Int}`: Mapping of species names to indices.

# Returns
- `r::Vector{Float64}`: Reaction rates for each reaction in mol/(m³·s).

# Notes
- Handles different reaction types: elementary, three-body, and falloff.
- Computes reverse reaction rates for reversible reactions.
- Assumes units are calorie, cm3 and Kelvin.
"""

# --- Main Entry Point ---

function compute_reaction_rates!(
    r::AbstractVector{<:Real},
    X::AbstractVector{<:Real},
    kinetics_data::SplitKinetics,
    species_list::Vector{Species}
)
    T = X[1]
    concentrations_m3 = @view X[2:end]
    # Convert to /cm3 for Arrhenius equation
    # Also use a zero-is-zero clamp to prevent negative concentrations (see thesis paper)
    # Using a soft clamp for AD compatibility
    concentrations_cm3 = max.(concentrations_m3 * 1e-6, 1e-25)
    
    # Initialize all reaction rates to zero
    fill!(r, 0.0)
    
    # --- Loop 1: Elementary Reactions ---
    for i in eachindex(kinetics_data.elementary_kinetics)
        k = kinetics_data.elementary_kinetics[i]
        original_idx = kinetics_data.elementary_indices[i]
        
        r[original_idx] = calculate_q_dot(k, T, concentrations_cm3, species_list) * 1e6
    end
    
    # --- Loop 2: Falloff Reactions ---
    for i in eachindex(kinetics_data.falloff_kinetics)
        k = kinetics_data.falloff_kinetics[i]
        original_idx = kinetics_data.falloff_indices[i]
        
        r[original_idx] = calculate_q_dot(k, T, concentrations_cm3, species_list) * 1e6
    end
    
    return nothing
end


# --- Helper Functions for Rate Calculations ---

# Helper for third-body concentration [M]
function calculate_third_body_M(efficiencies::Vector{Tuple{Int, Float64}}, C::AbstractVector{<:Real})
    M = 0.0
    # Use a Set for faster lookups if n_species is large, otherwise this is fine
    eff_species_indices = Set(idx for (idx, _) in efficiencies)
    
    # Add contributions from species with specified efficiencies
    for (species_idx, eff) in efficiencies
        M += eff * C[species_idx]
    end
    
    # Add contributions from species with default efficiency of 1.0
    for i in 1:length(C)
        if i ∉ eff_species_indices
            M += C[i]
        end
    end
    return M
end

# Dispatched helper for Troe blending factor F. This is for the Troe case.
function get_blending_factor_F(params::TroeParameters, T::Real, Pr::Real)
    # Unpack to ensure compiler sees variables
    troe_alpha, T3, T1, T2 = params.A, params.T3, params.T1, params.T2
    
    F_cent_term1 = (1 - troe_alpha) * exp(-T / T3)
    F_cent_term2 = troe_alpha * exp(-T / T1)
    F_cent_term3 = exp(-T2 / T)
    F_cent = F_cent_term1 + F_cent_term2 + F_cent_term3
    
    logF_cent = log10(F_cent)
    
    # Handle Pr=0 case to avoid log(0)
    if Pr > 1e-20
        logPr = log10(Pr)
        c = -0.4 - 0.67 * logPr
        n = 0.75 - 1.27 * logPr
        f = (logF_cent) / (1 + (logPr / n)^2)
        logF = c * f
        return 10.0^logF
    else
        return 1.0 # Or appropriate limit as Pr -> 0
    end
end

# Dispatched helper for Troe blending factor F. This is for the Lindemann case.
# It is called when `troe_params` is a `NoTroeParameters` object.
function get_blending_factor_F(params::NoTroeParameters, T::Real, Pr::Real)
    return 1.0 # Lindemann mechanism, no special blending factor
end


# --- Main Dispatched Rate Functions ---

# Method for Elementary and Three-Body Reactions
function calculate_q_dot(kinetics::ElementaryKinetics, T::Real, C::AbstractVector{<:Real}, species::Vector{Species})
    # Forward rate constant
    k_f = kinetics.A * T^kinetics.b * exp(-kinetics.Ea / (R_cal * T))
    
    if kinetics.is_three_body
        M = calculate_third_body_M(kinetics.efficiencies, C)
        k_f *= M
    end

    # Reactant concentrations in log space
    C_reactants = 0
    for i in 1:length(kinetics.reactant_indices)
        C_reactants += log(C[kinetics.reactant_indices[i]]) * kinetics.reactant_stoich[i]
    end
    
    r_fwd = k_f * exp(C_reactants)
    
    if kinetics.is_reversible
        # Compute reverse rate in log space
        C_products = 0.0
        for i in 1:length(kinetics.product_indices)
            C_products += log(C[kinetics.product_indices[i]]) * kinetics.product_stoich[i]
        end
        
        delta_F = compute_reaction_delta_F(T, kinetics, species)
        # Avoid division by zero if k_f is tiny
        K_c = exp(-delta_F / (R_joule * T))
        k_rev = k_f / K_c 
        r_rev = k_rev * exp(C_products)
        return r_fwd - r_rev
    else
        return r_fwd
    end
end

# Method for Falloff Reactions
function calculate_q_dot(kinetics::FalloffKinetics, T::Real, C::AbstractVector{<:Real}, species::Vector{Species})
    # High-pressure limit rate constant k_inf
    k_inf = kinetics.high_A * T^kinetics.high_b * exp(-kinetics.high_Ea / (R_cal * T))
    # Low-pressure limit rate constant k_0
    k_0 = kinetics.low_A * T^kinetics.low_b * exp(-kinetics.low_Ea / (R_cal * T))
    
    M = calculate_third_body_M(kinetics.efficiencies, C)
    Pr = (k_0 * M) / k_inf # Reduced pressure
    
    # Get blending factor F by dispatching on the type of troe_params
    F = get_blending_factor_F(kinetics.troe_params, T, Pr)
    
    k_f = k_inf * (1 / (1.0 + Pr)) * F
    
    # --- The rest is identical to the ElementaryKinetics method ---
    
    # Reactant concentrations in log space
    C_reactants = 0
    for i in 1:length(kinetics.reactant_indices)
        C_reactants += log(C[kinetics.reactant_indices[i]]) * kinetics.reactant_stoich[i]
    end
    
    r_fwd = k_f * exp(C_reactants)
    
    if kinetics.is_reversible
        # Compute reverse rate in log space
        C_products = 0.0
        for i in 1:length(kinetics.product_indices)
            C_products += log(C[kinetics.product_indices[i]]) * kinetics.product_stoich[i]
        end
        
        delta_F = compute_reaction_delta_F(T, kinetics, species)
        
        K_c = exp(-delta_F / (R_joule * T))
        
        k_rev = k_f / K_c 
        r_rev = k_rev * exp(C_products)
        return r_fwd - r_rev
    else
        return r_fwd
    end
end

# --- Thermodynamics Calculation ---
function compute_reaction_delta_F(T::Real, kinetics::AbstractKinetics, species_list::Vector{Species})
    delta_U = 0.0
    delta_S = 0.0
    
    # Reactants
    for i in 1:length(kinetics.reactant_indices)
        spec = species_list[kinetics.reactant_indices[i]]
        stoich = kinetics.reactant_stoich[i]        
        
        delta_U -= stoich * (h0(T, spec.thermo) - R_joule * T)
        delta_S -= stoich * species_entropy(T, spec.thermo)
    end
    
    # Products
    for i in 1:length(kinetics.product_indices)
        spec = species_list[kinetics.product_indices[i]]
        stoich = kinetics.product_stoich[i]        
        
        delta_U += stoich *(h0(T, spec.thermo) - R_joule * T)
        delta_S += stoich * species_entropy(T, spec.thermo)
    end
    
    return delta_U - T * delta_S
end

###############################################################################
# Chemical Combustion Software - Diffusion and conduction coefficients
#
# Estimates for kappa and D that presume all of the gas is pure nitrogen,
# Which is only 20% wrong...
#
# Author: J.S. van der Heide
# Date: 19-8-2025
###############################################################################

const N2_prefactor = (6 / (3 * (3.707* 1e-10)^2)) * sqrt((1.380649e-23^3) / (3.141592^3 * 28.0134 / 6.02214076e26))

function conductivity(T::Real)
    return N2_prefactor * T^0.5
end

function diffusivity(transA::TransData, transB::TransData, T::Real, P::Real)::Real
    # Combine parameters using Lorentz-Berthelot rules
    σ_AB = (transA.d + transB.d) / 2.0           # Å
    ϵ_AB = sqrt(transA.eps * transB.eps)         # K
    M_AB = sqrt(1.0/transA.M + 1.0/transB.M)  # g/mol
    
    # Calculate reduced temperature
    T_star = T / ϵ_AB
    
    # Get collision integral
    Ω = collision_integral(T_star)

    # Chapman-Enskog formula (returns D_AB in cm²/s) 
    D_AB = (1.859e-3 * M_AB * (T)^(3/2)) / ( P/101325 * σ_AB^2 * Ω)
    
    # Convert to m²/s
    return D_AB * 1e-4
end
        
function collision_integral(T_star::Real)::Real
    # Constants for Neufeld et al. approximation
    A = 1.06036
    B = 0.15610
    C = 0.19300
    D_val = 0.47635
    E = 1.03587
    F_val = 1.52996
    G = 1.76474
    H_val = 3.89411
    
    Ω = A / (T_star)^B +
        C / exp(D_val * T_star) +
        E / exp(F_val * T_star) +
        G / exp(H_val * T_star)
    
    return Ω
end

###############################################################################
# Chemical Combustion Software - Chemistry Configuration and Initialization
#
# This module defines the chemistry configuration parameters and initializes
# the species concentrations based on the provided configuration and mechanism
# data.
#
# Author: J.S.van der Heide
# Date: 8-2-2025
###############################################################################

# Chemistry configuration struct with detailed documentation
struct ChemistryConfig
    temperature::Float64                  # Temperature (K)
    pressure::Float64                     # Pressure (Pa)
    fuel_mixture::Dict{String, Float64}   # Fuel components and their percentages
    air_percentage::Float64               # Percentage of air in the mixture
end

# Outer constructor accepting keyword arguments
function ChemistryConfig(; temperature::Float64, pressure::Float64, fuel_mixture::Dict{String, Float64}, air_percentage::Float64)
    ChemistryConfig(temperature, pressure, fuel_mixture, air_percentage)
end

# Helper function to initialize concentrations
function initialize_concentrations(
    config::ChemistryConfig,
    species_list::Vector{Species},
    species_index_map::Dict{String, Int}
) :: Vector{Float64}

    # Constants
    R = 8.3145  # Universal gas constant (J/mol·K)

    # Total concentration (mol/m³) using the ideal gas law
    c_tot = config.pressure / (R * config.temperature)

    n_species = length(species_list)

    # Initialize concentration vector
    # The first element is reserved for temperature
    concentrations = zeros(Float64, n_species + 1)
    concentrations[1] = config.temperature  # Set initial temperature

    # Define standard air composition percentages (by mole fraction)
    air_composition = Dict(
        "N2"   => 0.78084,
        "O2"   => 0.20946
        # Add more components if necessary
    )

    # Calculate air concentrations
    air_mole_fractions = Dict{String, Float64}()
    for (species_name, fraction) in air_composition
        air_mole_fractions[species_name] = fraction * config.air_percentage
    end

    # Add fuel components to mole fractions
    total_fuel_percentage = sum(values(config.fuel_mixture))
    if total_fuel_percentage + config.air_percentage > 1
        error("Total fuel percentage exceeds available mixture fraction.")
    end
    for (species_name, fraction) in config.fuel_mixture
        air_mole_fractions[species_name] = get(air_mole_fractions, species_name, 0.0) + fraction
    end

    # Normalize mole fractions to ensure they sum up to total mixture
    total_mole_fraction = sum(values(air_mole_fractions))
    for key in keys(air_mole_fractions)
        air_mole_fractions[key] /= total_mole_fraction
    end

    # Assign concentrations to each species
    for (species_name, mole_fraction) in air_mole_fractions
        if haskey(species_index_map, species_name)
            index = species_index_map[species_name]
            concentrations[index + 1] = mole_fraction * c_tot  # +1 because temperature is at index 1
        else
            # Species not in mechanism; ignore or handle accordingly
            @warn "Species $(species_name) not found in mechanism data."
        end
    end

    # Any remaining species not specified are set to zero concentration
    # Optionally, you can also set a small background concentration if needed

    return concentrations
end


###############################################################################
# Chemical Combustion Software - Miscellaneous Functions
#
# This module defines plotting and initialising functions.
#
# Author: J.S. van der Heide
# Date: 8-2-2025
###############################################################################

# Function to plot temperature over time
function plot_temperature(t::Vector{Float64}, T::Vector{Float64}, tspan)
    plot(t, T,
         xlabel = "Time (s)",
         ylabel = "Temperature (K)",
         title = "Temperature Evolution (Tspan = $tspan)",
         legend = false)
    display(current())
end

# Function to plot species concentrations over time
function plot_concentrations(t::Vector{Float64}, concentrations::Array{Float64,2}, species_list::Vector{Species}, tspan)
    num_species = length(species_list)
    species_names = reshape([species.name for species in species_list], 1, :)

    # For clarity, you might select a subset of species to plot
    # Here, we plot all species
    plot(t, concentrations',
         xlabel = "Time (s)",
         ylabel = "Concentration (mol/m³)",
         title = "Species Concentrations (Tspan = $tspan)",
         label = species_names)
    display(current())
end

# Initialising the simulation parameters
function initialise(config, YAML_name)
    @info "------------------------------------------------------------"
    @info "Loading GRI-Mech 3.0 Mechanism Data"
    @info "------------------------------------------------------------"

    # Load the mechanism file
    mechanism_data = load_mechanism_data(YAML_name)

    # Verify that essential data fields are present
    if !haskey(mechanism_data, "phases") || !haskey(mechanism_data, "species") || !haskey(mechanism_data, "reactions")
        error("Mechanism data is missing essential fields ('phases', 'species', or 'reactions').")
    end

    # Process the chemical data
    species_list, reaction_list, species_index_map = process_chemical_data(mechanism_data)

    # Number of species and reactions
    num_species = length(species_list)
    num_reactions = length(reaction_list)

    @info "Number of species: $num_species"
    @info "Number of reactions: $num_reactions"

    # Initialize concentrations based on the configuration
    initial_concentrations = initialize_concentrations(config, species_list, species_index_map)

    # Build the stoichiometric matrix and kinetics tuples
    S, kinetics_tuples = build_chemistry_matrices(reaction_list, num_species)

    return (initial_concentrations, (S, kinetics_tuples, species_list, reaction_list, species_index_map))
end

###############################################################################
# Chemical Combustion Software - Verification functions
#
# This module defines functions that test if simulations are correct.
#
# Author: J.S. van der Heide
# Date: 20-8-2025
###############################################################################


# A function to check what the amount of atoms of a given concentration vector is
function atom_counter(X0, YAML_name, species_index_map)
    # Get data
    data = load_mechanism_data(YAML_name)
    
    # Initialize atom counts
    H = 0.0
    O = 0.0
    C = 0.0
    
    # Loop over all species and sum up atom counts
    for j in 1:length(data["species"])
        try
            H += data["species"][j]["composition"]["H"] * X0[1+species_index_map[data["species"][j]["name"]]]
        catch e
            H += 0
        end
        
        try
            O += data["species"][j]["composition"]["O"] * X0[1+species_index_map[data["species"][j]["name"]]]
        catch e
            O += 0
        end
        
        try
            C += data["species"][j]["composition"]["C"] * X0[1+species_index_map[data["species"][j]["name"]]]
        catch e
            C += 0
        end
    end

    # Print total atom counts
    @info "The amount of hydrogen atoms is $(H) mol"
    @info "The amount of oxygen atoms is $(O) mol"
    @info "The amount of carbon atoms is $(C) mol"
end

function energy_accounting(sol, species_list, species_index_map)

    # Define species categories
    fuel_species = ["CH4", "C2H6", "C3H8", "H2"]
    product_species = ["CO2", "H2O", "CO"]
    dissociation_species = setdiff(
        [species.name for species in species_list],
        union(fuel_species, product_species, ["N2", "AR"])  # Exclude inerts
    )

    # Get time points and solution data
    t = sol.t  # Time points
    n_times = length(t)
    n_species = length(species_list)
    X = sol.u  # Solution array: [T; concentrations]
    R_joule = 8.31446261815324  # Gas constant (J/(mol·K))

    # Initialize energy arrays (J/m³)
    fuel_energy = zeros(n_times)
    dissociation_energy = zeros(n_times)
    product_energy = zeros(n_times)
    heat_energy = zeros(n_times)

    # Compute energy for each time point
    for i in 1:n_times
        T = X[i][1]  # Temperature (K)
        concentrations = X[i][2:end]  # Species concentrations (mol/m³)

        # Compute species enthalpies and heat capacities
        u_species = zeros(n_species)  # Enthalpy (J/mol)
        cv_vec = zeros(n_species)     # Heat capacity at constant volume (J/(mol·K))
        for (species_index, species) in enumerate(species_list)
            cv_vec[species_index] = species_cp(T, species.thermo) - R_joule
            u_species[species_index] = h0(T, species.thermo) - R_joule * T
        end

        # Compute total heat capacity (J/(m³·K))
        c_v = sum(concentrations .* cv_vec)

        # Compute heat energy (J/m³)
        heat_energy[i] = c_v * T

        # Sum enthalpy (J/mol * mol/m³ = J/m³) for each category
        for name in [species.name for species in species_list]
            if name in fuel_species
                idx = species_index_map[name]
                fuel_energy[i] += u_species[idx] * concentrations[idx]
            end
            if name in dissociation_species
                idx = species_index_map[name]
                dissociation_energy[i] += u_species[idx] * concentrations[idx]
            end
            if name in product_species
                idx = species_index_map[name]
                product_energy[i] += u_species[idx] * concentrations[idx]
            end
        end
    end

    return t, fuel_energy, dissociation_energy, product_energy, heat_energy
end