using Oceananigans.TimeSteppers: reset!
using Printf

# Run the filtering operation. This is the main function that a user will call to perform offline filtering.
# The utility functions used here can also be called individually if desired to create a custom filtering workflow.

"""
    run_offline_Lagrangian_filter(config::OfflineFilterConfig)

Runs an offline Lagrangian filter on an Oceananigans `FieldTimeSeries` dataset as configured by `config`.

This function performs a series of steps to filter the data:
1.  **Prepare data on disk**: The input data is copied and manipulated on disk to be suitable for the forward and backward Lagrangian simulations.
2.  **Run forward simulation**: A `LagrangianFilter` model is created and run forward in time to compute the first half of the filter contributions.
3.  **Run backward simulation**: The input data is re-prepared for a backward pass, and the simulation is run a second time to compute the remaining contributions.
4.  **Combine results**: The forward and backward simulation outputs are summed to produce the final filtered data.
5.  **Post-processing**: Optional post-processing steps are performed, including regridding the data to the mean position, computing a comparative Eulerian filter, and converting the output file to NetCDF.
6.  **Cleanup**: Intermediate files are removed to save disk space.

Arguments
=========

- `config`: An instance of `OfflineFilterConfig` that specifies all parameters and file paths for the filtering process.
"""
function run_offline_Lagrangian_filter(config)

    # Copy and manipulate data on disk to have correct order and time shift
    create_input_data_on_disk(config; direction = "forward") 

    # Load in saved data from simulation
    input_data = load_data(config)
    @info "Loaded data from $(config.original_data_filename)"

    # Create the original variables - these will be auxiliary fields in the model
    original_vars = create_original_vars(config)
    @info "Created original variables: $(keys(original_vars))"

    # Create the filtered variables - these will be tracers in the model
    filtered_vars = create_filtered_vars(config)
    @info "Created filtered variables: $filtered_vars"

    # Create forcing for these filtered variables
    forcing = create_forcing(filtered_vars, config)
    @info "Created forcing for filtered variables"

    # Define model 
    model = LagrangianFilter(config.grid; tracers = filtered_vars, auxiliary_fields = original_vars, forcing = forcing, advection=config.advection)
    @info "Created model"

    # We can set initial values to improve the spinup, use the limit freq_c -> \infty
    # The map variables get automatically initialised to zero
    initialise_filtered_vars_from_data(model, input_data, config)    
    @info "Initialised filtered variables"

    # Define our outputs # 
    filtered_outputs = create_output_fields(model, config)
    @info "Defined outputs"

    # Define the filtering simulation 
    simulation = Simulation(model, Δt = config.Δt, stop_time = config.T) 
    @info "Defined simulation"

    # Tell the simulation to use the saved data.
    # Use UpdateStateCallsite so that velocities are updated at each substep if using multi-stage time steppers.
    simulation.callbacks[:update_input_data] = Callback(update_input_data!, callsite = UpdateStateCallsite(), parameters = input_data)

    # Add a progress monitor
    function progress(sim)
        @info @sprintf("Simulation time: %s\n", 
                    prettytime(sim.model.clock.time))             
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress,TimeInterval(config.T/10))

    #Write outputs
    simulation.output_writers[:vars] = JLD2Writer(model, filtered_outputs,
                                                            filename = config.forward_output_filename,
                                                            schedule = TimeInterval(config.T_out),
                                                            overwrite_existing = true)

    # Run forward simulation                                                        
    run!(simulation)


    # Now, run it backwards. Switch the data direction on disk
    create_input_data_on_disk(config; direction = "backward")

    # The filtered variables are already well initialised for the backward run, but the maps need reversing.
    change_sign_of_map_variables!(model, config)
  
    # Reset time
    reset!(model.clock)

    # Write outputs
    simulation.output_writers[:vars] = JLD2Writer(model, filtered_outputs,
                                                            filename = config.backward_output_filename,
                                                            schedule = TimeInterval(config.T_out),
                                                            overwrite_existing = true)

    # And run the backward simulation.
    run!(simulation)

    # Now sum the forward and backward components
    sum_forward_backward_contributions!(config)

    # Clean up temporary files
    # Delete the shifted input data file
    rm(config.original_data_filename[1:end-5] * "_filter_input.jld2")

    # Option to delete the forward and backward output files
    if config.delete_intermediate_files
        rm(config.forward_output_filename)
        rm(config.backward_output_filename)
    end

    # Option to regrid to mean position
    if config.map_to_mean
        regrid_to_mean_position!(config)
    end

    # Option to calculate Eulerian filter too
    if config.compute_Eulerian_filter
        compute_Eulerian_filter!(config)
    end

    # Option to output final netcdf
    if config.output_netcdf
        jld2_to_netcdf(config.output_filename, config.output_filename[1:end-5] * ".nc")
    end

end
