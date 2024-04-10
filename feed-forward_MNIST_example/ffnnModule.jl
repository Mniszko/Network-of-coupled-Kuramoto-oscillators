
module basicNeuralNetwork

using FileIO, Images
using Plots
using Random
using LinearAlgebra
using Statistics


export read_ubyte_images, read_mnist_labels, create_batches_with_labels, plot_ubyte_image

export DenseLayer, NeuralNetwork, forward_pass, relu, relu_derivative, softmax, softmax_derivative, linear_layer_derivative, linear_layer_identity, CostFunction, SquareCost, cost, derivative, backpropagation_pass, changing_weights

#=
MNIST readout

functions for reading MNIST dataset from ubyte format, making it useful for implementation here, batching and visualising
=#

function read_ubyte_images(filename)
    open(filename) do file
        magic_number = read(file, UInt32)
        num_images = ntoh(read(file, UInt32))
        num_rows = ntoh(read(file, UInt32))
        num_cols = ntoh(read(file, UInt32))
        images = Array{Float64}(undef, num_rows, num_cols, num_images) # Use Float64 array
        for i = 1:num_images
            for j = 1:num_rows
                for k = 1:num_cols
                    single_image = read(file, UInt8)
                    # Convert to Float64 and normalize
                    images[j, k, i] = single_image / 255.0
                end
            end
        end
        return images
    end
end

function read_mnist_labels(filename)
    open(filename) do file
        magic_number = read(file, UInt32)
        num_labels = ntoh(read(file, UInt32))
        labels_uint8 = Array{UInt8}(undef, num_labels)
        read!(file, labels_uint8)
        # Initialize a matrix for one-hot encoded labels
        labels_one_hot = Array{Float64}(undef, 10, num_labels)
        fill!(labels_one_hot, 0.0)  # Fill with 0.0
        
        for i in 1:num_labels
            label = labels_uint8[i]
            labels_one_hot[label + 1, i] = 1.0  # MNIST labels are 0-9, Julia arrays are 1-indexed
        end
        
        return labels_one_hot
    end
end

function create_batches_with_labels(images, labels, batch_size)
    num_images = size(images, 3)
    image_size = size(images, 1) * size(images, 2)
    
    # Flatten images
    flattened_images = reshape(images, image_size, num_images)
    
    # Shuffle images and labels with the same permutation
    permuted_indices = randperm(num_images)
    shuffled_images = flattened_images[:, permuted_indices]
    
    # For one-hot encoded labels, we need to shuffle columns, not rows
    shuffled_labels = labels[:, permuted_indices]
    
    # Split into batches
    image_batches = [shuffled_images[:, i:min(i+batch_size-1, end)] for i in 1:batch_size:num_images]
    label_batches = [shuffled_labels[:, i:min(i+batch_size-1, end)] for i in 1:batch_size:num_images]

    return image_batches, label_batches
end

function plot_ubyte_image(filename, number)
    images = read_ubyte_images(filename)
    image = images[:, :, number]
    image = rotl90(transpose(image))
    plot = heatmap(image, color=:grays, colorbar=false, axis=false, aspect_ratio=:equal)
    xlims!(plot, (0,28)); ylims!(plot, (0,28))
    return plot
end

#=
Fully Connected Feed Forward Neural Network

functions and structures for constructing neural network, its inference, backpropagation and weight updates
=#

struct DenseLayer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::Function
    activation_derivative::Function
end

#initialization
function DenseLayer(input_dim::Int, output_dim::Int, activation::Function, activation_derivative::Function)
    weights = randn(Float64, output_dim, input_dim) * sqrt(2.0 / input_dim)  # Ensure input_dim and output_dim are Integers
    bias = zeros(Float64, output_dim)
    return DenseLayer(weights, bias, activation, activation_derivative)
end

struct NeuralNetwork
    layers::Vector{DenseLayer}
    #activations serve purpose of calculating gradient without repeating inference
    activations::Vector{Vector{Float64}}
end

#initialization
function NeuralNetwork(layers::DenseLayer...)
    return NeuralNetwork(collect(layers), Vector{Vector{Float64}}())
end

#this isn't pure inference pass, for inference another forward_pass function should be constructed that doesn't take into account storing activations for training
function forward_pass(model::NeuralNetwork, input::Vector{Float64})
    empty!(model.activations)
    x = input
    for layer in model.layers
        push!(model.activations, x)
        x = layer.activation(layer.weights * x .+ layer.bias) 
    end
    return x
end

#would work better if polymorphic (like cost function)
function relu(x::AbstractVector)
    return max.(0f0,x)
end

#matrix relu
function relu_derivative(x::AbstractMatrix)
    rows, cols = size(x)
    answer = Array{Float64}(undef, rows, cols)
    for i in 1:rows
        for j in 1:cols
            if x[i, j] < 0
                answer[i, j] = 0.0
            else
                answer[i, j] = 1.0
            end
        end
    end
    return answer
end

#vector relu
function relu_derivative(x::AbstractVector)
    answer = []
    for val in x
        if val < 0
            push!(answer,Float64(0.))
        else
            push!(answer,Float64(1.))
        end
    end
    return answer
end

function softmax(x::AbstractVector)
    exp_x = exp.(x .- maximum(x))  # Subtract max for numerical stability
    return exp_x ./ sum(exp_x)
end

function softmax_derivative(x::AbstractArray)
    softmax_vals = softmax(x)
    return softmax_vals .* (1 .- softmax_vals)
end

linear_layer_identity(x) = x 
linear_layer_derivative(x) = Ones(size(x))

abstract type CostFunction end
struct SquareCost <: CostFunction end

#interface generics
cost(::CostFunction, output, label) = error("Not implemented") #cost
derivative(::CostFunction, output, label) = error("Not implemented") #costs derivative

#functions for square cost
cost(::SquareCost, output, label) = sum((output - label) .^ 2)
derivative(::SquareCost, output, label) = 2 .* (output .- label)

function backpropagation_pass(model::NeuralNetwork, output::Vector{Float64}, label::Vector{Float64}, costModel::CostFunction)
    delta_L =  model.layers[end].activation_derivative(
        derivative(costModel, output, label)
    )
    delta_l_previous = delta_L

    #for bigger nets it should be a global variable of given dimensions, to prevent lag during computation
    gradients = Vector{Matrix{Float64}}()
    
    #iterating through full array
    for i in length(model.layers):-1:1
        layer = model.layers[i]
        # activation from l-1 layer - thats why before delta_l redefinition
        activation_vec = model.activations[i]
        grad = delta_l_previous * transpose(activation_vec) #I'm not sure about this matrix multiplication
        push!(gradients, grad)
        delta_l = layer.activation_derivative(transpose(layer.weights) * delta_l_previous)
        delta_l_previous = delta_l
    end
    reverse!(gradients)

    #changing weights
    return gradients
end
function changing_weights(model::NeuralNetwork, gradients::Vector{Matrix{Float64}}, step_epsilon::Float64)
    for i in 1:length(model.layers)
        model.layers[i].weights .-= (step_epsilon .* gradients[i])
    end 
end

end