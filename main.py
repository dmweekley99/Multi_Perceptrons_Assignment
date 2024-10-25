import numpy as np

# Initialize weights with random values for two output neurons
def init_wts(input, output):
    # Weights are initialized between -0.5 and 0.5
    return np.random.uniform(-0.5, 0.5, (output, input + 1))  # +1 for bias

# Activation function that applies a threshold
def threshold(z):
    # Returns 1 if z > 0, otherwise returns 0
    return np.where(z > 0, 1, 0)

# Function to calculate output of the neural network
def calculate_output(weights, inputs):
    # Add bias to input by inserting 1 at the start
    inputs_with_bias = np.insert(inputs, 0, 1)
    z = np.zeros(weights.shape[0])  # Create an array to hold the outputs for each neuron
    
    # Calculate the weighted sum for each output neuron
    for i in range(weights.shape[0]):  # For each output neuron
        weighted_sum = 0
        for j in range(weights.shape[1]):  # For each input (including bias)
            weighted_sum += weights[i][j] * inputs_with_bias[j]
        z[i] = weighted_sum  # Store the weighted sum for the neuron
    
    return threshold(z)  # Apply thresholding to get the final output

# Function to classify the region based on (x, y) coordinates
def classify_region(x, y):
    # Returns a target output for each region
    if y > x and y > -x:
        return [1, 0]  # Region 1
    elif y < x and y > -x:
        return [1, 1]  # Region 2
    elif y < x and y < -x:
        return [0, 1]  # Region 3
    else:
        return [0, 0]  # Region 4

# Function to update weights based on the prediction error
def update_weights(w, i, o, t, lr):
    # Add bias to input
    inputs_with_bias = np.insert(i, 0, 1)
    error = t - o  # Calculate error
    w += lr * np.outer(error, inputs_with_bias)  # Update w using the outer product
    return w

# Function to train the neural network
def train_network(w, training_data, iter, lr):
    predict_correct = 0  # Track total correct predictions across all iterations
    
    for _ in range(iter):
        correct_predictions = 0  # Reset correct predictions for this iteration
        for x, y in training_data:
            inputs = np.array([x, y])  # Prepare input array
            target = np.array(classify_region(x, y))  # Get target output for the region
            outputs = calculate_output(w, inputs)  # Calculate output using the network
            
            # Check if the prediction is correct
            if np.array_equal(outputs, target):
                correct_predictions += 1  # Increment correct prediction count
            
            # Update w based on the current input, output, and target
            w = update_weights(w, inputs, outputs, target, lr)
        
        predict_correct += correct_predictions  # Update total correct predictions

    # Calculate overall percentage of correct predictions after training
    overall_percentage = (predict_correct / (iter * len(training_data))) * 100
    print(f"Overall training correct predictions:" + 
          f" {predict_correct}/{iter * len(training_data)} ({overall_percentage:.2f}%)")
    
    return w

# Function to test the neural network on test data
def test_network(w, testing_data):
    predict_correct = 0  # Track correct predictions in testing
    
    for x, y in testing_data:
        i = np.array([x, y])  # Prepare input array
        t = np.array(classify_region(x, y))  # Get target output for the region
        o = calculate_output(w, i)  # Calculate output using the network
        
        # Check if the prediction is correct
        if np.array_equal(o, t):
            predict_correct += 1  # Increment correct prediction count
    
    # Calculate overall percentage of correct predictions in testing
    overall_percentage = (predict_correct / len(testing_data)) * 100
    print(f"Overall testing correct predictions:" + 
          f"{predict_correct}/{len(testing_data)} ({overall_percentage:.2f}%)")

# Function to perform a single-step test based on user input
def single_step_test(w):
    x = float(input("Enter value for x: "))  # Get x input from user
    y = float(input("Enter value for y: "))  # Get y input from user
    i = np.array([x, y])  # Prepare input array
    o = calculate_output(w, i)  # Calculate output using the network
    
    # Manually classify based on the output neurons
    if np.array_equal(o, [1, 0]):
        region = 1
    elif np.array_equal(o, [1, 1]):
        region = 2
    elif np.array_equal(o, [0, 1]):
        region = 3
    else:
        region = 4
    
    print(f"({x}, {y}) is in Region {region}")  # Output the classified region

# Main function to drive the program
def main():
    run = True
    while(run):
        # Get user input for training parameters
        iterations = int(input("Enter number of training iterations" +
                               "(Enter negative number to stop): "))
        if iterations >= 0:
            # allows user to customize number of training samples
            training_samples = int(input("Enter number of training samples: ")) 
            learning_rate = float(input("Enter learning rate (lambda): "))
            # allows user to customize number of testing samples
            testing_samples = int(input("Enter number of testing samples: ")) 
            
            # Generate random training and testing data
            training_data = np.random.uniform(-500, 500, (training_samples, 2))
            testing_data = np.random.uniform(-500, 500, (testing_samples, 2))
            
            # Initialize the weights
            weights = init_wts(input=2, output=2)
            
            # Train the network
            weights = train_network(weights, training_data, iterations, learning_rate)
            
            # Test the network
            test_network(weights, testing_data)
            
            # Single step test for user-defined input
            single_step_test(weights)
        else:
            run = False
    print("Goodbye.")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()