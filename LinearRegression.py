import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the CSV file into a pandas dataframe
df = pd.read_csv('car_data/CO2 Emissions_Canada.csv')

# Display first few rows of the data
print(df.head())

# Provides information about what each column contains
print(df.info())

# Provides descriptive statistics
print(df.describe())

# Creates two arrays of data for engine_size and emissions
# emissions[i] is the target for the corresponding feature engine_size[i]
engine_sizes = df['Engine Size(L)']
emissions = df['CO2 Emissions(g/km)']


# cost function using y_hat = mx + b
def cost_function(engine_sizes, emissions, m, b):
    C = 0.0
    n = len(engine_sizes)

    #add the squares of the difference of the prediction value minus the actual value
    for i in range(n):
        C += (m*engine_sizes[i] + b - emissions[i]) ** 2

    C /= 2*n

    return C


# gradient descent calculation function; returns change in m and b
def gradient_descent(engine_sizes, emissions, m, b):
    delta_m = 0.0
    delta_b = 0.0

    n = len(engine_sizes)

    #calculate delta_m and delta_b
    for i in range(n):
        delta_m += (m*engine_sizes[i] + b - emissions[i]) * engine_sizes[i]
        delta_b += m*engine_sizes[i] + b - emissions[i]
    delta_m /= -1*n
    delta_b /= -1*n

    return delta_m, delta_b


# create function to train model
def train_model(engine_sizes, emissions, iters, l_r):
    #set initial values for m and b
    m = np.random.uniform(-5, 5)
    b = np.random.uniform(-5, 5)

    # Arrays to store m, b, values for each iteration for data visualization
    m_values = []
    b_values = []

    #create plot
    plt.figure(figsize=(10,5))
    plt.ion() #turns on interactive mode

    #plot scatter plot of actual data
    plt.scatter(engine_sizes, emissions, color='blue', label='Data Points')
    plt.xlabel('Engine Size (L)')
    plt.ylabel('CO2 Emissions (g/km)')



    # each iteration should calculate the cost function at the current values of m and b
    # each iteration should reevaluate m and b based on the output of the gradient_descent function
    for i in range(iters):
        curr_cost = cost_function(engine_sizes, emissions, m, b)

        # store m and b for visualization
        m_values.append(m)
        b_values.append(b)

        if (i % 100 == 0):
            print(f'The cost of this model after {i} iterations is {curr_cost}')

            # plot the line for the current 'line of best fit'
            x_vals = np.array(engine_sizes)
            y_vals = m * x_vals + b
            plt.plot(x_vals, y_vals, color='red', label=f'Iteration {i / 100}')

            # Update the graph
            plt.draw()
            plt.pause(0.1)  # Pause to see the update
        
        delta_m, delta_b = gradient_descent(engine_sizes, emissions, m, b)
        m += l_r * delta_m
        b += l_r * delta_b
    
    plt.ioff()  # Turn off interactive mode
    plt.legend()
    plt.show()

    # Plot how m and b adjusted over iterations
    plt.figure(figsize=(8, 6))
    plt.scatter(m_values, b_values, color='green')
    plt.xlabel('m values')
    plt.ylabel('b values')
    plt.title('Adjustments of m and b over iterations')
    plt.grid(True)
    plt.show()

    return m,b

# establish number of iterations that the model will train and the learning rate
# feel free to tweak these numbers!
iterations = 400
learning_rate = 0.1

m, b = train_model(engine_sizes, emissions, iterations, learning_rate)
print(f"Final values: m = {m}, b = {b}")