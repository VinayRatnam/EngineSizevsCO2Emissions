# Engine Size vs CO2 Emissions
This project demonstrates how a linear regression model can be used to predict a car's CO2 emissions based on its engine size. The goal is to explore the relationship between engine size (in liters) and CO2 emissions (in grams per kilometer), providing insight into how engine size influences a vehicle’s environmental impact. This project was created as part of a hands-on exercise in applying machine learning techniques to a real-world dataset with over 4000 data points.

# Dataset
The dataset used in this project comes from the Canadian Vehicle Emissions dataset, which contains various features like engine size, fuel type, fuel consumption, and CO2 emissions. However, for the purpose of this analysis, we focus on two key variables:

Engine Size (L): The size of the engine in liters.
CO2 Emissions (g/km): The amount of CO2 emitted by the vehicle in grams per kilometer.

# Objective
The objective of this project is to:

Use simple linear regression to model the relationship between engine size and CO2 emissions.
Find the optimal values for the slope (m) and intercept (b) of the regression line, where the relationship is defined by the equation:
y=mx+b
where y represents the predicted CO2 emissions, x represents the engine size, m is the slope (indicating how much CO2 emissions change with engine size), and b is the y-intercept (representing the CO2 emissions when engine size is zero).

# Approach
Data Preprocessing: The dataset was loaded into a pandas DataFrame, and we focused on the relevant features for this analysis. The data was then split into independent (engine size) and dependent (CO2 emissions) variables.

Linear Regression Model:

We initialized the parameters m and b randomly.
Gradient Descent was used to optimize these parameters by minimizing the cost function (mean squared error).
The model was trained over several iterations, with real-time updates on the cost function and how the line of best fit improved.
Visualization:

The scatter plot of engine sizes vs. CO2 emissions was generated, with the line of best fit superimposed to visualize how the model improved over iterations.
We also tracked how the slope (m) and intercept (b) changed during training.

# Findings
After training the linear regression model for 1000 iterations, the following results were obtained:

Optimal Slope (m): 36.87
Optimal Intercept (b): 134.03
Interpretation of Results:
Slope (m = 36.87): This indicates that for every 1-liter increase in engine size, the CO2 emissions increase by approximately 36 grams per kilometer. Larger engines typically consume more fuel, leading to higher emissions, and this value quantifies that relationship.

Intercept (b = 134.03): This suggests that for a vehicle with a hypothetical engine size of 0 liters, the predicted CO2 emissions would be approximately 134 grams per kilometer. While a zero engine size is unrealistic, this value acts as a reference point for the model.

Model Performance:
The model effectively captured the linear relationship between engine size and CO2 emissions, and the cost function steadily decreased throughout training, indicating that the model was improving.
However, this simple linear regression model does not account for other factors (like fuel type, vehicle weight, or aerodynamics) that may also influence emissions, so further improvements could involve adding additional features to the model.

# Conclusion
This project demonstrates how machine learning techniques, specifically linear regression, can be used to model and predict relationships between variables. The findings show that engine size has a significant positive correlation with CO2 emissions, and the model provides a simple yet powerful way to estimate emissions based on engine size alone.

# Future Work
In future iterations of this project, we could:

Explore multivariate linear regression by incorporating additional features (e.g., fuel type, vehicle weight, or fuel consumption) to improve the accuracy of the model.
Analyze the residuals and error distribution to better understand the limitations of the model and make improvements.
