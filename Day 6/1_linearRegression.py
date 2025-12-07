import matplotlib.pyplot as plt

'''
In ML, we write the equation for a linear regression model as follows:

y = b + w1*x1

where:
- y is the predicted label—the output.
- b is the bias of the model. Bias is the same concept as the y-intercept in the algebraic equation for a line. 
    In ML, bias is sometimes referred to as w0. Bias is a parameter of the model and is calculated during training.
- w1 is the weight of the feature. Weight is the same concept as the slope m in the algebraic equation for a line. 
    Weight is a parameter of the model and is calculated during training.
- x1 is a feature—the input.

'''

"""
Loss Functions Summary
----------------------

**L1 Loss**
    The sum of the absolute differences between the predicted values and the actual values.
    Formula:
        Σ | actual_value - predicted_value |

**Mean Absolute Error (MAE)**
    The average of L1 losses across all N examples.
    Formula:
        (1 / N) * Σ | actual_value - predicted_value |

**L2 Loss**
    The sum of the squared differences between the predicted values and the actual values.
    Formula:
        Σ (actual_value - predicted_value)²

**Mean Squared Error (MSE)**
    The average of L2 losses across all N examples.
    Formula:
        (1 / N) * Σ (actual_value - predicted_value)²

**Root Mean Squared Error (RMSE)**
    The square root of the mean squared error (MSE).
    Formula:
        √((1 / N) * Σ (actual_value - predicted_value)²)
"""

x = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]
y = [18, 15, 18, 16, 15, 14, 24]

plt.scatter(x, y)

plt.xlim([0,8])
plt.ylim([0, 35])

plt.show()