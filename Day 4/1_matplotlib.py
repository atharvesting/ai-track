import matplotlib.pyplot as plt
import pandas as pd

'''
When working with plotting,
- The figure (fig) is the overall canvas/window where the plots are drawn.
- The axes (ax) refers to the individual region for plotting inside the figure.
'''
fig, ax = plt.subplots()

a = pd.read_csv('Day 4/data.csv')  # read the csv into a DF object

plt.scatter(a['QuizAvg'], a['FinalGrade'])  # x-axis: Quiz Averages for each person , y-axis: Final Grades
ax.set_xlabel("Scores")  # Setting the lable for the x-axis
ax.set_ylabel("FinalGrade")  # Setting the label for the y-axis

ax.set_title("Score to Grade Graph") # Setting the title for the entire axis/plot
plt.savefig("Day 4/figure.png")  # Saving the result as a png file
plt.show()  # Displaying the result in a separate window