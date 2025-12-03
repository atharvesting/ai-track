import pandas as pd

a = pd.read_csv('Day 3/data.csv')  # reads the CSV file and returns a Pandas DataFrame by default
print(a)

a.to_csv(index=False)  # simply returns CSV as a string, use index=False to avoid adding unnecessary unnamed indexing

a.head()  # returns the first 5 rows of the DF by default. No. of rows can be specified.
a.tail()  # returns the last 5.

# Note that both of these methods return a DF
a.info()  # detailed summary including no. of entries, column names and their datatypes, memory usage, etc.
a.describe(include='all')  # detailed statistical summary. "include='all'" is datatype-agnostic. only numeric columns described by default.

# Taking the stat summary DF generated and saving it into a new CSV file. The file doesn't need to exist beforehand.
b = a.describe(include='all')
b.to_csv('Day 3/insight.csv', index=False)

# Accessing Data
a[['Name', 'Gender']]  # column-wise access.


a.iloc[0:5, [0,2,5,6]]  # Access is purely based on integer indices in .iloc()
# the method specifies row range (0th to 4th row) and which columns to include (0th, 2nd, 5th column etc.)

a.loc[0:5, ['Name', 'GPA']]  # Access is more flexible and can reference label names as well.

'''
Boolean Filtering:

a[a['Gender'] == 'F']
a[a['GPA'] > 3.8]
a[ (a['GPA'] > 3.6) & (a['ProjectScore'] < 93) ]

OR can also be applied using the pipesign |

'''

a['QuizAvg'] = ( a['ExamScore1'] + a['ExamScore2'] + a['ExamScore3'] ) / 3  # updating values using existing columns

a.sort_values('GPA', ascending=False)  # does not change the original DF
a.sort_values('GPA', ascending=False).head().loc[:, ['Name', 'Major']]  # Names and respective Majors of Top 5 students in terms of GPA

