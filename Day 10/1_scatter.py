import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

fpath = 'Day 10/insurance.csv'
data = pd.read_csv(fpath)

# sns.scatterplot(x=data.bmi, y=data.charges, hue=data.smoker)
# sns.regplot(x=data.bmi, y=data.charges)  # Adding a regression line


'''
The sns.lmplot:
- Instead of setting x=insurance_data['bmi'] to select the 'bmi' column in insurance_data, 
    we set x="bmi" to specify the name of the column only.
- Similarly, y="charges" and hue="smoker" also contain the names of columns.
- We specify the dataset with data=insurance_data.

'''
# sns.lmplot(x="bmi", y="charges", hue="smoker", data=data)

sns.swarmplot(x=data.smoker, y=data.charges)


plt.show()