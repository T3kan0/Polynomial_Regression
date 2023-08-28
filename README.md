# Polynomial_Regression
Tekano Mbonani

## System Docs 📃
This is an example of the supervised Machine Learning algorithms/techniques often used to model relationships between features and labels. In this code, I show examples of linear regression, as well as Polynomial regression, and compare the different models.  

## Software Requirements 🔌
You will need to install the following software on your system in order to run/edit the Python script.
* Mac OS/ Ubuntu 18.04 OS
* Python 3.7
* Textedit/ IDE - spyder or jupyter-notebook
* Python libraries
  * Numpy
  * Matplotlib
  * Sklearn
  * PIL
  * glob
    
### About the Data 💾 
The data used here was generated randomly with the **numpy** python library. The data is meant to only help us see how the different Polynomial orders can fit the data. For us to see things to avoid when fitting the data, such as overfitting when the order of the polynomial function gets high. In this case, the data was best fit with a polynomial function of the first order, i.e., Linear. 
### Profile Model 🧮
The code employs supervised machine learning from the library ***sklearn*** for the analysis. The data is split into trainig and testing data, then the model is trained with the said training data, before it can be fit to the test data. Similar to many other models, the training data was 80% of the overall data, while the 20% was used to test the data.
