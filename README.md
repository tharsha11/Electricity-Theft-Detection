# Electricity-Theft-Detection
It is used to find weather electricity is stolen or not. We used python and ml as backend and streamlit for web page creation. We collected data from nearby e&d office that consist of 155 residencies our aim is to find accurate ml model that shows weather electricity is stolen or not based on the previous statistical and tamperings. We trained data with different ml models like svm, random forest, decision tree, logistic regression, knn and found that svm provides more accuracy among them. So we deployed the model using svm and tested with real time data. 
Hyper paramater tuning- since our dataset contains less data points it overfits easily to overcome that we had done this. Normally in deep learning if we increase no of hidden layers the accuracy will the increased but some point the accuracy decrease to overcome this hyper paramater tuning is done. It contains a list of numbers from 1-1000 and checks for each no of layer and provide at which no of layer accuracy is more.
# How to run the project
put all the python files and pkl file in same package and then follow the below steps
run the csp.py code using streamlit run csp.py command.
csp.py code contains the streamlit code that takes you to a web page after runnig where we can find whether the electricity is faithfull or unfaithfull and also it shows machine learning algorithms with their accuracies.
main.py file contains the main code of the project. That main.py code is downloaded to a svm model using pickle and that model is used in csp.py
I provided out report and research paper in this repository.
