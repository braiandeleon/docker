import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
# Ensemble algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, ConfusionMatrixDisplay, recall_score, roc_auc_score
# Create a pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To measure performance
from sklearn import metrics

# Save the model
import joblib

np.random.seed(2023)

hrdata = pd.read_csv('HR_Employee_Attrition.csv')
hrdata.head()
hrdata.shape
train_hrdata = hrdata.drop(columns=['EmployeeCount','EmployeeNumber','JobLevel', 
                                    'Over18', 'StandardHours', 'TotalWorkingYears'])
train_hrdata['Attrition'] = train_hrdata.Attrition.map({'Yes':1,
                                              'No':0})

categorical_attributes = ['BusinessTravel', 'OverTime',
                          'Department', 'EducationField', 
                          'Gender','JobRole','MaritalStatus']

rf = RandomForestClassifier(max_depth=10, 
                            max_features=12, # 
                            n_estimators=180, # 
                            random_state=2023, 
                            n_jobs=-1)

cat_pipe = ColumnTransformer([('ordinal_encoder', OrdinalEncoder(), categorical_attributes)],
                             remainder='passthrough')

pipe_model = Pipeline([
      ('encoder', cat_pipe),
      ('classification', rf )
    ])


df1=  train_hrdata[train_hrdata.Attrition==0].sample(600).reset_index(drop=True)
df2=  train_hrdata[train_hrdata.Attrition == 1]
train_set = pd.concat([df1 , df2 , df2] , axis=0).reset_index(drop=True)

x = train_set.drop(columns=['Attrition']) ### Drop before having the target variable
y = train_set['Attrition']

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    random_state=2023, 
                                                    test_size=0.2,
                                                    stratify =y)

pipe_model.fit(x_train, y_train)
y_pred = pipe_model.predict(x_test)

print('Accuracy Score of Random Forest Classifier is: ', metrics.accuracy_score(y_test, y_pred))
print('Recall Score of Random Forest Classifier Model is: ', metrics.recall_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

val_cols = list(train_set.columns)

val_cols.remove('Attrition')

fig, ax = plt.subplots(1, 2, figsize=(20,5))
ax[0].set_title('Confusion Matrix:')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False, cmap='Blues', ax=ax[0])
ax[0].grid(False)

scoring = pipe_model.predict_proba(x_test[val_cols])[:,1]
# Compute ROC metrics:
fpr, tpr, thresholds = roc_curve(y_test.values, scoring)
roc_auc = auc(fpr, tpr)
                        
ax[1].set_title('ROC Curve - Classifier')
ax[1].plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc, c='teal')
ax[1].plot([0,1],[0,1],'--', c='skyblue')
ax[1].legend(loc='lower right')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_xlabel('False Positive Rate')

hrdata["turnover_score"] = pipe_model.predict_proba(hrdata[val_cols])[:,1] # 

hrdata[['EmployeeNumber','turnover_score']].head()

# Write locally the results of using the model
# hrdata[['EmployeeNumber','turnover_score']].to_csv('turnover_score_by_employee_number.csv', index=False)

selected_columns = ['EmployeeNumber', 'turnover_score']
hrdata[selected_columns].to_csv('turnover_score_by_employee_number.csv', index=False)

joblib.dump(pipe_model, 'clf.zahoree')

clf = joblib.load('clf.zahoree')

hrdata2 = pd.read_csv('HR_Employee_Attrition.csv')

collaborator_rn = np.random.choice(range(1,hrdata2.shape[1]))

collaborator = pd.DataFrame(hrdata2.iloc[collaborator_rn,:]).T

collaborator.drop(columns=['EmployeeCount', 
                           'Attrition',
                           'JobLevel', 
                           'Over18',
                           'StandardHours', 
                           'TotalWorkingYears'], inplace=True)

collaborator.to_json(orient="records")

example = {"Age":22,  
           "BusinessTravel":"Travel_Frequently",  
           "DailyRate":29, 
           "Department":"Research & Development", 
           "DistanceFromHome":15, 
           "Education":3,  
           "EducationField":"Life Sciences",  
           "EmployeeNumber":6569999,  
           "EnvironmentSatisfaction":13,  
           "Gender":"Male",  
           "HourlyRate":61,  
           "JobInvolvement":2,  
           "JobRole":"Research Scientist",  
           "JobSatisfaction":1,  
           "MaritalStatus":"Married",  
           "MonthlyIncome":51,  
           "MonthlyRate":24907, 
           "NumCompaniesWorked":1, 
           "OverTime":"Yes", 
           "PercentSalaryHike":23, 
           "PerformanceRating":4, 
           "RelationshipSatisfaction":4,  
           "StockOptionLevel":1,  
           "TrainingTimesLastYear":3,  
           "WorkLifeBalance":3,  
           "YearsAtCompany":5,  
           "YearsInCurrentRole":2,  
           "YearsSinceLastPromotion":5,  
           "YearsWithCurrManager":2}

new_example = json.dumps(example)

# Use an existing example in the dataset:
#data = json.loads(request)

# Use the new example:
data = json.loads(new_example)


def hr_predict(request):
    df = pd.DataFrame([request])
    ID = df['EmployeeNumber'][0]
    df.drop(columns=['EmployeeNumber'], inplace=True)
    prediction = clf.predict_proba(df) #
    output = {'ID': ID , 'prediction': list(prediction[:,1])[0]}
    return output

hr_predict(data)

from flask import Flask, request, jsonify
import pandas as pd
import json

app = Flask(__name__)

def get_turnover_score(name):
    name = int(request.args.get('employeenumber', ''))  # Get the 'id' parameter from the request query string
    df = pd.read_csv('turnover_score_by_employee_number.csv')

    desired_row = df[df['EmployeeNumber'] == name]

    if not desired_row.empty:
        nth_turnover_score = desired_row.at[desired_row.index[0], 'turnover_score']
        return jsonify(message=f'Employee Number {name} has a probability of {round(100*float(nth_turnover_score),2)}% of staying forever with the company')
    else:
        return jsonify(message=f'No record found for Employee Number {name}.')

def post_turnover_score():
    try:
        data = json.loads(request.data)  # Load the JSON string from the request body
        df = pd.DataFrame([data])
        ID = df['EmployeeNumber'][0]
        df.drop(columns=['EmployeeNumber'], inplace=True)
        prediction = clf.predict_proba(df) #
        output = round(100*float(prediction[0, 1]),2)

        if output:
            return jsonify(message=f'New Data Received: Employee Number {ID} has a probability of {output}% of staying forever with the company')
        else:
            return jsonify(message=f'No record found for EmployeeNumber')
        return output
    
    except json.JSONDecodeError:
        return jsonify(message='Invalid JSON data in the request body.'), 400
    except KeyError:
        return jsonify(message='Invalid JSON key in the request body. Make sure it contains an "id" field.'), 400

@app.route('/turnover', methods=['GET', 'POST'])
def hr_predict():
    if request.method == 'GET':
        name = int(request.args.get('employeenumber', ''))  # Get the 'id' parameter from the request query string
        return get_turnover_score(name)
    elif request.method == 'POST':
        return post_turnover_score()
    else:
        return jsonify(message='Method not allowed.'), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)