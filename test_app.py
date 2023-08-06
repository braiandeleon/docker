import requests
import json

#Primera prueba, de existencia
url = 'http://127.0.0.1:5000/turnover?employeenumber=601'  # Replace this with the URL of the API you want to query
response = requests.get(url)
if response.status_code == 200:
    # Request was successful, process the response data
    data = response.json()  # Assuming the response is in JSON format
    if data["message"] == 'Employee Number 601 has a probability of 7.07% of staying forever with the company':
        print("Test 1: Passed")
    else:
        print("Test 1: Failed")

#Segunda prueba, de no existencia
url = 'http://127.0.0.1:5000/turnover?employeenumber=6010'  # Replace this with the URL of the API you want to query
response = requests.get(url)
if response.status_code == 200:
    # Request was successful, process the response data
    data = response.json()  # Assuming the response is in JSON format
    if data["message"] == 'No record found for Employee Number 6010.':
        print("Test 2: Passed")
    else:
        print("Test 2: Failed")

#Tercera Prueba, de metodo POST
url = 'http://127.0.0.1:5000/turnover'  # Replace this with the URL of the API you want to query


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
        
response = requests.post(url, json=example)

if response.status_code == 200:
    response = response.json()
    # Access the "message" key
    if response.get('message') == 'New Data Received: Employee Number 6569999 has a probability of 73.55% of staying forever with the company':
        print('Test 3: Passed')
else:
    print(f"Test 3: Failed {response.status_code}")