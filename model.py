import urllib.request
import json
import os
import ssl
import pandas as pd


def predict_data(data_list_dict):

    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    # this line is needed if you use self-signed certificate in your scoring service.
    allowSelfSignedHttps(True)

    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    data = {
        "Inputs": {
            "input1": data_list_dict
        },
        "GlobalParameters": {}
    }

    body = str.encode(json.dumps(data))

    url = 'http://20.187.185.229:80/api/v1/service/attrition-endpoint1/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = 'jCgjA6tYRednVCJhAcpcHMvn0vgr2so4'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type': 'application/json',
               'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))


def filter_data(df):
    template_columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                        'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
                        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
                        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                        'YearsWithCurrManager']
    df = df[template_columns]
    return df


def transorm_data(df):
    df['Attrition'] = True
    df['OverTime'] = df['OverTime'].replace({'Yes': True, 'No': False})
    df['Over18'] = df['Over18'].replace({'Y': True, 'N': False})
    return df.to_dict(orient='records')


def output_df_template(data):
    template_columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                        'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
                        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
                        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                        'YearsWithCurrManager', 'Scored Probabilities']

    output_df = pd.DataFrame(data, columns=template_columns)
    return output_df
