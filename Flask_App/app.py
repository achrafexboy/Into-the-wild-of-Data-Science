# from crypt import methods
from json import load
from pickle import TRUE
import sqlite3
from flask import render_template, Flask, redirect, url_for, request, make_response, flash, abort
from ai import results, MissingValues, method_chosing, DataReduction, method_chosing_data_red, FeatureScaling, method_chosing_feature_selc, np
from ai import df1 as dataFrame
import pandas as pd
import seaborn as sns

# titanic_data = sns.load_dataset('titanic')
# titanic_data.head()
# DF = titanic_data.copy()

# print(DF.head())


app = Flask(__name__)

app.config['SECRET_KEY'] = '998273675bfbebc4d8be595e'

new_df = pd.DataFrame()  # create a new dataframe
print("newwww",new_df)


df = dataFrame


@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login.html')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=["POST", "GET"])
def succ():
    email = request.form.get("email")
    password = request.form.get("password")
    if email == "admin@admin.com" and password == "admin":
        return redirect(url_for('home'))

@app.route('/home.html')
@app.route('/home')
def home():
    return render_template('home.html', data=[dataFrame.head(12).to_html(classes='data', header="true")])


@app.route('/missingValues', methods=["POST", "GET"])
def missingValues():
    if request.method == 'POST':
        print("POST")
        if(request.form.get("rate")):
            print('rate')
            rate = float(request.form.get("rate"))
            if(request.form.get('rateOpt')):
                rateOpt = float(request.form.get("rateOptional"))
            else:
                rateOpt = 0.1
            method = request.form.get("method")  # row or column delete
            # mean median eod or arbitrary for numerical data
            numericalMethod = request.form.get("numericalMethod")
            # mode or arbitrary for categorical data
            categoricalMethod = request.form.get("categoricalMethod")
            print("Nummmmmm", numericalMethod)
            print(categoricalMethod)
            print("method: ", method)
            # print("rate: ", rate)

            # Infos!!!!!!!!!!!!!!!!!
            "if the user doesn't choose a method for numerical data, the method will be arbitrary"
            # End of Infos
            if(numericalMethod == None or categoricalMethod == None):
                # flash("Chose a method for numirical and categorical data")
                return render_template('missingValues.html', results=dataFrame)
            else:
                methods = []
                # Infos!!!!!!!!!!!!!!!!!
                "if the user chose to delete rows or columns we will ignore the other methods"
                # End of Infos
                if(method != None):
                    methods = [method]
                    classTest = MissingValues(df, methods, rate, rateOpt)
                    
                    new_df = method_chosing(classTest)
                    print("The new",new_df)
                    print('Numerical: ', classTest.missing_numeric_columns)
                    print('Categorical: ', classTest.missing_categorical_columns)
                else:
                    methods = [numericalMethod, categoricalMethod]
                    print("methods array: ", methods)
                    classTest = MissingValues(df, methods, rate)
                    new_df = method_chosing(classTest)
                    print("The new",new_df.isnull().mean().to_dict())
                    print('Numerical: ', classTest.missing_numeric_columns)
                    print('Categorical: ', classTest.missing_categorical_columns)
            return render_template('missingValues.html', results=df, finalResult=new_df.isnull().mean().to_dict(), new_df=new_df)

        else:
            print("No rate")
            flash("Enter a rate please")
            # print("error")
    return render_template('missingValues.html', results=dataFrame)


@app.route('/featureScaling', methods=["POST", "GET"]   )
def featureScaling():
    if request.method == 'POST':
        if(request.form.get("columnName")) and request.form.get("method"):
            method = request.form.get("method")
            columnName = request.form.get("columnName")
            Fsc = FeatureScaling(df)
            new_df = method_chosing_feature_selc(Fsc, method = method, param = columnName)
            return render_template('featureScaling.html', dataBefore=[dataFrame.head(12).to_html(classes='data', header="true")], columns = dataFrame.select_dtypes(include=np.number).columns, dataAfter=[new_df.head(12).to_html(classes='data', header="true")])
    return render_template('featureScaling.html', dataBefore=[dataFrame.head(12).to_html(classes='data', header="true")], columns=dataFrame.select_dtypes(include=np.number).columns)




if __name__ == "__main__":
    app.run(debug=True)


#==========================#
#Notes#
"Adding a column for varibles type in the page"
#==========================#
#==========================#
