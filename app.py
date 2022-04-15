# from crypt import methods
from json import load
from pickle import TRUE
import sqlite3
from flask import render_template, Flask, redirect, url_for, request, make_response, flash, abort
from ai import results, MissingValues, method_chosing, df
import pandas as pd

app = Flask(__name__)

app.config['SECRET_KEY'] = '998273675bfbebc4d8be595e'

new_df = pd.DataFrame()  # create a new dataframe

@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    rep = make_response(render_template('login.html'))
    rep.set_cookie('answer','42')
    return rep

@app.route('/login', methods=["POST"])
def succ():
    email = request.form.get("email")
    password = request.form.get("password")
    with sqlite3.connect("elements.db") as db:
      cur = db.cursor()
      cur.execute("Select * from users where email= ? AND password = ?",(email, password))
      rows = cur.fetchall()
    if len(rows) == 0:
        return redirect(url_for('login'))
    return redirect(url_for('annexe',session = True))



@app.route('/register')
def register():
    rep = make_response(render_template('Registration.html'))
    rep.set_cookie('answer','42')
    return rep

@app.route('/register', methods=["POST","GET"])
def registration():
    email = request.form.get("email")
    password = request.form.get("password")
    with sqlite3.connect("elements.db") as db:
      cur = db.cursor()
      cur.execute("INSERT INTO users(email,password) VALUES(?,?)",(email,password))
      db.commit()
    return redirect(url_for('login'))



@app.route('/home', methods=["POST","GET"])
def annexe(session = True):
    if session :
        with sqlite3.connect("elements.db") as db:
            db.row_factory = sqlite3.Row
            cur = db.cursor()
            c = cur.execute("select * from Track")
            rows = [dict(row) for row in c.fetchall()]
            db.commit()
        return render_template('Annexe.html', Rows = rows)
    else : return redirect(url_for('login'))

@app.route('/missingValues', methods=["POST", "GET"])
def missingValues():
    if request.method == 'POST':
        if(request.form.get("rate")):
            rate = float(request.form.get("rate"))   
            method = request.form.get("method") #row or column delete
            numericalMethod = request.form.get("numericalMethod") # mean median eod or arbitrary for numerical data
            categoricalMethod = request.form.get("categoricalMethod") # mode or arbitrary for categorical data
            print(numericalMethod)
            print(categoricalMethod)
            print("method: ", method)
            #print("rate: ", rate)

            #Infos!!!!!!!!!!!!!!!!!
            "if the user doesn't choose a method for numerical data, the method will be arbitrary"
            #End of Infos
            if(numericalMethod == None or categoricalMethod == None):
                flash("Chose a method for numirical and categorical data")
            else:
                methods = []
                #Infos!!!!!!!!!!!!!!!!!
                "if the user chose to delete rows or columns we will ignore the other methods"
                #End of Infos
                if(method != None):
                    methods = [method]
                    classTest = MissingValues(df, methods, rate)
                    new_df = method_chosing(classTest)
                    print('Numerical: ', classTest.missing_numeric_columns)
                    print('Categorical: ', classTest.missing_categorical_columns)
                else:
                    methods = [numericalMethod, categoricalMethod]
                    print("methods array: ", methods)
                    classTest = MissingValues(df, methods, rate)
                    new_df = method_chosing(classTest)
                    print('Numerical: ', classTest.missing_numeric_columns)
                    print('Categorical: ', classTest.missing_categorical_columns)
            return render_template('missingValues.html', results = df, finalResult = new_df.isnull().mean().to_dict(), new_df = new_df)
            
            """if(new_df.empty):
                #flash("No missing values")
                return render_template('missingValues.html', results = df)
            else:
                return render_template('missingValues.html', results = df, finalResult = new_df.isnull().mean().to_dict(), new_df = new_df)"""
        else:
            flash("Enter a rate please")
            #print("error")   
    return render_template('missingValues.html', results = df)


if __name__ == "__main__" :
    app.run(debug=True)


#==========================#
#Notes#
"Adding a column for varibles type in the page"
#==========================#
#==========================#