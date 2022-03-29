# from crypt import methods
from json import load
import sqlite3
from flask import render_template, Flask, redirect, url_for, request, make_response, abort
from ai import resultDict, MissingValues, methode_chosing, df
app = Flask(__name__)

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
      cur.execute("Select * from users where email= ? AND password = ?",(email,password))
      rows = cur.fetchall()
      if len(rows) == 0:
          return redirect(url_for('login'))
    return redirect(url_for('annexe'))



@app.route('/register')
def register():
    rep = make_response(render_template('Registration.html'))
    rep.set_cookie('answer','42')
    return rep

@app.route('/register', methods=["POST"])
def registration():
    email = request.form.get("email")
    password = request.form.get("password")
    with sqlite3.connect("elements.db") as db:
      cur = db.cursor()
      cur.execute("INSERT INTO users(email,password) VALUES(?,?)",(email,password))
      db.commit()
    return redirect(url_for('login'))



@app.route('/home')
def annexe():
    with sqlite3.connect("elements.db") as db:
      db.row_factory = sqlite3.Row
      cur = db.cursor()
      c = cur.execute("select * from Track")
      rows = [dict(row) for row in c.fetchall()]
      db.commit()
      return render_template('Annexe.html', Rows = rows)

@app.route('/missingValues', methods=["POST", "GET"])
def missingValues():
    if request.method == 'POST':
        methode = request.form.get("methode")
        rate = float(request.form.get("rate"))
        print("methode: ", methode)
        print("rate: ", rate)
        classTest = MissingValues(df, methode, rate)
        new_df = methode_chosing(classTest)
        return render_template('missingValues.html', results = resultDict, finalResult = new_df.isnull().mean().to_dict())
    return render_template('missingValues.html', results = resultDict)


if __name__ == "__main__" :
    app.run(debug=True)