# from crypt import methods
from json import load
import sqlite3
from flask import render_template, Flask, redirect, url_for, request, make_response, abort

app = Flask(__name__)

@app.route('/login')
def login():
    rep = make_response(render_template('login.html'))
    rep.set_cookie('answer','42')
    return rep

@app.route('/af_login', methods=["POST"])
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

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register')
def register():
    rep = make_response(render_template('Registration.html'))
    rep.set_cookie('answer','42')
    return rep

@app.route('/af_register', methods=["POST"])
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

if __name__ == "__main__" :
    app.run(debug=True)