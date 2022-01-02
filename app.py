from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/testTypePage", methods=['GET', 'POST'])
def testType():
    test_no = request.form['test_type']
    if(int(test_no) == 0):
        return render_template("t_types.html")
    else:
        return getData()


@app.route("/getData", methods=['GET', 'POST'])
def getData():
    t_test_no = request.form['tType_select']
    return render_template("data.html")


if __name__ == ("__main__"):
    app.run(debug=True)
