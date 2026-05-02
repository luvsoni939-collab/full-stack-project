from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df_full = pd.read_csv("StudentMarksDataset.csv")
df = df_full[['Std_StudyHours', 'Std_Marks']].dropna()

# Train model
X = df[['Std_StudyHours']]
y = df['Std_Marks']

model = LinearRegression()
model.fit(X, y)

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form['hours'])
        if hours < 0:
            return "Study hours cannot be negative"
    except:
        return "Invalid input"

    prediction = model.predict([[hours]])[0]

    # ✅ Model equation added
    equation = f"Marks = {model.intercept_:.2f} + {model.coef_[0]:.2f} × StudyHours"

    return render_template('result.html',
                           hours=hours,
                           marks=round(prediction, 2),
                           equation=equation)


@app.route('/student', methods=['POST'])
def student():
    try:
        roll = int(request.form['roll'])
    except:
        return "Invalid roll number"

    student = df_full[df_full["roll_no"] == roll]

    if not student.empty:
        marks = student["Std_Marks"].values[0]
        avg = df["Std_Marks"].mean()

        performance = "Good" if marks > 75 else "Needs Improvement"

        return render_template('student.html',
                               roll=roll,
                               marks=marks,
                               avg=round(avg, 2),
                               performance=performance)
    else:
        return "Student not found"


@app.route('/graph')
def graph():
    # Scatter plot
    plt.figure()
    plt.scatter(df['Std_StudyHours'], df['Std_Marks'])
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Marks")
    plt.savefig("static/graph.png")
    plt.close()

    # ✅ Histogram added
    plt.figure()
    plt.hist(df['Std_Marks'])
    plt.xlabel("Marks")
    plt.ylabel("Frequency")
    plt.title("Distribution of Marks")
    plt.savefig("static/hist.png")
    plt.close()

    return render_template('graph.html')


# ------------------ RUN APP ------------------

if __name__ == '__main__':
    app.run(debug=True)