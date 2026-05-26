from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io, webbrowser, threading

app = Flask(__name__)

# ---------------- ML MODEL ----------------
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'cgpa': np.round(np.random.uniform(5,10,n),2),
    'skills': np.random.randint(1,11,n),
    'communication': np.random.randint(1,11,n),
    'internships': np.random.randint(0,5,n),
    'aptitude': np.random.randint(40,100,n)
})

data['placed'] = (
    (data.cgpa >= 7) |
    ((data.skills >= 7) &
     (data.communication >= 7) &
     (data.aptitude >= 60))
).astype(int)

X, y = data.drop('placed', axis=1), data['placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = round(
    accuracy_score(y_test, model.predict(X_test)) * 100, 2
)

# ---------------- GRAPH FUNCTIONS ----------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png',
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img

def create_cm():
    fig, ax = plt.subplots(figsize=(4,3))

    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    sns.heatmap(
        confusion_matrix(y_test, model.predict(X_test)),
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No','Yes'],
        yticklabels=['No','Yes'],
        ax=ax
    )

    ax.set_title('Confusion Matrix', color='white')
    ax.tick_params(colors='white')

    return fig_to_base64(fig)

def create_fi():
    fig, ax = plt.subplots(figsize=(4,3))

    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    features = ['CGPA','Skills','Comm','Intern','Aptitude']

    ax.barh(features,
            model.feature_importances_,
            color=['#38bdf8','#818cf8',
                   '#a78bfa','#f472b6','#34d399'])

    ax.set_title('Feature Importance', color='white')
    ax.tick_params(colors='white')

    return fig_to_base64(fig)

CM_IMG = create_cm()
FI_IMG = create_fi()

# ---------------- SUGGESTIONS ----------------
def add_tip(tips, icon, area, msg):
    tips.append({
        "icon": icon,
        "area": area,
        "msg": msg
    })

def get_suggestions(cgpa, skills, comm, intern, apt, placed):

    tips = []

    rules = [
        (cgpa < 7, "📘", "CGPA",
         f"Improve CGPA from {cgpa} to above 7.0."),

        (skills < (6 if placed else 7),
         "💻", "Technical Skills",
         f"Improve technical skills from {skills}/10."),

        (comm < (6 if placed else 7),
         "🎤", "Communication Skills",
         f"Improve communication skills from {comm}/10."),

        (intern == 0,
         "🏢", "Internships",
         "Complete at least one internship."),

        (apt < (60 if placed else 65),
         "🧠", "Aptitude",
         f"Improve aptitude score from {apt}/100.")
    ]

    for condition, icon, area, msg in rules:
        if condition:
            add_tip(tips, icon, area, msg)

    return tips

# ---------------- UI ----------------
HTML = """

<!DOCTYPE html>
<html>
<head>

<title>Placement Prediction</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

body{
    background:#0f172a;
    color:white;
    font-family:'Poppins',sans-serif;
    padding:20px;
}

.container{
    max-width:950px;
    margin:auto;
}

.title{
    text-align:center;
    margin-bottom:25px;
}

.title h1{
    font-size:36px;
    font-weight:700;
}

.title span{
    color:#38bdf8;
}

.title p{
    color:#94a3b8;
    font-size:14px;
    margin-top:6px;
}

.stats{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:15px;
    margin-bottom:20px;
}

.stat{
    background:#111827;
    padding:18px;
    border-radius:16px;
    text-align:center;
    border:1px solid #1e293b;
}

.stat h2{
    color:#38bdf8;
    font-size:28px;
}

.stat p{
    color:#94a3b8;
    font-size:13px;
}

.graphs{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:15px;
    margin-bottom:20px;
}

.card{
    background:#111827;
    border:1px solid #1e293b;
    border-radius:16px;
    padding:18px;
}

.graphs img{
    width:100%;
    border-radius:10px;
}

.form-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:15px;
}

input{
    width:100%;
    padding:12px;
    border:none;
    outline:none;
    border-radius:10px;
    background:#1e293b;
    color:white;
    font-size:14px;
}

input::placeholder{
    color:#94a3b8;
}

.full{
    grid-column:1/-1;
}

button{
    width:100%;
    padding:13px;
    border:none;
    border-radius:10px;
    background:linear-gradient(90deg,#0ea5e9,#6366f1);
    color:white;
    font-size:15px;
    font-weight:600;
    cursor:pointer;
    transition:.3s;
}

button:hover{
    transform:translateY(-2px);
    opacity:.92;
}

.result{
    display:none;
    margin-top:20px;
}

.success{
    border:1px solid rgba(34,197,94,.4);
    background:rgba(34,197,94,.08);
}

.fail{
    border:1px solid rgba(239,68,68,.4);
    background:rgba(239,68,68,.08);
}

.result h2{
    margin-bottom:8px;
}

.result p{
    color:#cbd5e1;
}

.tip{
    background:#1e293b;
    padding:12px;
    border-radius:10px;
    margin-top:12px;
}

.tip b{
    color:#facc15;
}

.footer{
    text-align:center;
    margin-top:25px;
    color:#64748b;
    font-size:13px;
}

@media(max-width:700px){

    .stats,
    .graphs,
    .form-grid{
        grid-template-columns:1fr;
    }

}

</style>

</head>

<body>

<div class="container">

<div class="title">
    <h1>Placement <span>Prediction</span> System</h1>
    <p>Machine Learning Based Student Placement Analysis</p>
</div>

<div class="stats">

    <div class="stat">
        <h2>{{accuracy}}%</h2>
        <p>Model Accuracy</p>
    </div>

    <div class="stat">
        <h2>300</h2>
        <p>Training Records</p>
    </div>

    <div class="stat">
        <h2>5</h2>
        <p>Input Features</p>
    </div>

</div>

<div class="graphs">

    <div class="card">
        <img src="data:image/png;base64,{{cm_img}}">
    </div>

    <div class="card">
        <img src="data:image/png;base64,{{fi_img}}">
    </div>

</div>

<div class="card">

    <div class="form-grid">

        <input class="full"
               id="name"
               placeholder="Student Name">

        <input id="cgpa"
               type="number"
               step="0.1"
               placeholder="CGPA">

        <input id="skills"
               type="number"
               placeholder="Technical Skills">

        <input id="comm"
               type="number"
               placeholder="Communication Skills">

        <input id="intern"
               type="number"
               placeholder="Internships">

        <input class="full"
               id="apt"
               type="number"
               placeholder="Aptitude Score">

        <div class="full">
            <button onclick="predict()">
                Predict Placement
            </button>
        </div>

    </div>

</div>

<div id="result" class="card result"></div>

<div class="footer">
    Random Forest Algorithm • Flask • Machine Learning
</div>

</div>

<script>

async function predict(){

    let data = {
        name:name.value || "Student",
        cgpa:cgpa.value,
        skills:skills.value,
        communication:comm.value,
        internships:intern.value,
        aptitude:apt.value
    }

    let res = await fetch('/predict',{

        method:'POST',

        headers:{
            'Content-Type':'application/json'
        },

        body:JSON.stringify(data)

    })

    let d = await res.json()

    let result = document.getElementById('result')

    result.style.display = 'block'

    result.className =
        'card result ' +
        (d.placed ? 'success':'fail')

    let html = `

        <h2>${d.name}</h2>

        <p>
        ${d.placed ?
        'PLACED ✅':
        'NOT PLACED ❌'}
        </p>

        <br>

        <p>
        Placement Probability:
        <b>${Math.round(d.probability*100)}%</b>
        </p>

    `

    if(d.suggestions.length){

        html += `<br><h3>Suggestions</h3>`

        d.suggestions.forEach(s=>{

            html += `

            <div class="tip">

                <b>
                ${s.icon} ${s.area}
                </b>

                <p>${s.msg}</p>

            </div>

            `

        })

    }

    else{

        html += `

        <br>

        <h3 style="color:#22c55e">
        Excellent Profile 🌟
        </h3>

        `

    }

    result.innerHTML = html

    result.scrollIntoView({
        behavior:'smooth'
    })

}

</script>

</body>
</html>

"""

# ---------------- ROUTES ----------------
@app.route('/')
def home():

    return render_template_string(
        HTML,
        accuracy=accuracy,
        cm_img=CM_IMG,
        fi_img=FI_IMG
    )

@app.route('/predict', methods=['POST'])
def predict():

    d = request.json

    arr = np.array([[
        float(d['cgpa']),
        int(d['skills']),
        int(d['communication']),
        int(d['internships']),
        int(d['aptitude'])
    ]])

    scaled = scaler.transform(arr)

    placed = bool(model.predict(scaled)[0])

    probability = float(
        model.predict_proba(scaled)[0][1]
    )

    suggestions = get_suggestions(
        float(d['cgpa']),
        int(d['skills']),
        int(d['communication']),
        int(d['internships']),
        int(d['aptitude']),
        placed
    )

    return jsonify({
        'name': d['name'],
        'placed': placed,
        'probability': round(probability,3),
        'suggestions': suggestions
    })

# ---------------- RUN APP ----------------
def open_browser():
    webbrowser.open(
        'http://127.0.0.1:5000'
    )

if __name__ == '__main__':

    print("\\nPlacement Prediction System Running...\\n")

    threading.Timer(
        1,
        open_browser
    ).start()

    app.run(debug=False)