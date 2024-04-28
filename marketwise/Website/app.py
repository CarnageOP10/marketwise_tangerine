from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///leads.db'
db = SQLAlchemy(app)

# Initialize NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Lead model
class Lead(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    job_title = db.Column(db.String(100))
    company = db.Column(db.String(100))
    interest = db.Column(db.String(100))

def load_data():
    df_exhibitors = pd.read_csv('PS1/exhibitors_tokenized.csv')
    df_visitors = pd.read_csv('PS1/visitors_tokenized.csv')
    df_attended_events = pd.read_csv('PS1/attended_events.csv')

    return df_exhibitors, df_visitors, df_attended_events

df_exhibitors, df_visitors, df_attended_events = load_data()

def get_cosine_similarity():
    # Convert tokenized data to string
    tfidf = TfidfVectorizer()
    tfidf_matrix_exhibitors = tfidf.fit_transform(df_exhibitors['combined'])
    tfidf_matrix_visitors = tfidf.transform(df_visitors['combined'])

    # Get cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix_exhibitors, tfidf_matrix_visitors)
    cosine_sim_exhibitors = cosine_similarity(tfidf_matrix_exhibitors)
    return cosine_sim, cosine_sim_exhibitors

temp_exhibitors = pd.read_csv('PS1/exhibitors.csv')
company_names = temp_exhibitors['company_name']
company_repNames = temp_exhibitors['company_repName']
company_repMobileNumbers = temp_exhibitors['mobile_no']

cosine_sim, cosine_sim_exhibitors = get_cosine_similarity()

visitor_attended = {}
for index, row in df_attended_events.iterrows():
    if row['visitor'] in visitor_attended:
        visitor_attended[row['visitor']].append((row['exhibitor'], row['rating']))
    else:
        visitor_attended[row['visitor']] = [(row['exhibitor'], row['rating'])]

def get_recommendations(visitor_id, top_n=5):
    recs = []
    recommendations = list(enumerate(cosine_sim[:, visitor_id]))
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    recommendations = recommendations[:top_n]
    for recommendation in recommendations:
        recs.append((
            company_names[recommendation[0]],
            company_repNames[recommendation[0]],
            company_repMobileNumbers[recommendation[0]],
            recommendation[0]))
    
    if visitor_id in df_attended_events['visitor'].values:
        temp_recs = []
        for exhibitor, rating in visitor_attended[visitor_id]:
            recommendations = list(enumerate(cosine_sim_exhibitors[int(exhibitor)]))
            recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
            recommendations = recommendations[:2]
            for recommendation in recommendations:
                temp_recs.append((
                    company_names[recommendation[0]],
                    company_repNames[recommendation[0]],
                    company_repMobileNumbers[recommendation[0]],
                    recommendation[0],
                    rating))
        temp_recs.sort(key=lambda x: x[4], reverse=True)
        temp_recs = [rec[:-1] for rec in temp_recs]
        # Ensure recs and temp_recs have equal lengths before combining
        min_len = min(len(recs), len(temp_recs))
        recs = [recs[i] for i in range(min_len)]
        temp_recs = [temp_recs[i] for i in range(min_len)]
        # Combine recs and temp_recs into a single list
        recs = [val for pair in zip(recs, temp_recs) for val in pair]

    return recs[:top_n]
        

def tokenize_data(data):
    return (data['profession'].lower() + ' ' + data['city'].lower() + ' ' + data['state'].lower())

# Define a function to save feedback data to CSV
def save_to_csv(data):
    df = [data['visitor'], data['exhibitor'], data['rating']]
    df_attended_events.loc[len(df_attended_events)] = df
    df_attended_events.to_csv('PS1/attended_events.csv', index=False)

# Check if CSV file exists to determine if header should be written
def df_exists():
    try:
        pd.read_csv('attended_events.csv')
        return True
    except FileNotFoundError:
        return False
    
model_ps3 = joblib.load('PS3/linear_regression_model.pkl')
scaler_ps3 = joblib.load('PS3/scaler.pkl')
df_ps3 = pd.read_csv('PS3/events_dataset_numerical.csv')

model_ps4 = joblib.load('PS4/model.pkl')
scaler_ps4 = joblib.load('PS4/scaler.pkl')
df_ps4 = pd.read_csv('PS4/cleaned_data.csv')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        job_title = request.form['job_title']
        company = request.form['company']
        interest = request.form['interest']

        job_title_keywords = extract_keywords(job_title)
        company_keywords = extract_keywords(company)
        interest_keywords = extract_keywords(interest)

        new_lead = Lead(name=name, email=email, job_title=job_title_keywords, company=company_keywords, interest=interest_keywords)

        db.session.add(new_lead)
        db.session.commit()

        return redirect(url_for('index'))
    
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/ps1', methods=['GET', 'POST'])
def ps1():
    global df_visitors, cosine_sim, cosine_sim_exhibitors
    if request.method == 'POST':
        input_data = {
            'profession': request.form['profession'],
            'city': request.form['city'],
            'state': request.form['state']
        }
        input_data = tokenize_data(input_data)
        df_visitors.loc[len(df_visitors)] = input_data
        print(df_visitors.tail(5))
        print(len(df_visitors))
        cosine_sim, cosine_sim_exhibitors = get_cosine_similarity()

        recommendations = get_recommendations(len(df_visitors) - 1)
        return render_template('feedback1.html', recommendations=recommendations, visitor_id=len(df_visitors) - 1)
    return render_template('ps1.html')

@app.route('/feedback1', methods=['GET', 'POST'])
def feedback1():
    if request.method == 'POST':
        feedback_data = {
            'visitor': request.form['visitor'],
            'exhibitor': request.form['exhibitor'],
            'rating': request.form['rating']
        }
        save_to_csv(feedback_data)
        return redirect(url_for('thank_you'))
    return render_template('feedback1.html')


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

def extract_keywords(text):
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)

    keywords = [word for word, tag in tagged_words if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    return ' '.join(keywords)

@app.route('/ps3', methods=['GET', 'POST'])
def ps3():
    if request.method == 'POST':
        input_data = {
            'Event_Type': int(request.form['event_type']),
            'Location': int(request.form['location']),
            'Ticket_Price': float(request.form['ticket_price']),
            'Weather_Condition': int(request.form['weather_condition']),
            'Day_of_Week': int(request.form['day_of_week']),
            'Month': int(request.form['month']),
        }

        df_ps3.loc[len(df_ps3)] = input_data

        input_data = scaler_ps3.transform(np.array(list(input_data.values())).reshape(1, -1))

        prediction = model_ps3.predict(input_data)
        return render_template('feedback3.html', prediction=round(prediction[0]))
        
    return render_template('ps3.html')

@app.route('/feedback3', methods=['GET', 'POST'])
def feedback3():
    if request.method == 'POST':
        feedback_data = {
            'actual_attendance_count': int(request.form['actual_attendance_count']),
        }

        df_ps3.loc[len(df_ps3) - 1, 'Attendance_Count'] = feedback_data['actual_attendance_count']
        df_ps3.to_csv('PS3/events_dataset_numerical.csv', index=False)

        new_X = df_ps3.drop(['Attendance_Count'], axis=1).tail(1)
        new_y = df_ps3['Attendance_Count'].tail(1)

        model_ps3.fit(new_X, new_y)

        return redirect(url_for('thank_you'))
    return render_template('feedback3.html')

@app.route('/ps4', methods=['GET', 'POST'])
def ps4():
    if request.method == 'POST':
        input_data = {
            'Industry': int(request.form['industry']),
            'Company Size': int(request.form['company_size']),
            'Package Type': int(request.form['package_type']),
            'Sponsorship Amount': int(request.form['sponsorship_amount']),
            'Duration (Months)': int(request.form['duration']),
            'Marketing Activities': int(request.form['marketing_activities']),
            'Event Type': int(request.form['event_type']),
            'Event Month': int(request.form['event_month']),
            'Attendee Demographics': int(request.form['attendee_demographics']),
            'Feedback': int(request.form['feedback'])
        }

        df_ps4.loc[len(df_ps4)] = input_data

        input_data = scaler_ps4.transform(np.array(list(input_data.values())).reshape(1, -1))

        prediction = model_ps4.predict(input_data)
        return render_template('feedback4.html', prediction=round(prediction[0], 2))
        
    return render_template('ps4.html')

@app.route('/feedback4', methods=['GET', 'POST'])
def feedback4():
    if request.method == 'POST':
        feedback_data = {
            'actual_roi_metrics': float(request.form['actual_roi_metrics']),
        }

        df_ps4.loc[len(df_ps4) - 1, 'ROI Metrics'] = feedback_data['actual_roi_metrics']
        df_ps4.to_csv('PS4/cleaned_data.csv', index=False)

        return redirect(url_for('thank_you'))
    return render_template('feedback4.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    df_visitors.to_csv('PS1/visitors_tokenized.csv', index=False)
    joblib.dump(model_ps3, 'PS3/linear_regression_model.pkl')
    joblib.dump(model_ps4, 'PS4/model.pkl')

