from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
import pandas as pd
import numpy as np
import os
import io
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "bebas_aja_buat_flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

DATA_PATH = os.path.join(DATASET_FOLDER, "dataset.csv")

OUTPUT_SUMMARY = os.path.join(BASE_DIR, "simulation_summary.csv")

# ---- Model parameters (bisa di-tune di UI nanti jika mau) ----
params = {
    'alpha_sleep_duration': 0.5,
    'beta_physical': 1.2,
    'gamma_stress': 0.7,
    'delta_disorder': 2.0,
    'stress_from_sleep': 0.3,
    'stress_from_activity': -0.4,
    'stress_recovery': -0.05
}

# ---- helper mapping for sleep disorders ----
disorder_map = {'None': 0.0, 'Insomnia': 0.2, 'Sleep Apnea': 0.3}

def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # basic cleaning / ensure columns exist
    expected_cols = ['Person ID','Gender','Age','Occupation','Sleep Duration','Quality of Sleep',
                     'Physical Activity Level','Stress Level','BMI Category','Blood Pressure',
                     'Heart Rate','Daily Steps','Sleep Disorder']
    # If dataset has slightly different headers, try to match by lower-case
    if not set(expected_cols).issubset(set(df.columns)):
        # try to normalize column names
        df.columns = [c.strip() for c in df.columns]
    # Add computed columns used in model
    df['SleepDisorderFactor'] = df['Sleep Disorder'].map(disorder_map).fillna(0.0) if 'Sleep Disorder' in df.columns else 0.0
    pa_col = 'Physical Activity Level'
    if pa_col not in df.columns:
        df[pa_col] = df.get('Daily Steps', 0) / 100  # fallback heuristic
    pa_min, pa_max = df[pa_col].min(), df[pa_col].max()
    df['PA_norm'] = (df[pa_col] - pa_min) / (pa_max - pa_min + 1e-9)
    df = df.reset_index(drop=True)
    return df

def simulate_row(row, days=30, scenario='baseline', extra_steps=0, stress_reduction=0.0):
    # === Ambil nilai awal ===
    sq = float(row.get('Quality of Sleep', 5.0))
    stress = float(row.get('Stress Level', 5.0))
    sleep_hours = float(row.get('Sleep Duration', 6.5))
    pa_norm = float(row.get('PA_norm', 0.5))
    disorder = float(row.get('SleepDisorderFactor', 0.0))
    daily_steps = float(row.get('Daily Steps', 0.0))

    # --- New Variables ---
    bmi_cat = row.get('BMI Category', 'Normal')
    bp = row.get('Blood Pressure', '120/80')
    hr = float(row.get('Heart Rate', 75))

    # BMI category → numeric
    bmi_map = {
        'Underweight': 1, 'Normal': 2, 'Normal Weight': 2,
        'Overweight': 3, 'Obese': 4
    }
    bmi_index = bmi_map.get(bmi_cat, 2)

    # Extract BP numbers
    try:
        sys_bp, dia_bp = map(int, bp.split('/'))
    except:
        sys_bp, dia_bp = (120, 80)

    # === SKENARIO ===
    if scenario == 'more_steps':
        daily_steps += extra_steps
        pa_norm = min(1.0, pa_norm + (extra_steps / 10000.0))

    if scenario == 'stress_intervention':
        stress = max(0.0, stress - stress_reduction)

    # === History ===
    history = {
        'day': [], 
        'SleepQuality': [], 
        'Stress': [],
        'SleepHours': [],
        'PA_norm': [],
        'SleepDisorderFactor': [],
        'BMI_index': [],
        'Systolic': [],
        'Diastolic': [],
        'HeartRate': []
    }

    # === SIMULASI HARIAN ===
    for t in range(days):
        # --- Pengaruh BMI terhadap Sleep Disorder ---
        disorder_effect = disorder + 0.05 * (bmi_index - 2)

        # --- Perubahan Sleep Quality ---
        delta_sleep_hours = params['alpha_sleep_duration'] * (sleep_hours - 7.0)
        delta_pa = params['beta_physical'] * (pa_norm - 0.5)
        delta_stress_neg = - params['gamma_stress'] * (stress - 5.0) / 5.0
        delta_disorder = - params['delta_disorder'] * disorder_effect

        sq = sq + delta_sleep_hours + delta_pa + delta_stress_neg + delta_disorder

        # BP effect on sleep quality
        if sys_bp >= 130 or dia_bp >= 85:
            sq -= 0.05

        sq = max(1.0, min(10.0, sq))

        # --- Stress ---
        stress_delta = params['stress_from_sleep'] * max(0.0, (7.0 - sleep_hours)) \
                       + params['stress_from_activity'] * pa_norm

        # Heart rate affects recovery
        HR_factor = (80 - hr) / 100
        stress_recovery = -0.05 + HR_factor

        stress = stress + stress_delta + stress_recovery

        # BP effect on stress
        if sys_bp >= 130 or dia_bp >= 85:
            stress += 0.1

        stress = max(0.0, min(10.0, stress))

        # === Simpan ke History ===
        history['day'].append(t)
        history['SleepQuality'].append(round(sq,3))
        history['Stress'].append(round(stress,3))
        history['SleepHours'].append(sleep_hours)
        history['PA_norm'].append(round(pa_norm,3))
        history['SleepDisorderFactor'].append(round(disorder_effect,3))
        history['BMI_index'].append(bmi_index)
        history['Systolic'].append(sys_bp)
        history['Diastolic'].append(dia_bp)
        history['HeartRate'].append(hr)

    return history


# ---- Routes ----
@app.route('/')
def index():
    df = load_dataset()
    n = len(df) if df is not None else 0
    sample_cols = df.columns.tolist() if df is not None else []
    return render_template('index.html', dataset_exists=(df is not None), nrows=n, cols=sample_cols,
                           docx_path="/mnt/data/UTS Teori dan Praktikum Pemodelan dan Simulasi.docx")

@app.route('/upload', methods=['POST'])
def upload():
    # If user didn't upload anything
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == "":
        return "No selected file", 400

    # Ensure folder exists
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    # Save CSV as dataset.csv
    file.save(DATA_PATH)

    return redirect(url_for('index'))

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
        flash("Dataset berhasil dihapus!", "success")
        return redirect(url_for('index'))

    flash("Dataset tidak ditemukan.", "error")
    return redirect(url_for('index'))



@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """
    Expected form fields:
    - scenario: baseline / more_steps / stress_intervention
    - days: integer
    - extra_steps: integer (if scenario more_steps)
    - stress_reduction: float (if stress_intervention)
    - person_id: optional integer; if missing, run for all
    """
    df = load_dataset()
    if df is None:
        return "Dataset not found. Upload a CSV first.", 400

    scenario = request.form.get('scenario','baseline')
    days = int(request.form.get('days', 30))
    extra_steps = int(request.form.get('extra_steps', 0))
    stress_reduction = float(request.form.get('stress_reduction', 0.0))
    person_id = request.form.get('person_id', None)

    results = {}
    summary_rows = []
    if person_id:
        # find by Person ID or index
        try:
            pid = int(person_id)
            row = df[df['Person ID']==pid].iloc[0]
        except Exception:
            # try treating as index
            row = df.iloc[int(person_id)]
        hist = simulate_row(row, days=days, scenario=scenario, extra_steps=extra_steps, stress_reduction=stress_reduction)
        results[str(row.get('Person ID','0'))] = hist
        summary_rows.append({
            'Person ID': row.get('Person ID','0'),
            'Initial_SleepQuality': row.get('Quality of Sleep', np.nan),
            'Final_SleepQuality': hist['SleepQuality'][-1],
            'Initial_Stress': row.get('Stress Level', np.nan),
            'Final_Stress': hist['Stress'][-1],
            'Scenario': scenario
        })
    else:
        # run for all rows (sensible for up to 400 rows)
        for idx, row in df.iterrows():
            hist = simulate_row(row, days=days, scenario=scenario, extra_steps=extra_steps, stress_reduction=stress_reduction)
            pid = row.get('Person ID', idx)
            results[str(pid)] = hist
            summary_rows.append({
                'Person ID': pid,
                'Initial_SleepQuality': row.get('Quality of Sleep', np.nan),
                'Final_SleepQuality': hist['SleepQuality'][-1],
                'Initial_Stress': row.get('Stress Level', np.nan),
                'Final_Stress': hist['Stress'][-1],
                'Scenario': scenario
            })

    # save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    # Return JSON to render client-side with Plotly
    return jsonify({'results': results, 'summary': summary_df.to_dict(orient='records')})

@app.route('/download_summary')
def download_summary():
    if not os.path.exists(OUTPUT_SUMMARY):
        return "No summary available yet. Run simulation first.", 404
    return send_file(OUTPUT_SUMMARY, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
