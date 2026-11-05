// Nutraceutical Disease Risk Prediction - Vercel Dashboard Engine

const PRESETS = {
  default: {
    age: 40, gender: "F", height: 170.0, weight: 70.0, activity: "Moderate",
    diet: "Good", vitd: 25.0, iron: 15.0, calcium: 900.0, b12: 2.5,
    omega: 1.5, zinc: 12.0, magnesium: 300.0, protein: 65.0,
    waist: 82.0, sys_bp: 120.0, dia_bp: 78.0
  },
  cardio: {
    age: 58, gender: "M", height: 175.0, weight: 94.0, activity: "Low",
    diet: "Poor", vitd: 14.0, iron: 12.0, calcium: 550.0, b12: 1.8,
    omega: 0.6, zinc: 8.5, magnesium: 210.0, protein: 55.0,
    waist: 104.0, sys_bp: 146.0, dia_bp: 92.0
  },
  anemia: {
    age: 26, gender: "F", height: 162.0, weight: 51.0, activity: "Moderate",
    diet: "Poor", vitd: 16.0, iron: 6.2, calcium: 620.0, b12: 1.2,
    omega: 0.8, zinc: 7.0, magnesium: 190.0, protein: 38.0,
    waist: 70.0, sys_bp: 110.0, dia_bp: 70.0
  },
  osteo: {
    age: 68, gender: "F", height: 158.0, weight: 52.0, activity: "Low",
    diet: "Average", vitd: 12.0, iron: 11.0, calcium: 420.0, b12: 1.9,
    omega: 0.9, zinc: 8.0, magnesium: 220.0, protein: 44.0,
    waist: 76.0, sys_bp: 132.0, dia_bp: 82.0
  },
  optimal: {
    age: 32, gender: "M", height: 180.0, weight: 76.0, activity: "High",
    diet: "Excellent", vitd: 38.0, iron: 20.0, calcium: 1150.0, b12: 4.5,
    omega: 2.8, zinc: 15.0, magnesium: 380.0, protein: 95.0,
    waist: 80.0, sys_bp: 116.0, dia_bp: 74.0
  }
};

const NUMERIC_COLS = [
  "Age", "Height", "Weight", "BMI_calc", "Waist", "Systolic_BP", "Diastolic_BP",
  "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein",
  "Calories", "Sugars", "Fiber", "Total_Fat", "Sat_Fat", "Sodium", "Potassium",
  "Waist_to_Height_Ratio", "Sodium_to_Potassium_Ratio", "Calcium_to_Magnesium_Ratio",
  "VitD_Calcium_Index", "Sugar_to_Fiber_Ratio", "Sat_Fat_Ratio", "Pulse_Pressure",
  "MAP", "BMI_Age_Interaction", "Iron_to_Protein_Ratio"
];

let fallbackWeights = null;

// Fetch fallback weights for instant client-side evaluation
fetch("linear_weights.json")
  .then(res => res.json())
  .then(data => { fallbackWeights = data; })
  .catch(err => console.log("Fallback weights note:", err.message));

// Input sync helper
function setupSync(sliderId, numberId) {
  const slider = document.getElementById(sliderId);
  const number = document.getElementById(numberId);
  if (!slider || !number) return;

  slider.addEventListener("input", () => {
    number.value = slider.value;
    onInputChange();
  });

  number.addEventListener("input", () => {
    slider.value = number.value;
    onInputChange();
  });
}

function getFormData() {
  const age = parseFloat(document.getElementById("num-age").value) || 40;
  const gender = document.getElementById("select-gender").value;
  const height = parseFloat(document.getElementById("num-height").value) || 170;
  const weight = parseFloat(document.getElementById("num-weight").value) || 70;
  const activity = document.getElementById("select-activity").value;
  const diet = document.getElementById("select-diet").value;

  const vitd = parseFloat(document.getElementById("num-vitd").value) || 25;
  const iron = parseFloat(document.getElementById("num-iron").value) || 15;
  const calcium = parseFloat(document.getElementById("num-calcium").value) || 900;
  const b12 = parseFloat(document.getElementById("num-b12").value) || 2.5;

  const omega = parseFloat(document.getElementById("num-omega").value) || 1.5;
  const zinc = parseFloat(document.getElementById("num-zinc").value) || 12;
  const magnesium = parseFloat(document.getElementById("num-magnesium").value) || 300;
  const protein = parseFloat(document.getElementById("num-protein").value) || 65;

  const waist = parseFloat(document.getElementById("num-waist")?.value) || 84.0;
  const sys_bp = parseFloat(document.getElementById("num-sys")?.value) || 122.0;
  const dia_bp = parseFloat(document.getElementById("num-dia")?.value) || 78.0;

  return {
    Age: age,
    Gender: gender,
    Height: height,
    Weight: weight,
    Physical_Activity: activity,
    Diet_Quality: diet,
    Vitamin_D: vitd,
    Iron: iron,
    Calcium: calcium,
    Vitamin_B12: b12,
    Omega_3: omega,
    Zinc: zinc,
    Magnesium: magnesium,
    Protein: protein,
    Waist: waist,
    Systolic_BP: sys_bp,
    Diastolic_BP: dia_bp
  };
}

function setFormData(data) {
  const fields = [
    ["age", data.age], ["height", data.height], ["weight", data.weight],
    ["vitd", data.vitd], ["iron", data.iron], ["calcium", data.calcium],
    ["b12", data.b12], ["omega", data.omega], ["zinc", data.zinc],
    ["magnesium", data.magnesium], ["protein", data.protein],
    ["waist", data.waist], ["sys", data.sys_bp], ["dia", data.dia_bp]
  ];

  fields.forEach(([key, val]) => {
    if (val !== undefined) {
      const s = document.getElementById(`slider-${key}`);
      const n = document.getElementById(`num-${key}`);
      if (s) s.value = val;
      if (n) n.value = val;
    }
  });

  if (data.gender) document.getElementById("select-gender").value = data.gender;
  if (data.activity) document.getElementById("select-activity").value = data.activity;
  if (data.diet) document.getElementById("select-diet").value = data.diet;

  updateSummary();
  runPrediction();
}

function updateSummary() {
  const data = getFormData();
  const heightM = data.Height / 100.0;
  const bmi = heightM > 0 ? (data.Weight / (heightM * heightM)) : 0;

  let bmiCat = "Normal";
  let bmiClass = "low";
  if (bmi < 18.5) {
    bmiCat = "Underweight";
    bmiClass = "med";
  } else if (bmi < 25) {
    bmiCat = "Normal weight";
    bmiClass = "low";
  } else if (bmi < 30) {
    bmiCat = "Overweight";
    bmiClass = "med";
  } else {
    bmiCat = "Obesity";
    bmiClass = "high";
  }

  document.getElementById("sum-bmi-val").innerText = bmi.toFixed(1);
  document.getElementById("sum-bmi-cat").innerText = bmiCat;
  document.getElementById("sum-activity-val").innerText = data.Physical_Activity;
  document.getElementById("sum-diet-val").innerText = data.Diet_Quality;
  document.getElementById("sum-nutrients-val").innerText = "8 / 8 Active";
}

function scoreState(score) {
  if (score < 0.33) return { text: "Lower model score", class: "low" };
  if (score < 0.67) return { text: "Intermediate model score", class: "med" };
  return { text: "Higher model score", class: "high" };
}

// Client-side fallback computation
function evaluateFallback(targetKey, data) {
  if (!fallbackWeights || !fallbackWeights[targetKey]) {
    // Calibrated heuristic approximation
    const bmi = data.Weight / ((data.Height / 100.0) ** 2);
    if (targetKey === "cardio") {
      const z = -4.5 + (data.Age * 0.05) + (bmi * 0.05) + (data.Gender === "M" ? 0.3 : -0.2) + ((data.Systolic_BP || 120) * 0.015);
      return 1 / (1 + Math.exp(-z));
    }
    if (targetKey === "diabetes") {
      const z = -4.8 + (data.Age * 0.04) + (bmi * 0.08) - (data.Physical_Activity === "High" ? 0.4 : 0.0);
      return 1 / (1 + Math.exp(-z));
    }
    if (targetKey === "osteo") {
      const z = -4.2 + (data.Age * 0.06) - (bmi * 0.04) + (data.Gender === "F" ? 0.5 : -0.5) - (data.Calcium * 0.0006);
      return 1 / (1 + Math.exp(-z));
    }
    if (targetKey === "anemia") {
      const z = -2.8 - (data.Iron * 0.06) - (data.Protein * 0.01) + (data.Gender === "F" ? 0.6 : -0.3);
      return 1 / (1 + Math.exp(-z));
    }
    return 0.15;
  }

  const model = fallbackWeights[targetKey];
  const heightM = data.Height / 100.0;
  const bmiCalc = heightM > 0 ? (data.Weight / (heightM * heightM)) : 24.0;
  
  let bmiCategory = "Normal";
  if (bmiCalc < 18.5) bmiCategory = "Underweight";
  else if (bmiCalc >= 30) bmiCategory = "Obesity";
  else if (bmiCalc >= 25) bmiCategory = "Overweight";

  // Build full row
  const row = {
    Age: data.Age, Height: data.Height, Weight: data.Weight, BMI_calc: bmiCalc,
    Waist: data.Waist, Systolic_BP: data.Systolic_BP, Diastolic_BP: data.Diastolic_BP,
    Vitamin_D: data.Vitamin_D, Iron: data.Iron, Calcium: data.Calcium,
    Vitamin_B12: data.Vitamin_B12, Omega_3: data.Omega_3, Zinc: data.Zinc,
    Magnesium: data.Magnesium, Protein: data.Protein,
    Calories: null, Sugars: null, Fiber: null, Total_Fat: null, Sat_Fat: null, Sodium: null, Potassium: null,
    Waist_to_Height_Ratio: data.Waist / (data.Height + 1e-5),
    Sodium_to_Potassium_Ratio: null,
    Calcium_to_Magnesium_Ratio: data.Calcium / (data.Magnesium + 1e-5),
    VitD_Calcium_Index: data.Vitamin_D * data.Calcium,
    Sugar_to_Fiber_Ratio: null,
    Sat_Fat_Ratio: null,
    Pulse_Pressure: Math.max(0, data.Systolic_BP - data.Diastolic_BP),
    MAP: data.Diastolic_BP + (Math.max(0, data.Systolic_BP - data.Diastolic_BP) / 3.0),
    BMI_Age_Interaction: bmiCalc * data.Age,
    Iron_to_Protein_Ratio: data.Iron / (data.Protein + 1e-5)
  };

  // Transform numeric features with median imputation + standard scaling
  const x = [];
  for (let i = 0; i < NUMERIC_COLS.length; i++) {
    const col = NUMERIC_COLS[i];
    let val = row[col];
    if (val === null || val === undefined || isNaN(val)) {
      val = model.num_medians[i];
    }
    const scaled = (val - model.num_means[i]) / (model.num_scales[i] || 1.0);
    x.push(scaled);
  }

  // Transform categorical features (Gender, Physical_Activity, Diet_Quality, BMI_Category)
  const cats = model.cat_categories;
  const catVals = [data.Gender, data.Physical_Activity, data.Diet_Quality, bmiCategory];
  for (let i = 0; i < cats.length; i++) {
    const allowed = cats[i];
    const cur = catVals[i];
    for (let j = 0; j < allowed.length; j++) {
      x.push(allowed[j] === cur ? 1.0 : 0.0);
    }
  }

  // Logistic dot product
  let z = model.intercept;
  for (let i = 0; i < x.length && i < model.coef.length; i++) {
    z += model.coef[i] * x[i];
  }
  return 1 / (1 + Math.exp(-z));
}

async function runPrediction() {
  const profile = getFormData();
  const btn = document.getElementById("btn-submit");
  if (btn) btn.innerHTML = `<span>⏳ Evaluating...</span>`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });

    if (res.ok) {
      const data = await res.json();
      if (data.success && data.predictions) {
        updateUIWithScores({
          cardio: data.predictions.Cardio.score,
          diabetes: data.predictions.Diabetes.score,
          osteo: data.predictions.Osteoporosis.score,
          anemia: data.predictions.Anemia.score
        }, "Vercel Serverless API");
        return;
      }
    }
    throw new Error("API call fallback");
  } catch (err) {
    // Fallback to client-side evaluation
    const scores = {
      cardio: evaluateFallback("cardio", profile),
      diabetes: evaluateFallback("diabetes", profile),
      osteo: evaluateFallback("osteoporosis", profile),
      anemia: evaluateFallback("anemia", profile)
    };
    updateUIWithScores(scores, "Client ML Engine");
  } finally {
    if (btn) btn.innerHTML = `<span>⚡ Generate Disease Model Scores</span>`;
  }
}

function updateUIWithScores(scores, source) {
  const badge = document.getElementById("api-status-badge");
  if (badge) {
    badge.innerHTML = `<span class="dot"></span> ${source} Active`;
  }

  const map = [
    { key: "cardio", idPct: "score-cardio", idBar: "bar-cardio", idState: "state-cardio", val: scores.cardio },
    { key: "diabetes", idPct: "score-diabetes", idBar: "bar-diabetes", idState: "state-diabetes", val: scores.diabetes },
    { key: "osteo", idPct: "score-osteo", idBar: "bar-osteo", idState: "state-osteo", val: scores.osteo },
    { key: "anemia", idPct: "score-anemia", idBar: "bar-anemia", idState: "state-anemia", val: scores.anemia }
  ];

  map.forEach(item => {
    const pct = Math.max(0.1, Math.min(99.9, item.val * 100));
    const state = scoreState(item.val);

    const pctEl = document.getElementById(item.idPct);
    const barEl = document.getElementById(item.idBar);
    const stateEl = document.getElementById(item.idState);

    if (pctEl) pctEl.innerText = `${pct.toFixed(1)}%`;
    if (barEl) {
      barEl.style.width = `${pct}%`;
      barEl.className = `meter-fill ${state.class}`;
    }
    if (stateEl) {
      stateEl.innerText = state.text;
      stateEl.className = `state-badge ${state.class}`;
    }
  });

  // Update dynamic comparison rows
  const tableCardio = document.getElementById("table-val-cardio");
  const tableDiabetes = document.getElementById("table-val-diabetes");
  const tableOsteo = document.getElementById("table-val-osteo");
  const tableAnemia = document.getElementById("table-val-anemia");

  if (tableCardio) tableCardio.innerText = `${(scores.cardio * 100).toFixed(1)}%`;
  if (tableDiabetes) tableDiabetes.innerText = `${(scores.diabetes * 100).toFixed(1)}%`;
  if (tableOsteo) tableOsteo.innerText = `${(scores.osteo * 100).toFixed(1)}%`;
  if (tableAnemia) tableAnemia.innerText = `${(scores.anemia * 100).toFixed(1)}%`;
}

function onInputChange() {
  updateSummary();
  const live = document.getElementById("check-live")?.checked;
  if (live) {
    runPrediction();
  }
}

// Initializer
document.addEventListener("DOMContentLoaded", () => {
  const syncPairs = [
    ["slider-age", "num-age"],
    ["slider-height", "num-height"],
    ["slider-weight", "num-weight"],
    ["slider-vitd", "num-vitd"],
    ["slider-iron", "num-iron"],
    ["slider-calcium", "num-calcium"],
    ["slider-b12", "num-b12"],
    ["slider-omega", "num-omega"],
    ["slider-zinc", "num-zinc"],
    ["slider-magnesium", "num-magnesium"],
    ["slider-protein", "num-protein"],
    ["slider-waist", "num-waist"],
    ["slider-sys", "num-sys"],
    ["slider-dia", "num-dia"]
  ];

  syncPairs.forEach(([s, n]) => setupSync(s, n));

  ["select-gender", "select-activity", "select-diet"].forEach(id => {
    document.getElementById(id)?.addEventListener("change", onInputChange);
  });

  // Presets
  document.querySelectorAll("[data-preset]").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      if (PRESETS[key]) {
        setFormData(PRESETS[key]);
      }
    });
  });

  // Submit button
  document.getElementById("btn-submit")?.addEventListener("click", (e) => {
    e.preventDefault();
    runPrediction();
  });

  // Advanced vitals toggle
  const advToggle = document.getElementById("toggle-advanced");
  const advSection = document.getElementById("advanced-vitals-section");
  if (advToggle && advSection) {
    advToggle.addEventListener("click", () => {
      advSection.style.display = advSection.style.display === "none" ? "flex" : "none";
      advToggle.innerText = advSection.style.display === "none" ? "➕ Show Clinical Vitals (Waist, Blood Pressure)" : "➖ Hide Clinical Vitals";
    });
  }

  // Initial trigger
  updateSummary();
  runPrediction();
});

