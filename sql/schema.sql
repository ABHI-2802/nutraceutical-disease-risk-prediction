CREATE TABLE health_profile (
    profile_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    age SMALLINT NOT NULL CHECK (age > 0),
    gender VARCHAR(20), height_cm NUMERIC(6,2), weight_kg NUMERIC(6,2),
    physical_activity VARCHAR(30), diet_quality VARCHAR(30),
    vitamin_d NUMERIC, iron NUMERIC, calcium NUMERIC, vitamin_b12 NUMERIC,
    omega_3 NUMERIC, zinc NUMERIC, magnesium NUMERIC, protein NUMERIC,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE model_prediction (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    profile_id BIGINT REFERENCES health_profile(profile_id),
    target_name VARCHAR(30) NOT NULL CHECK (target_name IN ('Anemia','Osteoporosis','Diabetes','Cardio')),
    score NUMERIC(8,7) NOT NULL CHECK (score >= 0 AND score <= 1),
    model_version VARCHAR(100) NOT NULL,
    disclaimer_acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Example: latest scores by disease (never treat these as clinical diagnoses)
SELECT target_name, AVG(score) AS mean_model_score, COUNT(*) AS predictions
FROM model_prediction GROUP BY target_name ORDER BY target_name;

-- Example: prediction volume by day
SELECT DATE(created_at) AS prediction_date, target_name, COUNT(*) AS prediction_count
FROM model_prediction GROUP BY DATE(created_at), target_name ORDER BY prediction_date, target_name;
