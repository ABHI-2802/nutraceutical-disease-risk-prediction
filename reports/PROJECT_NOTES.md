# Important modeling notes

- Labels are treated as independent binary targets; co-occurrence is allowed.
- Dataset has 1,000 rows and 22 columns; no missing cells or exact duplicate rows in the supplied copy.
- The dataset provenance and whether disease labels were clinically observed or generated from rules is unknown from the CSV alone.
- Disease status features (`VitD_Status`, `Iron_Status`, `Calcium_Status`) are excluded. All four disease target columns are excluded from every feature matrix.
- Because the dataset is modest and structured, do not advertise the model as clinically valid or claim that high test scores prove real-world performance.
- App labels outputs as dataset-pattern scores; not diagnoses.
