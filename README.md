# ITU BDS MLOPS'25 - Project

This repository contains our implementation of a fully reproducible end-to-end MLOps pipeline developed for the ITU BDS MLOps 2025 course. The goal was to transform the original exploratory notebook into a modular, automated ML system using DVC for data management, MLflow for experiment tracking, and GitHub Actions + Dagger for CI/CD automation.


### 1. Repository Structure 

```text
Happy-Days/
│
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml     # CI/CD Orchestration: Executes Dagger, then uploads the generated artifacts.
│
├── data/
│   └── raw_data.csv.dvc        # DVC Pointer
│
├── src/                        # Core Python Pipeline
│   ├── data_prep.py            # Phase 1: Executes 'dvc pull' and performs data cleaning.
│   ├── train.py                # Phase 2: Training, MLflow run logging, and local model saving.
│   └── deploy.py               # Phase 3: Model selection, registration to the 'mlruns' directory.
│
├── main.go                     # Orchestrator (Dagger): Defines the pipeline steps and exports the generated folders.
├── requirements.txt            # Dependencies: Defines the Python environment for the Dagger container.
└── README.md
```
## How To Run the Project on Github UI

Navigate to Actions: On your GitHub repository page, click the Actions tab.

Select Workflow: On the left sidebar, click on "MLOps Pipeline"

Trigger Run: Look for the "Run workflow" button or dropdown menu near the top of the workflow page.

Execute: Click the button to start the job on a GitHub-hosted runner.



## How To Run the Project Locally 

If you wish to run the project locally, you must have the following tools installed on your local machine:

  **Go:** Required to run the Dagger orchestrator (`main.go`).
  
  **Dagger CLI:** The command-line interface for running the Dagger pipeline.

  **Docker Desktop:** Must be running! Required by Dagger to execute the pipeline.

### Execution

1. **Start by cloning the repository**

2. **Execute the MLOps Pipeline:**
   
    Type the following command from the root repo:
    **go run main.go**

    this command will handle everything: initializing the Dagger container, installing Python dependencies, performing the DVC pull, and             executing the three pipeline stages (`data_prep.py`, `train.py`, `deploy.py`). 

### Post-Execution

After a successful run, the following directories will be generated in your project root, containing the results of the pipeline:

* **`mlruns/`:** Contains the local MLflow Tracking data and the Model Registry, where the final **Logistic Regression** model is registered etc.
* **`artifacts/`:** Contains the final generated files, including `model.pkl`, `train_data_gold.csv`, and `model_results.json`.

