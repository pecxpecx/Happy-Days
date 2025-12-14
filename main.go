// main.go
package main

import (
    "context"
    "fmt"
    "os"
    "dagger.io/dagger"
)

// Main function to initialize Dagger and run the pipeline
func main() {
    ctx := context.Background()

    // 1. Initialize Dagger Client
    client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
    if err != nil {
        fmt.Println("Error connecting to Dagger:", err)
        os.Exit(1)
    }
    defer client.Close()

    // 2. Run the Pipeline stages
    if err := RunPipeline(ctx, client); err != nil {
        fmt.Println("Pipeline execution failed:", err)
        os.Exit(1)
    }
    fmt.Println("Pipeline executed successfully!")
}

// RunPipeline defines and executes the MLOps workflow
func RunPipeline(ctx context.Context, client *dagger.Client) error {
    
    // Mount the entire host repository (the current working directory)
    repo := client.Host().Directory(".")

    // --- 1. Define Base Environment and Install Dependencies ---

    // Set up the Python container
    base := client.Container().
        From("python:3.11-slim").
        // Mount the host repo root at /repo
        WithDirectory("/repo", repo).
        // Set working directory to repo root (where requirements.txt lives)
        WithWorkdir("/repo") 

    // Install Python dependencies
    fmt.Println("--- 🛠️ Installing Dependencies ---")
    base = base.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

    // --- 2. Data Preparation Stage (src/data_prep.py) ---
    // The script expects raw data at /repo/data/raw_data.csv and outputs to /repo/artifacts/
    fmt.Println("--- 🚀 Stage 1: Running Data Preparation ---")
    dataPrep := base.WithExec([]string{"python", "src/data_prep.py"})
    if _, err := dataPrep.Stdout(ctx); err != nil {
        return err
    }
    
    // --- 3. Model Training Stage (src/train.py) ---
    // The script reads from /repo/artifacts/ and outputs to /repo/mlruns/
    fmt.Println("--- 🚀 Stage 2: Running Model Training ---")
    train := dataPrep.WithExec([]string{"python", "src/train.py"})
    if _, err := train.Stdout(ctx); err != nil {
        return err
    }

    // --- 4. Model Deployment Stage (src/deploy.py) ---
    // The script interacts with the MLflow registry in /repo/mlruns/
    fmt.Println("--- 🚀 Stage 3: Running Model Deployment ---")
    deployment := train.WithExec([]string{"python", "src/deploy.py"})
    if _, err := deployment.Stdout(ctx); err != nil {
        return err
    }
    
    // --- 5. Export Final Artifacts (required for GitHub Workflow) ---
    
    // We export to the runner's workspace directly
    fmt.Println("--- Exporting Final Artifacts: artifacts/ and mlruns/ ---")

    // Export the artifacts folder
    _, err := deployment.
        Directory("/repo/artifacts").
        Export(ctx, "artifacts")
    if err != nil {
        return fmt.Errorf("failed to export artifacts: %w", err)
    }

    // Export the mlruns history folder
    _, err = deployment.
        Directory("/repo/mlruns").
        Export(ctx, "mlruns")
    if err != nil {
        return fmt.Errorf("failed to export mlruns: %w", err)
    }

    return nil
}