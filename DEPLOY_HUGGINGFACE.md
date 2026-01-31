# Deploy to Hugging Face Spaces

Step-by-step guide to deploy the Ethiopian Crop Recommendation app on Hugging Face Spaces.

## Prerequisites

- [Hugging Face account](https://huggingface.co/join)
- [Git](https://git-scm.com/) installed
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (optional, for `huggingface-cli login`)

---

## Option A: Deploy via Hugging Face Website (Easiest)

### 1. Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a **Space name** (e.g. `crop-recommendation-ethiopia`)
3. Select **Docker** as the SDK
4. Choose **CPU basic** (or upgrade if needed)
5. Set visibility (Public or Private)
6. Click **Create Space**

### 2. Clone and push your code

```bash
# Clone your Space (replace YOUR_USERNAME with your HF username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/crop-recommendation-ethiopia
cd crop-recommendation-ethiopia

# Copy files from your project
cp -r /path/to/crop_recommendation/src .
cp -r /path/to/crop_recommendation/static .
cp -r /path/to/crop_recommendation/models .
cp /path/to/crop_recommendation/Dockerfile .
cp /path/to/crop_recommendation/requirements-deploy.txt .
cp /path/to/crop_recommendation/README.md .

# Commit and push
git add .
git commit -m "Deploy crop recommendation app"
git push
```

### 3. Wait for build

Hugging Face will build the Docker image and start your app. Check the **Logs** tab for progress. Once ready, your app will be live at:

```
https://huggingface.co/spaces/YOUR_USERNAME/crop-recommendation-ethiopia
```

---

## Option B: Deploy from existing Git repo

If your project is already a Git repo:

### 1. Add Hugging Face as remote

```bash
cd /home/dmk/Documents/crop_recommendation

# Create Space first at huggingface.co/new-space, then:
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/crop-recommendation-ethiopia
```

### 2. Push to Hugging Face

```bash
git add Dockerfile requirements-deploy.txt README.md src/ static/ models/
git commit -m "Add Hugging Face deployment"
git push hf main
```

> **Note:** Hugging Face uses `main` as default branch. If your branch is `master`, use `git push hf master:main`.

---

## Files included in deployment

| File / Folder             | Purpose                               |
| ------------------------- | ------------------------------------- |
| `Dockerfile`              | Container definition for HF Spaces    |
| `requirements-deploy.txt` | Minimal Python dependencies           |
| `README.md`               | Space metadata (YAML) + documentation |
| `src/`                    | FastAPI application                   |
| `static/`                 | Web UI (HTML, CSS, JS)                |
| `models/`                 | Trained model, scaler, label encoder  |

---

## Verify locally before pushing

```bash
# Build and run with Docker (simulates HF environment)
docker build -t crop-rec .
docker run -p 7860:7860 crop-rec

# Open http://localhost:7860
```

---

## Troubleshooting

### Build fails

- Check **Logs** in the Space for error messages
- Ensure `models/` contains: `best_crop_model.pkl`, `scaler_merged.pkl`, `label_encoder_merged.pkl`
- Verify `requirements-deploy.txt` has no typos

### App loads but predict fails

- Confirm model files are in `models/` and committed (not in `.gitignore`)
- Model files are ~4MB total; no need for Git LFS

### Static files 404

- Ensure `static/index.html` exists
- Root `/` redirects to `/static/`; the form fetches `/predict` from the same origin

### Out of memory

- Free-tier CPU Spaces have limited RAM; the app is lightweight and should run fine
- If needed, upgrade to a paid Space

---

## Updating the Space

After making changes:

```bash
git add .
git commit -m "Your update message"
git push hf main
```

Hugging Face will rebuild and redeploy automatically.
