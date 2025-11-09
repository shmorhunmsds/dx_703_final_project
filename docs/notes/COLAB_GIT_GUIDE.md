# Google Colab + GitHub Workflow Guide

**Quick reference for working with our project repository in Google Colab**

---

## 🚀 First Time Setup (Do this once per Colab session)

### Step 1: Clone the Repository

```python
# Run this in a Colab code cell
!git clone https://github.com/YOUR_USERNAME/dx_703_final_project.git
%cd dx_703_final_project
```

Replace `YOUR_USERNAME` with the actual GitHub username.

### Step 2: Configure Git (Optional but Recommended)

```python
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"
```

---

## 📥 Pulling Latest Changes (Start of work session)

**Always pull before you start working to get the latest changes:**

```python
# Navigate to the project directory
%cd /content/dx_703_final_project

# Pull latest changes from main branch
!git pull origin main
```

**If you get merge conflicts**, the safest approach in Colab:
```python
# Stash your local changes
!git stash

# Pull the latest
!git pull origin main

# Apply your stashed changes back
!git stash pop
```

---

## 📤 Pushing Your Changes (End of work session)

### Step 1: Check What Changed

```python
!git status
```

This shows which files you've modified.

### Step 2: Add Your Changes

```python
# Add all changed files
!git add .

# OR add specific files only
!git add Milestone_01.ipynb
!git add some_script.py
```

### Step 3: Commit Your Changes

```python
!git commit -m "Brief description of what you changed"
```

**Example commit messages:**
- `"Completed Problem 2 questions"`
- `"Added baseline model for text classification"`
- `"Fixed data preprocessing pipeline"`

### Step 4: Push to GitHub

```python
!git push origin main
```

**If you get an authentication error**, you'll need to use a Personal Access Token (see Authentication section below).

---

## 🔐 Authentication (GitHub Personal Access Token)

GitHub no longer accepts passwords for git operations. You need a **Personal Access Token (PAT)**.

### Create a Personal Access Token:

1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Colab Access")
4. Set expiration (30 days is fine for class project)
5. Check the `repo` scope (gives full repository access)
6. Click "Generate token"
7. **COPY THE TOKEN** (you won't see it again!)

### Use Token in Colab:

**Option A: Include in the git URL** (easier but less secure)
```python
# Clone with token
!git clone https://YOUR_TOKEN@github.com/USERNAME/dx_703_final_project.git

# Push with token
!git push https://YOUR_TOKEN@github.com/USERNAME/dx_703_final_project.git main
```

**Option B: Use git credential helper** (more secure)
```python
# Store credentials for the session
!git config --global credential.helper store

# First push will ask for username and token
# Username: your GitHub username
# Password: paste your Personal Access Token (NOT your GitHub password)
!git push origin main
```

---

## 📋 Complete Workflow Example

```python
# ==========================================
# START OF COLAB SESSION
# ==========================================

# 1. Clone repo (first time) OR navigate to it
%cd /content/dx_703_final_project

# 2. Pull latest changes
!git pull origin main

# 3. Check current status
!git status


# ==========================================
# DO YOUR WORK HERE
# - Run your notebook cells
# - Make changes to code
# - Test your models
# ==========================================


# ==========================================
# END OF COLAB SESSION - SAVE YOUR WORK
# ==========================================

# 4. Check what changed
!git status

# 5. Add your changes
!git add .

# 6. Commit with a message
!git commit -m "Describe what you did"

# 7. Push to GitHub
!git push origin main

# 8. Verify it worked
!git status
```

---

## 🔍 Useful Git Commands for Colab

### Check Repository Status
```python
!git status                    # See what files changed
!git log --oneline -5          # See last 5 commits
!git branch                    # See current branch
```

### View Changes
```python
!git diff                      # See uncommitted changes
!git diff HEAD~1               # Compare with previous commit
```

### Undo Changes (USE CAREFULLY!)
```python
# Discard changes to a specific file
!git checkout -- filename.py

# Discard ALL uncommitted changes (DANGEROUS!)
!git reset --hard HEAD

# Undo last commit but keep changes
!git reset --soft HEAD~1
```

---

## 🎯 Best Practices for Team Work

### 1. **Always Pull Before You Start**
```python
!git pull origin main
```
This prevents merge conflicts.

### 2. **Commit Often with Clear Messages**
```python
!git commit -m "Added data preprocessing for Problem 2"
```
Not: `"fixed stuff"` or `"changes"`

### 3. **Push Regularly**
Don't wait until the deadline! Push after each work session.

### 4. **Check Status Before Pushing**
```python
!git status
```
Make sure you're committing what you intend to.

### 5. **Don't Commit Large Files**
Our `.gitignore` already excludes:
- Dataset files (`huffpost_splits/`)
- Model checkpoints (`.h5`, `.pt`)
- Large CSVs and images (if needed, discuss with team)

### 6. **Communicate with Team**
- Use Slack/Discord to coordinate who's working on what
- Avoid editing the same cells at the same time
- Use comments in code to mark your sections

---

## 🐛 Common Issues & Solutions

### Issue: "Permission denied (publickey)"
**Solution:** Use Personal Access Token instead of SSH
```python
# Use HTTPS URL with token instead of SSH
!git remote set-url origin https://YOUR_TOKEN@github.com/USERNAME/dx_703_final_project.git
```

### Issue: "Your branch is behind 'origin/main'"
**Solution:** Pull the latest changes
```python
!git pull origin main
```

### Issue: "Merge conflict"
**Solution:** Simplest approach in Colab:
```python
# Save your work elsewhere (copy cells to a text file)
# Reset to remote state
!git fetch origin
!git reset --hard origin/main

# Then manually re-apply your changes
```

### Issue: "Changes not staged for commit"
**Solution:** You forgot to `git add`
```python
!git add .
!git commit -m "Your message"
```

### Issue: Colab disconnected and lost work
**Solution:** Enable auto-save and checkpoint regularly
- Colab saves to Google Drive automatically
- Manually download `.ipynb` file periodically
- Push to GitHub frequently!

---

## 📁 Project Structure

```
dx_703_final_project/
├── Milestone_01.ipynb          # Main notebook (EDIT THIS)
├── README.md                   # Project overview
├── analysis_scripts/           # Helper scripts (DON'T PUSH)
├── huffpost_splits/            # Dataset (DON'T PUSH - too large)
├── *.png                       # Visualizations (PUSH THESE)
├── *.csv                       # Data summaries (PUSH THESE)
└── .gitignore                  # Specifies what not to push
```

**What to commit:**
✅ `Milestone_01.ipynb` (your work!)
✅ Visualization PNG files
✅ Data CSV files (summaries, stats)
✅ README updates

**What NOT to commit:**
❌ `analysis_scripts/` folder
❌ `huffpost_splits/` (dataset)
❌ `__pycache__/` folders
❌ `.ipynb_checkpoints/`
❌ Large model files

---

## 🎓 Quick Reference Card

| Task | Command |
|------|---------|
| Clone repo | `!git clone https://github.com/USER/REPO.git` |
| Pull latest | `!git pull origin main` |
| Check status | `!git status` |
| Add all changes | `!git add .` |
| Commit | `!git commit -m "message"` |
| Push | `!git push origin main` |
| View history | `!git log --oneline` |

---

## 👥 Team Workflow Tips

### Before Starting Work:
1. Open Colab
2. Clone or navigate to repo
3. **Pull latest changes**: `!git pull origin main`
4. Check status: `!git status`
5. Start working

### After Finishing Work:
1. Save your notebook
2. Check changes: `!git status`
3. Add files: `!git add .`
4. Commit: `!git commit -m "what you did"`
5. **Push**: `!git push origin main`
6. Notify team in chat: "Pushed my changes for Problem X"

### During Collaboration:
- **Peter** handles Problem 1 (cells 15-29)
- **August** handles Problem 2 (cells 31+)
- **Emma** reviews Problem 2 code
- Always pull before starting, push when done
- Use cell comments to mark your sections

---

## 🆘 Need Help?

1. **Git Documentation:** https://git-scm.com/doc
2. **GitHub Guides:** https://guides.github.com/
3. **Ask the team** in your project Slack/Discord
4. **TA Office Hours** for git help
5. **Stack Overflow:** Search "git [your error message]"

---

**💡 Pro Tip:** Create a new Colab code cell at the top of your notebook with your common git commands for easy reference:

```python
# ============================================
# GIT WORKFLOW - Run at start and end of session
# ============================================

# START: Pull latest changes
# !git pull origin main

# END: Save your work
# !git add .
# !git commit -m "Describe what you did"
# !git push origin main
```

Keep this cell commented out and uncomment lines as needed!

---

**Last Updated:** October 22, 2025
**For:** DX 703 Final Project - Milestone 1
