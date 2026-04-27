install:
    pip install --upgrade pip &&\
        pip install -r requirements.txt

format: 
    black *.py 

train:
    python train.py

eval:
    echo "## Model Metrics" > report.md
    cat ./Results/metrics.txt >> report.md
    
    echo '\n## Confusion Matrix Plot' >> report.md
    echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
    
    cml comment create report.md
        
update-branch:
    git config --global user.name $(USER_NAME)
    git config --global user.email $(USER_EMAIL)
    git commit -am "Update with new results"
    git push --force origin HEAD:update

hf-login: 
    pip install -U huggingface_hub
    git pull origin main
    git switch main
    hf auth login --token $(HF_TOKEN) --add-to-git-credential

push-hub: 
    hf upload manuelcastiblan/Drug-Classification ./App --repo-type=space --commit-message="Sync App files"
    hf upload manuelcastiblan/Drug-Classification ./Model /Model --repo-type=space --commit-message="Sync Model"
    hf upload manuelcastiblan/Drug-Classification ./Results /Metrics --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
