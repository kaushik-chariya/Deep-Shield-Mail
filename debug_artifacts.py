import os
import mlflow
import dagshub

token = os.getenv("DEEPSHIELD_TEST")
tracking_uri = f"https://kaushik-chariya:{token}@dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"

os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
os.environ["DAGSHUB_USER_TOKEN"] = token

dagshub.init(repo_owner="kaushik-chariya", repo_name="Deep-Shield-Mail", mlflow=True)
mlflow.set_tracking_uri(tracking_uri)

client = mlflow.MlflowClient(tracking_uri=tracking_uri)

# Naya run ID
run_id = "eff2da41389f47d0a0e2379202d7f896"
artifacts = client.list_artifacts(run_id)
print("ROOT artifacts:")
for a in artifacts:
    print(f"  {a.path}  (dir={a.is_dir})")

# Saare subdirs check karo
for a in artifacts:
    if a.is_dir:
        sub = client.list_artifacts(run_id, a.path)
        print(f"\n  {a.path}/ contents:")
        for s in sub:
            print(f"    {s.path}")
