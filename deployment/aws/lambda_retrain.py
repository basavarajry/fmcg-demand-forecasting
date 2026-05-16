"""
AWS Lambda handler for automated retraining triggers.

Deploy with:
  zip -r lambda.zip deployment/aws/lambda_retrain.py src/ configs/
  aws lambda create-function --function-name fmcg-retrain-trigger ...
"""

import json
import os


def handler(event, context):
    """Trigger retraining when S3 new data event or CloudWatch alarm fires."""
    trigger_reason = "unknown"
    if "Records" in event:
        trigger_reason = "new_s3_data"
    elif event.get("source") == "aws.cloudwatch":
        trigger_reason = "wmape_threshold"

    api_url = os.environ.get("API_URL", "http://localhost:8000")
    token = os.environ.get("API_ACCESS_TOKEN", "")

    import urllib.request

    payload = json.dumps({"phases": ["ml", "tft"], "force": True}).encode()
    req = urllib.request.Request(
        f"{api_url}/retrain",
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
        return {"statusCode": 200, "body": body, "trigger": trigger_reason}
    except Exception as e:
        return {"statusCode": 500, "body": str(e), "trigger": trigger_reason}
