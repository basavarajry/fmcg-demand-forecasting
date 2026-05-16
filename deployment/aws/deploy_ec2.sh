#!/bin/bash
# Deploy FMCG Demand Forecasting API to AWS EC2
set -euo pipefail

INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
KEY_NAME="${KEY_NAME:-forecast-key}"
SECURITY_GROUP="${SECURITY_GROUP:-forecast-sg}"
S3_BUCKET="${S3_BUCKET:-fmcg-forecast-models}"
REGION="${AWS_REGION:-us-east-1}"

echo "=== FMCG Forecast EC2 Deployment ==="

# Upload model artifacts to S3
if [ -d "models" ]; then
  aws s3 sync models/ "s3://${S3_BUCKET}/models/" --region "${REGION}"
  echo "Models synced to s3://${S3_BUCKET}/models/"
fi

# User data script for EC2
cat > /tmp/user-data.sh << 'USERDATA'
#!/bin/bash
yum update -y
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone and deploy (replace with your repo URL)
cd /home/ec2-user
# git clone <your-repo-url> fmcg-demand-forecasting
# cd fmcg-demand-forecasting

# Pull models from S3
# aws s3 sync s3://${S3_BUCKET}/models/ models/

# docker compose up -d
USERDATA

echo "User-data script prepared at /tmp/user-data.sh"
echo "Launch EC2 instance manually or via:"
echo "  aws ec2 run-instances --image-id ami-0c55b159cbfafe1f0 --instance-type ${INSTANCE_TYPE} \\"
echo "    --key-name ${KEY_NAME} --security-groups ${SECURITY_GROUP} \\"
echo "    --user-data file:///tmp/user-data.sh --region ${REGION}"

echo "=== Deployment script complete ==="
