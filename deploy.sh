#!/bin/bash

# Set variables
EC2_USER="ec2-user"
EC2_IP="3.9.165.20"
PEM_KEY="~/.ssh/ai-fhir-server.pem"
ENV_DIR="/home/ec2-user/livekit-agent"
APP_DIR="/home/ec2-user/livekit-agent/livekit-agent"  # Replace with your actual agent path

echo "🚀 Deploying LiveKit Agent to EC2 ($EC2_IP)..."

# Step 1: Upload latest code using rsync
echo "📂 Syncing files..."
rsync -avz --progress -e "ssh -i $PEM_KEY" --exclude '__pycache__' --exclude '.venv' --exclude '.git' ./ $EC2_USER@$EC2_IP:$APP_DIR

# Step 2: SSH into EC2 and restart the agent
echo "🔄 Restarting agent..."
ssh -i $PEM_KEY $EC2_USER@$EC2_IP <<EOF
  cd $ENV_DIR
  source venv/bin/activate
  cd $APP_DIR
  pkill -f "python3 agent.py start" || true
  nohup python3 agent.py start > agent.log 2>&1 &
  echo "✅ Agent restarted"
  exit
EOF

echo "✅ Deployment complete!"