# Source the Conda configuration
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create a new conda environment with Python 3.8
ENV_NAME="myenv"
# Check if the environment already exists
conda info --envs | grep -w "$ENV_NAME" > /dev/null
if [ $? -eq 0 ]; then
    echo "Activating Conda environment $ENV_NAME"
else
    echo "Creating and activating new Conda environment $ENV_NAME with Python 3.8"
    conda create -n "$ENV_NAME" python=3.8
fi

# install phishintention
PACKAGE_NAME="phishintention"
installed_packages=$(conda run -n "$ENV_NAME" conda list)
if echo "$installed_packages" | grep -q "$PACKAGE_NAME"; then
  echo "PhishIntention is already installed, skip installation"
else
  git clone https://github.com/lindsey98/PhishIntention.git
  cd PhishIntention
  chmod +x ./setup.sh
  ./setup.sh
  cd ../
  rm -rf PhishIntention
fi

## Install MyXDriver
PACKAGE_NAME="xdriver"
installed_packages=$(conda run -n "$ENV_NAME" conda list)
if echo "$installed_packages" | grep -q "$PACKAGE_NAME"; then
  echo "MyXdriver_pub is already installed, skip installation"
else
  git clone https://github.com/lindsey98/MyXdriver_pub.git
  cd MyXdriver_pub
  chmod +x ./setup.sh
  ./setup.sh
  cd ../
fi

## Install other requirements
conda run -n "$ENV_NAME" pip install -r requirements.txt

# Install PaddleOCR
conda run -n "$ENV_NAME" pip install "paddleocr>=2.0.1"

# Install Image Captioning model
PACKAGE_NAME="lavis"
installed_packages=$(conda run -n "$ENV_NAME" conda list)
if echo "$installed_packages" | grep -q "$PACKAGE_NAME"; then
  echo "lavis is already installed, skip installation"
else
  git clone https://github.com/lindsey98/LAVIS.git
  cd LAVIS
  conda run -n "$ENV_NAME" pip install -e .
  cd ../
  rm -rf LAVIS
fi

# Download the ranking model
mkdir checkpoints
cd checkpoints
curl -L "https://huggingface.co/Kelsey98/ranking_model/download/resolve/main/epoch4_model.pt" -o "epoch4_model.pt"
echo "Model $MODEL_NAME downloaded successfully!"
