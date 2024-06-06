install_dir="/tmp/"

function success {
	echo "[+] $1"
}

function fail {
	echo "[-] $1"
}

function warn {
	echo "[!] $1"
}

function prompt {
	ret=""
	while true; do
    	read -p "$1 [y/n]: " yn
	    case $yn in
	        [Yy]* ) ret=1; break;;
	        [Nn]* ) ret=0; break;;
	        * ) echo "Please answer yes or no.";;
    	esac
	done
	return $ret
}

function install_chrome {
	debfile="google-chrome-stable_current_amd64.deb"
	wget "https://dl.google.com/linux/direct/$debfile" -P "$install_dir"
	if [ $? -ne 0 ];
	then
		fail "Could not download Chrome"
		return 1
	fi
	sudo dpkg -i "$install_dir$debfile"
	if [ $? -ne 0 ];
	then
		fail "Could not install Chrome package"
		return 2
	fi
	success "Successfully installed Chrome"
	return 0
}


declare -A browsers
browsers=(["google-chrome-stable"]=install_chrome)

function check_browsers {
	for browser in ${!browsers[@]};
	do
		installed=false
		sudo dpkg -l "$browser" > /dev/null 2>&1
		if [ $? -eq 0 ];
		then
			success "$browser is installed. (version: $($browser --version))"
			installed=true
		else
			warn "$browser does not seem to be installed"
			prompt "Do you want to install its latest stable version?"
			if [ $? -eq 1 ];
			then
				success "Installing $browser"
				${browsers[$browser]}
				if [ $? -eq 0 ];
				then
					installed=true
				fi
			else
				fail "Skipping $browser installation"
			fi
		fi
		echo -e ""
	done
	return 0
}

# Install chrome binary
#check_browsers

# Create a new conda environment with Python 3.8
# Check if the environment already exists
conda info --envs | grep -w "$ENV_NAME" > /dev/null
if [ $? -eq 0 ]; then
    echo "Activating Conda environment $ENV_NAME"
else
    echo "Creating and activating new Conda environment $ENV_NAME with Python 3.8"
    conda create -n "$ENV_NAME" python=3.8
fi

PACKAGE_NAME="phishintention"
if conda list -n "$ENV_NAME" | grep -q "$PACKAGE_NAME"; then
    echo "$PACKAGE_NAME is already installed, skip installation"
elif [ -d "PhishIntention" ]; then
    echo "Directory PhishIntention already exists, skip cloning"
    cd PhishIntention
    chmod +x ./setup.sh
    export ENV_NAME="$ENV_NAME" && ./setup.sh
    cd ../
    rm -rf PhishIntention
else
    git clone -b development --single-branch https://github.com/lindsey98/PhishIntention.git
    cd PhishIntention
    chmod +x ./setup.sh
    export ENV_NAME="$ENV_NAME" && ./setup.sh
    cd ../
    rm -rf PhishIntention
fi

# Install PaddleOCR
if command -v nvcc &> /dev/null; then
  # cuda is available
  conda run -n "$ENV_NAME" pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
else # cpu-only
  conda run -n "$ENV_NAME" pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
conda run -n "$ENV_NAME" pip install "paddleocr>=2.0.1"

# Install Image Captioning model
PACKAGE_NAME="lavis"
if conda list -n "$ENV_NAME" | grep -q "$PACKAGE_NAME"; then
    echo "$PACKAGE_NAME is already installed, skip installation"
else
    git clone https://github.com/lindsey98/LAVIS.git
    cd LAVIS
    conda run -n "$ENV_NAME" pip install -e .
    cd ../
    rm -rf LAVIS
fi

## Install other requirements
if [ -f requirements.txt ]; then
    conda run -n "$ENV_NAME" pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping additional package installations."
fi

# Download the ranking model
mkdir -p checkpoints
cd checkpoints
conda run -n "$ENV_NAME" pip install gdown
conda run -n "$ENV_NAME" gdown --id 1bpy-SRDOkL96j9r3ErBd7L5mDUdLAWaU
