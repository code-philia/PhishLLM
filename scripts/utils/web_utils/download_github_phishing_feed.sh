#!/bin/bash

cd datasets/
rm phishing-links-ACTIVE-TODAY.txt
rm phishing-links-ACTIVE-NOW.txt
rm phishing-links-ACTIVE.txt
rm phishing-links-NEW-last-hour.txt
rm phishing-links-NEW-today.txt

rm phishing-domains-NEW-today.txt
rm phishing-domains-NEW-last-hour.txt
rm phishing-domains-ACTIVE.txt

# Setting up retry variables
max_retries=5
retry_delay=10  # Time delay in seconds between retries

# Retry loop for wget
for (( i=1; i<=max_retries; i++ )); do
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-TODAY.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-NOW.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-last-hour.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-NEW-today.txt

    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-NEW-today.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-NEW-last-hour.txt
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt

    if [ $? -eq 0 ]; then
        echo "Download successful."
        break
    else
        echo "Download failed. Retrying in $retry_delay seconds... (Attempt $i of $max_retries)"
        sleep $retry_delay
    fi
done
cd ../