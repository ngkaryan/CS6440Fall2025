#!/usr/bin/env bash
set -uo pipefail

echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq openjdk-11-jdk-headless curl

echo "Downloading Synthea..."
curl -L -o synthea.jar https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

echo "Generating 10 patients (~2–3 minutes)..."
rm -rf synthea_output output
java -jar synthea.jar -p 10 Massachusetts
mv output synthea_output


echo "Build finished – index will be created on first app start"
