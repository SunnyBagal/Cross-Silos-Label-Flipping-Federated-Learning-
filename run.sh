#!/bin/bash

# Start server
python server.py &

# Start multiple clients (simulating hospitals)
sleep 5
HOSPITALS=("Hernandez Rogers and Vang," "White-White" "Nunez-Humphrey" "Sons and Miller" "Kim Inc" "Cook PLC")

for HOSP in "${HOSPITALS[@]}"; do
  HOSPITAL_NAME=$HOSP python client.py &
done
wait
