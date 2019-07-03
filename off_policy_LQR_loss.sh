#!/bin/sh
for value in {1..50}
do
    python3 off_policy_LQR_loss.py
done
echo ALL done
