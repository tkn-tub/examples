# Start environment if Uniflex is installed in some
source ~/Uniflex/dev/bin/activate

# 2a. Run control program in master node:
uniflex-broker
# 2b. Run control program in master node:
python3 rrm_agent.py --config ./config_master_simulation.yaml

# 2c. Run modules in slave node:
#Simulation
uniflex-agent --config ./config_slave.yaml
uniflex-agent --config ./config_slave2.yaml
uniflex-agent --config ./config_slave3.yaml

# For debugging mode run with -v option
