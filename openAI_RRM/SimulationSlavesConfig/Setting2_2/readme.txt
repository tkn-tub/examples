# Start environment if Uniflex is installed in some
source ~/Uniflex/dev/bin/activate

# 2a. Run control program in master node:
uniflex-broker
# 2b. Run control program in master node:
python3 rl_agent.py --config ./config_master_simulation.yaml
# you can choose thompson_agent.py or thompson_agent2.py, too

# 2c. Run modules in slave node:
#Simulation
uniflex-agent --config ./config_slave.yaml
uniflex-agent --config ./config_slave2.yaml

# For debugging mode run with -v option
