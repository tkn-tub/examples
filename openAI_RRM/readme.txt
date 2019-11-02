# Start environment if Uniflex is installed in some
source ~/Uniflex/dev/bin/activate

# 2a. Run control program in master node:
uniflex-broker
# 2b. Run control program in master node:
python3 rl_agent.py --config ./config_master.yaml
# or
python3 rl_agent.py --config ./config_master_simulation.yaml
# you can choose rl_agent_multi.py, thompson_agent.py or thompson_agent2.py, too

# 2c. Run modules in slave node:
#Linux WiFi AP
uniflex-agent --config ./config_slave.yaml
#Simulation
uniflex-agent --config ./SimulationSlavesConfig/##Name of Experiment##/config_slave.yaml
uniflex-agent --config ./SimulationSlavesConfig/##Name of Experiment##/config_slave2.yaml
# and so on

# For debugging mode run with -v option
