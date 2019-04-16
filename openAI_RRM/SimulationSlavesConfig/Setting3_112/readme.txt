# 1. Run control program and all modules on local node
uniflex-agent --config ./config_local.yaml

source ~/Uniflex/dev/bin/activate

# 2a. Run control program in master node:
uniflex-broker
# 2b. Run control program in master node:
python3 rrm_agent.py --config ./config_master.yaml
# 2c. Run modules in slave node:
uniflex-agent --config ./config_slave.yaml
uniflex-agent --config ./config_slave2.yaml
uniflex-agent --config ./config_slave3.yaml

# For debugging mode run with -v option
