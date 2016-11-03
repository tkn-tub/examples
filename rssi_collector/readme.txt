# 1. Run control program in master node:
uniflex-broker
# 2. Run control program in master node:
uniflex-agent --config ./config_master.yaml
# 3. Run modules in slave node:
uniflex-agent --config ./config_slave.yaml

# For debugging mode run with -v option
