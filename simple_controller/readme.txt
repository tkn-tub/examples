# 1. Run control program and all modules on local node
uniflex-agent --config ./config_local.yaml

# 2a. Run control program in master node:
uniflex-broker
# 2b. Run control program in master node:
uniflex-agent --config ./config_master.yaml
# 2c. Run modules in slave node:
uniflex-agent --config ./config_slave.yaml

# For debugging mode run with -v option
