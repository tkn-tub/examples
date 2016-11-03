# Install node red:

    curl -sL https://deb.nodesource.com/setup_0.10 | sudo -E bash -
    sudo apt-get install -y nodejs
    sudo apt-get install -y build-essential
    sudo npm install -g --unsafe-perm node-red

# Install additional nodes:

    cd $HOME/.node-red
    sudo npm install zmq
    sudo npm install uniflex/node-red-uniflex
    sudo npm install node-red-node-smooth

# Run example flow graph - moving average filter:

    cd ./examples/node_red
    node-red my_filter.json

![my_filter](./my_filter.png)

# Run uniflex-agent with master config:

    uniflex-agent --config ./config_master.yaml

# Run uniflex-agent with slave config:

    uniflex-agent --config ./config_slave.yaml

# For debugging mode run with -v option


