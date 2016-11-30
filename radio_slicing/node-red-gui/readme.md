# Install node red:
   
    sudo apt-get update
    apt-get install libzmq5-dev
    sudo apt-get install build-essential
    sudo apt-get install nodejs
    #sometimes needed:
    #sudo apt install nodejs-legacy
    sudo apt-get install npm
    sudo npm install -g node-gyp
    sudo npm install -g --unsafe-perm node-red

# Install additional nodes:

    cd $HOME/.node-red
    sudo npm install zmq
    sudo npm install uniflex/node-red-uniflex
    sudo npm install node-red-node-smooth

# Run example flow graph - moving average filter:

    cd ./examples/node_red
    node-red gui.json

# Run uniflex-agent with GUI Feeder app:

    uniflex-agent --config ./config.yaml

# For debugging mode run with -v option


