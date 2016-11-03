UniFlex IEEE 802.11 handover example
====================================

# 0. Requirements

# install libtins on each AP node

    sudo apt-get install libboost-all-dev
    git clone https://github.com/mfontanini/libtins.git
    cd libtins/
    mkdir build
    cd build/
    cmake ../
    make
    sudo make install

# build external scanner daemon
    cd scanner
    cd build/
    cmake ../
    make

# install DHCP server
https://github.com/flan/staticdhcpd

plus add patches ... tbd

# 1. Running

## 1.1 start local UniFlex agent on each AP:

    sudo ../../dev/bin/uniflex-agent --config config_ap.yaml

## 1.2 start local UniFlex agent on the Gateway node (Linux):

    sudo ../../dev/bin/uniflex-agent --config config_gw.yaml

## 1.3 start global UniFlex controller (BigAP):

    uniflex-agent --config config_master.yaml

# 2. How to reference to?

Just use the following bibtex:

    @INPROCEEDINGS{7502842, 
    author={A. Zubow and S. Zehl and A. Wolisz}, 
    booktitle={NOMS 2016 - 2016 IEEE/IFIP Network Operations and Management Symposium}, 
    title={BIGAP #x2014; Seamless handover in high performance enterprise IEEE 802.11 networks}, 
    year={2016}, 
    pages={445-453}, 
    keywords={business communication;cloud computing;mobility management (mobile radio);quality of experience;quality of service;radio spectrum management;resource allocation;BIGAP;MAC-layer handover;QoE;QoS;cloud storage;dynamic frequency selection capability;handover operation;high network performance;high performance enterprise IEEE 802.11 networks;load balancing;mobile HD video;mobility support;network outage duration;radio spectrum;seamless handover;seamless mobility;wireless clients;Handover;IEEE 802.11 Standard;Load management;Radio frequency;Switches;Wireless communication;Handover;Mobility;SDN;Wireless}, 
    doi={10.1109/NOMS.2016.7502842}, 
    month={April},}
