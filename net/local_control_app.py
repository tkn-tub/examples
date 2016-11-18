import logging
import datetime
import random
from uniflex.core import modules

__author__ = "Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"

"""
    Just simple local test.
"""
class MyNetController(modules.ControlApplication):
    def __init__(self):
        super(MyNetController, self).__init__()
        self.log = logging.getLogger('MyNetController')

    @modules.on_start()
    def my_start_function(self):
        print("start control app")

        node = self.localNode
        self.log.info("My local node: {}, Local: {}"
                      .format(node.hostname, node.local))

        for dev in node.get_devices():
            print("Dev: ", dev.name)

        for apps in node.get_control_applications():
            print("App: ", apps.name)

        # search for NetworkModule
        net_proto = None
        for m in node.get_modules():
            print("Module: %s, %s" % (m.name, m.uuid))
            if m.name == 'NetworkModule':
                net_proto = m

        iface = 'ens33'
        print(net_proto.get_iface_ip_addr(iface))
        print(net_proto.get_iface_hw_addr(iface))

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
