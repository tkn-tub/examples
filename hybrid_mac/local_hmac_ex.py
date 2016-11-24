import logging
import datetime

from sbi.wifi.params import HMACConfigParam, HMACAccessPolicyParam
from uniflex.core import modules

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"

'''
A local control program installing hMAC (TDMA) in Atheros based WiFi node.

Req.:
- supported Atheros wireless NIC
- hMAC patched Linux kernel (see README)
'''

class HybridMACLocalController(modules.ControlApplication):
    def __init__(self):
        super(HybridMACLocalController, self).__init__()
        self.log = logging.getLogger('HybridMACLocalController')

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start wifi ath hMAC test")

        try:
            node = self.localNode
            self.log.info(node)
            self.device = node.get_device(0)
            self.log.info(self.device)

            self.myHMACID = 'myHMACID'
            self.iface = 'ap1'
            total_slots = 10
            # slots are in microseonds
            slot_duration = 20000 # 20 ms
            # node on which scheme should be applied, e.g. nuc15 interface sta1
            dstHWAddr = "04:f0:21:17:36:68"

            # create new MAC for local node
            self.mac = HMACConfigParam(
                no_slots_in_superframe=total_slots,
                slot_duration_ns=slot_duration)

            be_slots = [1, 2, 3, 4]

            # assign access policies to each slot in superframe
            for slot_nr in range(total_slots):
                if slot_nr in be_slots:
                    acBE = HMACAccessPolicyParam()
                    acBE.addDestMacAndTosValues(dstHWAddr, 0)
                    self.mac.addAccessPolicy(slot_nr, acBE)
                else:
                    acGuard = HMACAccessPolicyParam()
                    acGuard.disableAll()  # guard slot
                    self.mac.addAccessPolicy(slot_nr, acGuard)

            # install configuration in MAC
            self.device.activate_radio_program(self.myHMACID, self.mac, self.iface)

        except Exception as e:
            self.log.error("{} Failed with install_mac_processor, err_msg: {}"
                           .format(datetime.datetime.now(), e))
            raise e

        self.log.info('... done')

    @modules.on_exit()
    def my_stop_function(self):
        self.log.info("stop wifi ath test")

        # install configuration in MAC
        self.device.deactivate_radio_program(self.myHMACID)
