import logging
import datetime
from uniflex.core import events
from sbi.wifi.params import HMACConfigParam, HMACAccessPolicyParam
from uniflex.core import modules
from uniflex.core.timer import TimerEventSender

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"

'''
A local controller program running on WiFi AP performing radio slicing (based on hMAC).
The objective is to isolate the primary users (PUs) from the secondaries (SUs) which are
being served by that WiFi AP.


Req.:
- supported Atheros wireless NIC
- hMAC patched Linux kernel (see README)
'''

class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class LocalRadioSlicer(modules.ControlApplication):
    def __init__(self):
        super(LocalRadioSlicer, self).__init__()
        self.log = logging.getLogger('LocalRadioSlicer')
        self.update_interval = 1 # 1 sec

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start wifi radio slicer")

        # store object referenes
        node = self.localNode
        self.log.info(node)
        self.device = node.get_device(0)
        self.log.info(self.device)

        self.myHMACID = 'RadioSlicerID'
        self.iface = 'ap1'
        total_slots = 10
        # slots are in microseonds
        slot_duration = 20000  # 20 ms

        # create new MAC for local node
        self.mac = HMACConfigParam(
            no_slots_in_superframe=total_slots,
            slot_duration_ns=slot_duration)

        # assign allow all to each slot
        for slot_nr in range(total_slots):
                acGuard = HMACAccessPolicyParam()
                acGuard.allowAll()  # allow all
                self.mac.addAccessPolicy(slot_nr, acGuard)

        # install configuration in MAC
        self.device.activate_radio_program(self.myHMACID, self.mac, self.iface)

        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.update_interval)

        self.log.info('... done')


    @modules.on_exit()
    def my_stop_function(self):
        self.log.info("stop wifi radio slicer")

        # install configuration in MAC
        self.device.deactivate_radio_program(self.myHMACID)


    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_slice_adapdation(self, event):
        print("Periodic slice adaptations ...")

        try:
            # TODO: enable me!!!
            if False:
                # step 1: get information about client STAs being served
                tx_bitrate_link = self.device.get_tx_bitrate_of_connected_devices(self.iface)
                for sta_mac_addr, sta_speed in tx_bitrate_link.items():
                    sta_tx_bitrate_val = sta_speed[0] # e.g. 12
                    sta_tx_bitrate_unit = sta_speed[1] # e.g. Mbit/s

                    # TODO: sven do something ...
                    # mac_addr -> (rate, unit)
                    pass

                # step 2: process link info & decide on new slice sizes
                # TODO: sven do something ...


                # step 3: update hMAC

                # TODO: adapt hMAC config
                # assign access policies to each slot in superframe
                for slot_nr in range(self.mac.getNumSlots()):
                    ac_slot = self.mac.getAccessPolicy(slot_nr)
                    ac_slot.disableAll()
                    # TODO: sven do something ...
                    # node on which scheme should be applied, e.g. nuc15 interface sta1
                    staDstHWAddr = "04:f0:21:17:36:68"
                    ac_slot.addDestMacAndTosValues(staDstHWAddr, 0)

                # update configuration in hMAC
                self.device.update_radio_program(self.myHMACID, self.mac, self.iface)

        except Exception as e:
            self.log.error("{} Failed updating mac processor, err_msg: {}"
                           .format(datetime.datetime.now(), e))
            raise e

        self.timer.start(self.update_interval)
