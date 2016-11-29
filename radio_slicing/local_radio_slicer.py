import logging
import datetime
from uniflex.core import events
from sbi.wifi.params import HMACConfigParam, HMACAccessPolicyParam
from uniflex.core import modules
from uniflex.core.timer import TimerEventSender

__author__ = "Anatolij Zubow, Sven Zehl"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow, zehl}@tkn.tu-berlin.de"

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
        self.iface = 'ap5'
        self.total_slots = 20
        # slots are in microseonds
        slot_duration = 20000  # 20 ms
        
        sta1 = "00:15:6d:86:0f:88" #fernseher
        self.min_rates = {sta1 : 15.0}
        self.phy_rates = {}
        self.min_slots = {}


        # create new MAC for local node
        self.mac = HMACConfigParam(
            no_slots_in_superframe=self.total_slots,
            slot_duration_ns=slot_duration)

        # assign allow all to each slot
        for slot_nr in range(self.total_slots):
                acGuard = HMACAccessPolicyParam()
                acGuard.allowAll()  # allow all
                self.mac.addAccessPolicy(slot_nr, acGuard)
        self.mac.printConfiguration()
        # install configuration in MAC
        #self.device.install_mac_processor(self.iface, self.mac)
        self.device.activate_radio_program(self.myHMACID, self.mac, self.iface)

        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.update_interval)

        self.log.info('... done')


    @modules.on_exit()
    def my_stop_function(self):
        self.log.info("stop wifi radio slicer")

        # install configuration in MAC
        #self.device.uninstall_mac_processor(self.iface, self.mac)
        self.device.deactivate_radio_program(self.myHMACID)

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_slice_adapdation(self, event):
        print("Periodic slice adaptations ...")
        if True:
            if True:
                # step 1: get information about client STAs being served
                tx_bitrate_link = self.device.get_tx_bitrate_of_connected_devices(self.iface)
                for sta_mac_addr, sta_speed in tx_bitrate_link.items():
                    sta_tx_bitrate_val = float(sta_speed[0]) # e.g. 12
                    sta_tx_bitrate_unit = sta_speed[1] # e.g. Mbit/s
                    print(str(sta_mac_addr)+": STA_TX_BITRATE PHY: "+str(sta_tx_bitrate_val)+" "+str(sta_tx_bitrate_unit))
                    sta_tx_bitrate_val = sta_tx_bitrate_val * 0.85
                    print(str(sta_mac_addr)+": STA_TX_BITRATE DATA: "+str(sta_tx_bitrate_val)+" "+str(sta_tx_bitrate_unit))
                    if sta_tx_bitrate_unit == "MBit/s":
                        self.phy_rates[sta_mac_addr] = sta_tx_bitrate_val
                    else:
                        print("CONVERT THE BITRATE!!!!!")
                        # TODO write converter

                # step 2: process link info & decide on new slice sizes
                
                for sta_mac_addr in self.phy_rates:
                    if sta_mac_addr in self.min_rates.keys():
                        print("STA is in min_rates")
                        slot_bitrate = float(self.phy_rates[sta_mac_addr]) / float(self.total_slots)
                        print("Slot Bitrate for STA: "+str(slot_bitrate))
                        print("Policy Min Bitrate: "+str(self.min_rates[sta_mac_addr]))
                        number_of_slots = round(self.min_rates[sta_mac_addr] / slot_bitrate + 0.5) 
                        self.min_slots[sta_mac_addr]= number_of_slots
                        print("Min Slots needed: "+str(self.min_slots[sta_mac_addr]))                     

                # step 3: update hMAC

                # assign access policies to each slot in superframe
                #Count needed slots for primary user
                total_used_slots = 0.0
                for sta_mac_addr in self.min_slots:
                    total_used_slots = total_used_slots + self.min_slots[sta_mac_addr]                    
                if total_used_slots > self.total_slots:
                    print("Warning!: More slots needed as available, target bitrate for all primary users not achievable! Needed: "+str(total_used_slots)+" Available: "+str(self.total_slots))
                    total_used_slots = self.total_slots            
                else:
                    print("Total used slots for primary: "+str(total_used_slots)) 
                self.mac.printConfiguration()
                for slot_nr in range(0,int(total_used_slots)):
                    print("Processing primary slot nr: "+str(slot_nr))  
                    #ac_slot = self.mac.getAccessPolicy(slot_nr)
                    ac_slot = HMACAccessPolicyParam()
                    for sta_mac_addr in self.min_slots:
                        print("Adding mac "+str(sta_mac_addr))
                        ac_slot.addDestMacAndTosValues(sta_mac_addr, 0)
                    self.mac.addAccessPolicy(slot_nr, ac_slot)
                #for slot_nr in range(int(total_used_slots), int(self.total_slots)):
                    # TODO Replace this with STAs of Guest Network
                #    print("Processing secondary slot nr: "+str(slot_nr))
                #    ac_slot = self.mac.getAccessPolicy(slot_nr)
                #    ac_slot = ac_slot.disableAll()
                    #self.mac.addAccessPolicy(slot_nr, ac_slot)
                    # TODO: sven do something ...
                    # node on which scheme should be applied, e.g. nuc15 interface sta1
                #    staDstHWAddr = "04:f0:21:17:36:68"
                #    ac_slot.addDestMacAndTosValues(staDstHWAddr, 0)

                # update configuration in hMAC
                self.mac.printConfiguration()
                self.device.update_radio_program(self.myHMACID, self.mac, self.iface)
                #self.device.update_mac_processor(self.iface, self.mac)

        #except Exception as e:
        #    self.log.error("{} Failed updating mac processor, err_msg: {}"
        #                   .format(datetime.datetime.now(), e))
        #    raise e

        self.timer.start(self.update_interval)
