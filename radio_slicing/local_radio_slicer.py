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
        self.update_interval = 10 # 1 sec

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
        self.total_slots = 30
        self.phy_to_data_factor = 0.4
        # slots are in microseonds
        slot_duration = 10000  # 20 ms

        sta1 = "00:15:6d:86:0f:84" #tv set, IP: 192.168.6.10
        sta2 = '00:16:ea:5f:2a:03' #internet radio, IP: 192.168.6.20
        sta3 = "ec:1f:72:82:09:56" #Mobile Phone youtube, IP 192.168.6.30
        sta4 = "00:15:6d:84:3c:12" #Guest Smartphone 1, IP: 192.168.7.10
        sta5 = "00:15:6d:84:3c:13" #Guest Smartphone 2, IP: 192.168.7.20
        self.min_rates = {sta1 : 10.0, sta2 : 1.0}
        self.min_rate_home_devices = 1.0
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
        primary_slots_exclusive = 0 #Slots used by primary devices getting exclusive slots
        needed_slots_standard_primary = 0 # Slots needed by standard primary users
        if True:
            if True:
                # step 1: get information about client STAs being served
                tx_bitrate_link = self.device.get_tx_bitrate_of_connected_devices(self.iface)
                for sta_mac_addr, sta_speed in tx_bitrate_link.items():
                    sta_tx_bitrate_val = float(sta_speed[0]) # e.g. 12
                    sta_tx_bitrate_unit = sta_speed[1] # e.g. Mbit/s

                    print(str(sta_mac_addr)+": STA_TX_BITRATE PHY: "+str(sta_tx_bitrate_val)+" "+str(sta_tx_bitrate_unit))
                    sta_tx_bitrate_val = sta_tx_bitrate_val * self.phy_to_data_factor
                    print(str(sta_mac_addr)+": STA_TX_BITRATE DATA: "+str(sta_tx_bitrate_val)+" "+str(sta_tx_bitrate_unit))

                    if sta_tx_bitrate_unit == "MBit/s":
                        self.phy_rates[sta_mac_addr] = sta_tx_bitrate_val #Store all PHY Rates of Primary STAs
                    else:
                        print("CONVERT THE BITRATE!!!!!")
                        # TODO write converter

                # step 2: process link info & decide on new slice sizes


                for sta_mac_addr in self.phy_rates:
                    slot_bitrate = float(self.phy_rates[sta_mac_addr]) / float(self.total_slots)
                    print("Slot Bitrate for STA: "+str(slot_bitrate))
                    if sta_mac_addr in self.min_rates.keys(): #If this primary device should get an exclusive slice
                        print("STA need to get explusive slice")
                        print("Policy Min Bitrate: "+str(self.min_rates[sta_mac_addr]))
                        number_of_slots = round(self.min_rates[sta_mac_addr] / slot_bitrate + 0.5)
                        self.min_slots[sta_mac_addr]= number_of_slots + 1 # plus 1 because of guard slots
                        primary_slots_exclusive = primary_slots_exclusive + number_of_slots 
                        print("Min Slots needed for exclusive primary: "+str(self.min_slots[sta_mac_addr]))
                    else: #Primary STA will get standard slot with other non exclusive primaries but trying to get min Rate per Primary
                        print("STA gets slice with other primaries")
                        print("Standard Min Bitrate for non-exclusive primaries: "+str(self.min_rate_home_devices))
                        number_of_slots = round(self.min_rate_home_devices / slot_bitrate + 0.5) + 1 # 1 for backup slots
                        print("Min Slots needed for non-exclusive primary: "+str(number_of_slots))
                        needed_slots_standard_primary = needed_slots_standard_primary + number_of_slots
                # step 3: update hMAC
                print("Exclusive primaries total slots: "+str(primary_slots_exclusive))
                print("Non exclusive primaries total slots: "+str(needed_slots_standard_primary))

                # assign access policies to each slot in superframe
                #Count needed slots for primary user
                total_used_slots_exclusive_primary = 0.0
                total_used_slots_non_exclusive_primary = 0.0
                total_used_slots = 0.0
                for sta_mac_addr in self.min_slots:
                    total_used_slots = total_used_slots + self.min_slots[sta_mac_addr]
                if total_used_slots > self.total_slots:
                    print("Warning!: More slots for exclusive primaries needed as available, target bitrate for all primary users not achievable! Needed: "+str(total_used_slots)+" Available: "+str(self.total_slots))
                    total_used_slots = self.total_slots
                else:
                    print("Total used slots for exclusive primaries: "+str(total_used_slots))
                total_used_slots_exclusive_primary = total_used_slots
                total_used_slots = total_used_slots + needed_slots_standard_primary
                if total_used_slots > self.total_slots:
                    print("Warning!: More slots for primaries needed as available, target bitrate for all primary users not achievable! Needed: "+str(total_used_slots)+" Available: "+str(self.total_slots))
                    total_used_slots = self.total_slots
                else:
                    print("Total used slots for primaries: "+str(total_used_slots))
                total_used_slots_non_exclusive_primary = total_used_slots - total_used_slots_exclusive_primary
                print("Total used slots for non-exclusive primaries: "+str(total_used_slots_non_exclusive_primary))
                check_exclusives = [0] * len(self.min_slots)
                for slot_nr in range(0,int(self.total_slots)):
                    print("Processing slot nr: "+str(slot_nr))
                    #ac_slot = self.mac.getAccessPolicy(slot_nr)
                    ac_slot = HMACAccessPolicyParam()
                    if slot_nr == 0:
                        print("Guard between secondaries and primaries")
                        ac_slot.disableAll()
                    elif slot_nr < total_used_slots_exclusive_primary:
                        sta_number = 0
                        for sta_mac_addr in self.min_slots:
                            if check_exclusives[sta_number] < self.min_slots[sta_mac_addr]:
                                ac_slot.addDestMacAndTosValues(sta_mac_addr, 0)
                                check_exclusives[sta_number] = check_exclusives[sta_number] + 1
                                if check_exclusives[sta_number] == self.min_slots[sta_mac_addr]:
                                    ac_slot.disableAll()
                                    print("Guard between primaries")
                                else: 
                                    print("Adding mac "+str(sta_mac_addr))
                                break
                            sta_number = sta_number + 1
                            #ac_slot.addDestMacAndTosValues(sta_mac_addr, 0)
                    #elif slot_nr == total_used_slots_exclusive_primary:
                        #ac_slot.disableAll()
                    elif slot_nr > total_used_slots_exclusive_primary and slot_nr < total_used_slots:
                        for sta_mac_addr in self.phy_rates:
                            if sta_mac_addr not in self.min_slots.keys():
                                print("Adding mac "+str(sta_mac_addr))
                                ac_slot.addDestMacAndTosValues(sta_mac_addr, 0)
                    elif slot_nr == total_used_slots:
                        ac_slot.disableAll()
                        print("Guard after primaries")
                    else:
                        #ac_slot.disableAll()
                        ac_slot.allowAll()
                        print("Secondaries")
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
                #self.device.update_mac_processor(self.iface, self.mac)
                self.device.update_radio_program(self.myHMACID, self.mac, self.iface)


        #except Exception as e:
        #    self.log.error("{} Failed updating mac processor, err_msg: {}"
        #                   .format(datetime.datetime.now(), e))
        #    raise e

        self.timer.start(self.update_interval)
