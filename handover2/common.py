from uniflex.core import events

__author__ = "Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class CQIReportingEvent(events.EventBase):
    '''
    Events reported by each AP.
    '''
    def __init__(self, candidate_sigpower, curr_sigpower):
        super().__init__()
        self.candidate_sigpower = candidate_sigpower
        self.curr_sigpower = curr_sigpower


class DHCPEvent(events.EventBase):
    '''
    Base class for event reported by DHCP
    '''
    def __init__(self, mac_addr, ip_addr):
        super().__init__()
        self.mac_addr = mac_addr
        self.ip_addr = ip_addr


class DHCPNewEvent(DHCPEvent):
    '''
    Event reported by DHCP server for new leases
    '''
    def __init__(self, mac_addr, ip_addr):
        super().__init__(mac_addr, ip_addr)


class DHCPDelEvent(DHCPEvent):
    '''
    Event reported by DHCP server for deleted leases
    '''
    def __init__(self, mac_addr, ip_addr):
        super().__init__(mac_addr, ip_addr)
