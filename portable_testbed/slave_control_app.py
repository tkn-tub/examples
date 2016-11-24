import logging
import subprocess

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class SlaveController(modules.ControlApplication):
    def __init__(self):
        super(SlaveController, self).__init__()
        self.log = logging.getLogger('SlaveController')
        self.running = False

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

        self.dut_iface = "eno1"
        self.bn_iface = "wlp2s0"

    def _run_command(self, command):
        sp = subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, shell=True)
        out, err = sp.communicate()

        if out:
            self.log.debug("standard output of subprocess:")
            self.log.debug(out)
        if err:
            self.log.debug("standard error of subprocess:")
            self.log.debug(err)

        return [sp.returncode, out.decode("utf-8"), err.decode("utf-8")]

    def _setup_l2_tunneling(self):
        self.log.info(
            "Create VXLAN (virtual) interface and add it to the Linux bridge")
        self._run_command(
            "ip link add vxlan10 type vxlan id 10 group 224.10.10.10 ttl 10 dev " + self.bn_iface)
        self._run_command("ip link set vxlan10 up")
        self._run_command("brctl addif br0 vxlan10")

        self.log.info("Set bigger MTU to fit packets from DUT nodes")
        self._run_command("ifconfig wlp2s0 mtu 1550")
        self._run_command("ifconfig vxlan10 mtu 1500")

    def _teardown_l2_tunneling(self):
        self.log.info("Destroy VXLAN (virtual) interface")
        self._run_command("brctl delif br0 vxlan10")
        self._run_command("ip link set vxlan10 down")
        self._run_command("ip link delete vxlan10")

        self.log.info("Set default MTU on wireless iface")
        self._run_command("ifconfig wlp2s0 mtu 1500")

    @modules.on_start()
    def my_start_function(self):
        print("Start Control Application")
        self.running = True
        self._setup_l2_tunneling()

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False
        self._teardown_l2_tunneling()

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")
        self.timer.start(self.timeInterval)
        return
